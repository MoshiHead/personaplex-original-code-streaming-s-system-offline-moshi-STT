"""
train_rag.py
============
Fine-tuning script for RAG-aware Moshi LM.

What this does
--------------
1.  Loads the pretrained LMModel (Moshi) and Mimi codec.
2.  Adds a lightweight ReferenceAdapter — a small MLP that projects
    mean-pooled reference text embeddings (from the LM's own text_emb
    table) into a [dim]-dimensional conditioning vector.  This is the
    equivalent of Kyutai's ARC encoder, built entirely from existing
    model components, requiring no external service.
3.  Applies LoRA to the main transformer and depformer inside LMModel.
4.  Enables gradient checkpointing on the transformer blocks.
5.  Freezes Mimi entirely, freezes base LM weights (only LoRA deltas
    + ReferenceAdapter train).
6.  Trains with the per-segment loss from rag_loss.py:
      - 5x weight on <ref> token frame
      - 0.5x on filler text tokens
      - 2x on body text tokens (grounding signal)
      - contrastive margin loss on body frames (wrong reference)
7.  Saves checkpoints every N steps.

ReferenceAdapter — why no external encoder
------------------------------------------
The original Moshi-RAG system uses Kyutai's ARC encoder (a remote HTTP
service) to convert reference text → [T, dim] conditioning tensors.
We cannot use ARC because:
  a) It is not open-sourced.
  b) Its output distribution is co-trained with the production model;
     injecting ARC vectors into a freshly fine-tuned model would require
     the model to already understand those vectors.

Instead we build conditioning vectors directly from the LM's own
text_emb table, which is guaranteed to be in-distribution:

    reference text
         │
    text_tokenizer.encode()
         │
    lm.text_emb(token_ids)     # [T_ref, dim] — same table the LM uses
         │
    mean_pool                  # [dim]
         │
    ReferenceAdapter (MLP)     # [dim] → [dim]  (2 layers, trained)
         │
    broadcast to [B, 1, dim]
         │
    added to embed_codes() output every frame in the body segment

The adapter has ~2M parameters for dim=4096, trains quickly, and the
LM learns to use its output through the body-segment grounding loss.

Usage
-----
    python train_rag.py \
        --train-data   data/train.jsonl \
        --eval-data    data/eval.jsonl \
        --moshi-weight /path/to/moshi.safetensors \
        --mimi-weight  /path/to/mimi.safetensors \
        --tokenizer    /path/to/tokenizer.model \
        --out-dir      checkpoints/rag_lora \
        --device       cuda

    # Resume from checkpoint
    python train_rag.py ... --resume checkpoints/rag_lora/step_1000

Hyper-parameters (matching provided config)
--------------------------------------------
    --lora-rank 128  --lora-scaling 2.0
    --lr 2e-6  --weight-decay 0.1
    --batch-size 16  --max-steps 2000
    --duration-sec 100
    --first-codebook-weight 100.0  --text-padding-weight 0.5
    --ref-token-weight 5.0  --body-weight 2.0  --filler-weight 0.5
    --contrastive-weight 0.3  --contrastive-margin 0.3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from functools import partial

import numpy as np
import sentencepiece
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from huggingface_hub import hf_hub_download

# Project imports — same package as offline inference
from .models import loaders
from .models.lm import LMModel

# Our pipeline modules
from .rag_loss import compute_rag_losses, SegmentMeta, make_segment_weight_tensor
from .dataset import build_data_loader
from .interleaver import InterleavedTokenizer, Interleaver, Batch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reference adapter — replaces ARC encoder
# ---------------------------------------------------------------------------

class ReferenceAdapter(nn.Module):
    """
    Converts a reference text string into a [dim]-dimensional conditioning
    vector that is added to the LM's input embeddings during body frames.

    Architecture
    ------------
    text_emb(ref_tokens) → mean_pool → LayerNorm → Linear(dim, dim*2)
    → GELU → Linear(dim*2, dim) → LayerNorm

    The two LayerNorms keep activations in a stable range so the adapter
    output is compatible with the transformer's input scale from day one.

    Parameters: ~2 × dim² ≈ 2M params for dim=1024, 34M for dim=4096.

    Why mean-pool over the sequence?
    The body segment is conditioned on one persistent reference vector,
    not a per-token sequence.  Mean pooling over all reference token
    embeddings produces a dense semantic summary.  For longer references
    (>200 tokens) this loses some detail, but the grounding loss trains
    the adapter to retain the most answer-relevant content.
    """

    def __init__(self, dim: int, text_emb: nn.Embedding):
        super().__init__()
        self.text_emb  = text_emb   # shared with LM — not an extra copy
        self.norm_in   = nn.LayerNorm(dim)
        self.fc1       = nn.Linear(dim, 1024, bias=True)
        self.fc2       = nn.Linear(1024, dim, bias=True)
        # Learned output scale — starts at 0.01 so the adapter contributes
        # a tiny signal initially and grows as training needs it.
        # Avoids the zero-gradient trap: LayerNorm(zeros) has near-zero
        # gradient w.r.t. upstream params if the final layer outputs zeros.
        self.out_scale = nn.Parameter(torch.tensor(0.01))
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [T_ref] int64 — tokenised reference text

        Returns:
            [dim] float — conditioning vector, one per reference string
        """
        token_ids = token_ids.clamp(0, self.text_emb.num_embeddings - 1)
        embeds = self.text_emb(token_ids.unsqueeze(0))  # [1, T_ref, dim]
        pooled = embeds.mean(dim=1).squeeze(0)           # [dim]
        x = self.norm_in(pooled)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x * self.out_scale                        # [dim]


    def encode_batch(
        self,
        references: list[str],
        tokenizer: sentencepiece.SentencePieceProcessor,
        device: torch.device,
        max_tokens: int = 256,
    ) -> torch.Tensor:
        """
        Encode a list of reference strings → [B, dim] conditioning matrix.
        Empty strings (non-RAG examples) produce zero vectors.

        Args:
            references : list of B reference strings ('' for non-RAG slots)
            tokenizer  : sentencepiece tokenizer
            device     : target device
            max_tokens : cap reference length to avoid OOM on long passages

        Returns:
            [B, dim] float tensor
        """
        B   = len(references)
        dim = self.fc2.out_features
        out = torch.zeros(B, dim, device=device)

        for i, ref in enumerate(references):
            if not ref:
                continue   # non-RAG slot stays zero
            ids = tokenizer.encode(ref)[:max_tokens]
            if not ids:
                continue
            id_tensor = torch.tensor(ids, dtype=torch.long, device=device)
            out[i] = self.forward(id_tensor)

        return out   # [B, dim]


# ---------------------------------------------------------------------------
# LoRA application
# ---------------------------------------------------------------------------

def apply_lora(
    lm: LMModel,
    rank: int = 128,
    lora_alpha: float = 256.0,   # = rank * scaling (128 * 2.0)
    target_modules: list[str] | None = None,
    ft_embed: bool = False,
) -> LMModel:
    """
    Apply LoRA to the main transformer and depformer inside LMModel.

    We use peft's get_peft_model which wraps in-place.  LoRA is applied to
    all nn.Linear layers whose names match target_modules.

    Default targets: in_proj, out_proj, fc1, fc2 — the attention and FFN
    projections inside StreamingTransformer.  Adjust if your StreamingTransformer
    uses different layer names.

    ft_embed=False (default) keeps text_emb and emb frozen, matching the
    provided config.
    """
    from peft import LoraConfig, get_peft_model, TaskType

    if target_modules is None:
        # These are the standard names inside a Transformer block.
        # Inspect lm.transformer with print(lm.transformer) if names differ.
        target_modules = [
            "in_proj", "out_proj",   # attention
            "fc1", "fc2",            # feedforward
            "linear", "proj",        # catch-all for variants
        ]

    config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        # TaskType.CAUSAL_LM is a hint to peft; doesn't change behaviour for
        # custom models but avoids warnings
        task_type=TaskType.FEATURE_EXTRACTION,
        modules_to_save=["text_linear"] if not ft_embed else ["text_linear", "text_emb"],
    )

    lm = get_peft_model(lm, config)
    lm.print_trainable_parameters()
    return lm


# ---------------------------------------------------------------------------
# Gradient checkpointing
# ---------------------------------------------------------------------------

def enable_gradient_checkpointing(lm: nn.Module) -> None:
    """
    Enable gradient checkpointing on the transformer layers inside LMModel.

    peft's get_peft_model exposes enable_input_require_grads which is needed
    for gradient checkpointing to work when inputs don't have requires_grad.
    Then we enable checkpointing on the underlying transformer modules.
    """
    # peft wrapper
    if hasattr(lm, "enable_input_require_grads"):
        lm.enable_input_require_grads()

    # Enable on the two StreamingTransformer modules
    base = lm.base_model if hasattr(lm, "base_model") else lm
    for name in ("transformer", "depformer"):
        module = getattr(base, name, None)
        if module is not None and hasattr(module, "gradient_checkpointing_enable"):
            module.gradient_checkpointing_enable()
            logger.info(f"gradient checkpointing enabled on lm.{name}")
        elif module is not None:
            # StreamingTransformer may not expose the method directly;
            # wrap its layers manually
            if hasattr(module, "layers"):
                module.gradient_checkpointing = True
                logger.info(f"set gradient_checkpointing=True on lm.{name}")


# ---------------------------------------------------------------------------
# Streaming sum injection — equivalent of apply_pending_streaming_sum_condition
# ---------------------------------------------------------------------------

def inject_streaming_sum(
    embed_codes_output: torch.Tensor,   # [B, T, dim]
    streaming_sum: torch.Tensor,        # [B, dim]
    seg_metas: list[SegmentMeta | None],
    T: int,
) -> torch.Tensor:
    """
    Add the reference conditioning vector to embed_codes output at body frames.

    streaming_sum[b] is added to every frame from body_start onward for batch
    element b.  Non-RAG elements (seg_meta=None) or elements with zero
    streaming_sum (reference='') are unaffected.

    This is the forward-pass equivalent of:
        apply_pending_streaming_sum_condition()  (System 2, BatchRunner)

    Shape: embed_codes_output [B, T, dim] is modified in-place and returned.
    """
    B, _, dim = embed_codes_output.shape
    for b, meta in enumerate(seg_metas):
        if meta is None:
            continue
        body_start = meta.body_start
        if body_start >= T:
            continue
        # streaming_sum[b]: [dim] → broadcast to [T-body_start, dim]
        cond = streaming_sum[b].unsqueeze(0)   # [1, dim]
        embed_codes_output[b, body_start:] = (
            embed_codes_output[b, body_start:] + cond
        )
    return embed_codes_output

def embed_codes_split(lm: LMModel, sequence: torch.Tensor):
    """Same math as LMModel.embed_codes, but keeps audio-sum and text_emb separate."""
    audio_sum = None
    for cb_index in range(lm.num_audio_codebooks):
        audio_emb = lm.emb[cb_index](sequence[:, cb_index + lm.audio_offset])
        audio_sum = audio_emb if audio_sum is None else audio_sum + audio_emb
    text_emb = lm.text_emb(sequence[:, 0])
    return audio_sum, text_emb


def inject_streaming_sum_text_channel(text_emb, streaming_sum, seg_metas, T):
    for b, meta in enumerate(seg_metas):
        if meta is None:
            continue
        if meta.body_start >= T:
            continue
        text_emb[b, meta.body_start:] = text_emb[b, meta.body_start:] + streaming_sum[b].unsqueeze(0)
    return text_emb
# ---------------------------------------------------------------------------
# Patched forward pass
# We monkey-patch LMModel.forward_train to accept streaming_sum
# without modifying the original model source.
# ---------------------------------------------------------------------------

def forward_train_with_streaming_sum(
    self: LMModel,
    codes: torch.Tensor,
    streaming_sum: torch.Tensor | None = None,
    seg_metas: list[SegmentMeta | None] | None = None,
):
    """
    Drop-in replacement for LMModel.forward_train that injects streaming_sum.

    Identical to the original except embed_codes() output is modified by
    inject_streaming_sum() before being passed to forward_embeddings().

    Attached to LMModel instances at training time; not persistent.
    """
    from .models.lm import _delay_sequence, _undelay_sequence

    B, K, T = codes.shape
    initial  = self._get_initial_token().expand(B, -1, -1)
    delayed  = _delay_sequence(self.delays, codes, initial)
    delayed  = torch.cat([initial, delayed], dim=2)

    audio_sum, text_emb = embed_codes_split(self, delayed[:, :, :-1])
    if streaming_sum is not None and seg_metas is not None:
        text_emb = inject_streaming_sum_text_channel(text_emb, streaming_sum, seg_metas, T)
                                                    
    embedded = audio_sum + text_emb

    # --- forward through transformer + depformer (same as original) ---
    transformer_out, text_logits = self.forward_embeddings(embedded)
    logits = self.forward_depformer_training(delayed[:, :, 1:], transformer_out)

    logits, logits_mask = _undelay_sequence(
        self.delays[self.audio_offset : self.audio_offset + self.dep_q],
        logits,
        fill_value=float("NaN"),
    )
    logits_mask &= (
        codes[:, self.audio_offset : self.audio_offset + self.dep_q]
        != self.zero_token_id
    )
    text_logits, text_logits_mask = _undelay_sequence(
        self.delays[:1], text_logits, fill_value=float("NaN")
    )
    text_logits_mask &= codes[:, :1] != self.zero_token_id

    from .models.lm import LMOutput
    return LMOutput(logits, logits_mask, text_logits, text_logits_mask)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(
    out_dir: Path,
    step: int,
    lm: nn.Module,
    adapter: ReferenceAdapter,
    optimizer: torch.optim.Optimizer,
    scheduler,
    stats: dict,
) -> None:
    ckpt_dir = out_dir / f"step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA weights (peft saves only the trainable delta, not the full model)
    if hasattr(lm, "save_pretrained"):
        lm.save_pretrained(ckpt_dir / "lora")
    else:
        torch.save(lm.state_dict(), ckpt_dir / "lm_state.pt")

    # Save adapter
    torch.save(adapter.state_dict(), ckpt_dir / "adapter.pt")

    # Save optimizer + scheduler
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step,
        "stats": stats,
    }, ckpt_dir / "optim.pt")

    logger.info(f"[ckpt] saved step {step} → {ckpt_dir}")


def load_checkpoint(
    ckpt_dir: Path,
    lm: nn.Module,
    adapter: ReferenceAdapter,
    optimizer: torch.optim.Optimizer,
    scheduler,
) -> int:
    """Load from checkpoint directory, return the step number."""
    # Load LoRA weights
    lora_path = ckpt_dir / "lora"
    if lora_path.exists() and hasattr(lm, "load_adapter"):
        lm.load_adapter(str(lora_path), adapter_name="default")
        logger.info(f"[ckpt] loaded LoRA from {lora_path}")
    elif (ckpt_dir / "lm_state.pt").exists():
        lm.load_state_dict(torch.load(ckpt_dir / "lm_state.pt"))

    # Load adapter
    adapter_path = ckpt_dir / "adapter.pt"
    if adapter_path.exists():
        adapter.load_state_dict(torch.load(adapter_path))
        logger.info(f"[ckpt] loaded adapter from {adapter_path}")

    # Load optimizer
    optim_path = ckpt_dir / "optim.pt"
    if optim_path.exists():
        state = torch.load(optim_path)
        optimizer.load_state_dict(state["optimizer"])
        if scheduler and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        step = state["step"]
        logger.info(f"[ckpt] resumed from step {step}")
        return step

    return 0


# ---------------------------------------------------------------------------
# Wrong-reference construction for contrastive loss
# ---------------------------------------------------------------------------

def build_wrong_references(
    references: list[str],
    seg_metas: list[SegmentMeta | None],
    rng: random.Random,
) -> list[str]:
    """
    For each RAG slot, swap its reference with a random other slot's reference.
    Non-RAG slots (meta=None or ref='') get empty string.

    The contrastive loss only fires on body frames, so wrong refs for non-RAG
    slots produce zero gradient regardless.
    """
    rag_refs = [
        (i, references[i])
        for i, m in enumerate(seg_metas)
        if m is not None and references[i]
    ]
    if len(rag_refs) < 2:
        # Can't swap with only one RAG example — return empty (skip contrastive)
        return [""] * len(references)

    wrong = list(references)
    for i, ref in rag_refs:
        # Pick a different RAG slot's reference
        others = [(j, r) for j, r in rag_refs if j != i]
        j, other_ref = rng.choice(others)
        wrong[i] = other_ref
    return wrong


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def training_step(batch, lm, adapter, tokenizer, args, rng, device):
    codes     = batch.codes.to(device)
    seg_metas = batch.segment_metas
    B, K, T   = codes.shape
    base_lm   = lm.base_model if hasattr(lm, "base_model") else lm

    references = [
        (ca.get("reference", "") if isinstance(ca, dict) else "")
        for ca in (batch.condition_attributes or [None] * B)
    ]
    correct_sum = adapter.encode_batch(references, tokenizer, device)

    wrong_refs = build_wrong_references(references, seg_metas, rng)
    wrong_sum  = adapter.encode_batch(wrong_refs, tokenizer, device)
    has_contrastive = any(r for r in wrong_refs)

    output = forward_train_with_streaming_sum(
        base_lm, codes=codes, streaming_sum=correct_sum, seg_metas=seg_metas,
    )

    filtered_metas = [m for m in seg_metas if m is not None]
    text_loss, audio_loss, contrastive_loss = compute_rag_losses(
        model=base_lm,
        output=output,
        codes=codes,
        seg_metas=filtered_metas,
        args=args,
        forward_fn=partial(forward_train_with_streaming_sum, base_lm),
        wrong_streaming_sum=wrong_sum if has_contrastive else None,
    )

    loss = text_loss + audio_loss + args.contrastive_weight * contrastive_loss
    loss.backward()

    return {
        "loss": loss.item(), "text_loss": text_loss.item(),
        "audio_loss": audio_loss.item(), "contrastive": contrastive_loss.item(),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    device = torch.device(args.device)

    # ── Load models ───────────────────────────────────────────────────────
    logger.info("loading tokenizer")
    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)

    logger.info("loading mimi (frozen)")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, device)
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad_(False)   # Mimi never trains

    logger.info("loading moshi LM")
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    lm_base: LMModel = loaders.get_moshi_lm(
        args.moshi_weight, device=device, cpu_offload=False
    )
    lm_base.eval()

    # ── Vocabulary surgery: add <ref> token ───────────────────────────────
    # The new token index is text_card (one past the last existing token).
    # We extend text_linear output and text_emb input by one row.
    old_card = lm_base.text_card
    new_card = old_card + 1
    rag_token_id = old_card   # the new token's ID

    # text_linear: [old_card, dim] → [new_card, dim]
    old_linear = lm_base.text_linear
    new_linear  = nn.Linear(
        old_linear.in_features, new_card,
        bias=old_linear.bias is not None,
        device=device,
    )
    with torch.no_grad():
        new_linear.weight[:old_card] = old_linear.weight
        # Init new row to mean of existing rows (in-distribution start)
        new_linear.weight[old_card]  = old_linear.weight.mean(dim=0)
        if old_linear.bias is not None:
            new_linear.bias[:old_card] = old_linear.bias
            new_linear.bias[old_card]  = old_linear.bias.mean()
    lm_base.text_linear = new_linear

    # text_emb: ScaledEmbedding [old_card+1, dim] → [new_card+1, dim]
    # (+1 because ScaledEmbedding has an extra slot for the zero_idx)
    old_emb = lm_base.text_emb
    new_emb = type(old_emb)(
        new_card + 1,
        old_emb.embedding_dim,
        norm=(old_emb.norm is not None),
        zero_idx=old_emb.zero_idx,
        device=device,
    )
    with torch.no_grad():
        new_emb.weight[: old_card + 1] = old_emb.weight
        new_emb.weight[old_card + 1]   = old_emb.weight.mean(dim=0)
    lm_base.text_emb   = new_emb
    lm_base.text_card  = new_card

    # Expose rag_token_id on the model so loss functions can access it
    lm_base.rag_token_id = rag_token_id
    model_dtype = next(lm_base.parameters()).dtype   
    lm_base = lm_base.to(dtype=model_dtype)          
    logger.info(f"vocabulary extended: {old_card} → {new_card}, rag_token_id={rag_token_id}")

    # ── ReferenceAdapter ──────────────────────────────────────────────────
    adapter = ReferenceAdapter(
        dim=lm_base.dim,
        text_emb=lm_base.text_emb,   # shared weights, not a copy
    )
    model_dtype = next(lm_base.parameters()).dtype
    adapter = adapter.to(device = device, dtype= model_dtype)
    logger.info(f"ReferenceAdapter: {sum(p.numel() for p in adapter.parameters()):,} params")

    # ── LoRA ──────────────────────────────────────────────────────────────
    lm = apply_lora(
        lm_base,
        rank=args.lora_rank,
        lora_alpha=args.lora_rank * args.lora_scaling,
        ft_embed=args.ft_embed,
    )

    # ── Gradient checkpointing ────────────────────────────────────────────
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(lm)

    lm.train()

    # ── Optimizer: LoRA params + adapter params only ──────────────────────
    trainable_params = [p for p in lm.parameters() if p.requires_grad]
    adapter_params   = list(adapter.parameters())
    logger.info(
        f"trainable: {sum(p.numel() for p in trainable_params):,} LM params "
        f"+ {sum(p.numel() for p in adapter_params):,} adapter params"
    )

    optimizer = AdamW(
        [
            {"params": trainable_params, "lr": args.lr},
            {"params": adapter_params,   "lr": args.lr * 5},  # adapter trains faster
        ],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.max_steps,
        eta_min=args.lr * 0.1,
    )

    # ── Dataset ───────────────────────────────────────────────────────────
    interleaver = Interleaver(
        tokenizer=tokenizer,
        audio_frame_rate=mimi.frame_rate,
        text_padding=lm_base.base_model.text_padding_token_id
            if hasattr(lm_base, "base_model") else lm_base.text_padding_token_id,
        end_of_text_padding=lm_base.base_model.end_of_text_padding_id
            if hasattr(lm_base, "base_model") else lm_base.end_of_text_padding_id,
        zero_padding=lm_base.base_model.zero_token_id
            if hasattr(lm_base, "base_model") else lm_base.zero_token_id,
        keep_main_only=False,
        use_bos_eos=True,
        main_speaker_label="MODEL",
        device=str(device),
    )
    instruct_tokenizer = InterleavedTokenizer(
        mimi=mimi,
        interleaver=interleaver,
        duration_sec=args.duration_sec,
    )

    train_loader = build_data_loader(
        instruct_tokenizer=instruct_tokenizer,
        args=args,
        batch_size=args.batch_size,
        seed=42,
        rank=0,
        world_size=1,
        is_eval=False,
    )
    eval_loader = build_data_loader(
        instruct_tokenizer=instruct_tokenizer,
        args=args,
        batch_size=args.batch_size,
        seed=0,
        rank=0,
        world_size=1,
        is_eval=True,
    ) if args.eval_data else None

    # ── Resume ────────────────────────────────────────────────────────────
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(
            Path(args.resume), lm, adapter, optimizer, scheduler
        )

    # ── Training loop ─────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    step = start_step
    log_interval  = args.log_every
    eval_interval = args.eval_every
    save_interval = args.save_every

    running = {k: 0.0 for k in ("loss", "text_loss", "audio_loss", "contrastive")}
    running_n = 0
    t0 = time.time()

    logger.info(f"starting training from step {step}")

    optimizer.zero_grad()
    accum_steps = 0

    for batch in train_loader:
        if step >= args.max_steps:
            break

        # ── Forward + backward ────────────────────────────────────────────
        scalars = training_step(
            batch=batch,
            lm=lm,
            adapter=adapter,
            tokenizer=tokenizer,
            args=args,
            rng=rng,
            device=device,
        )
        accum_steps += 1

        for k, v in scalars.items():
            running[k] += v
        running_n += 1

        # ── Gradient accumulation + optimizer step ────────────────────────
        if accum_steps >= args.grad_accum:
            torch.nn.utils.clip_grad_norm_(
                list(lm.parameters()) + list(adapter.parameters()),
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            accum_steps = 0

            # ── Logging ───────────────────────────────────────────────────
            if step % log_interval == 0:
                elapsed = time.time() - t0
                avg     = {k: running[k] / max(running_n, 1) for k in running}
                lr      = scheduler.get_last_lr()[0]
                logger.info(
                    f"step {step:5d}/{args.max_steps} "
                    f"loss={avg['loss']:.4f} "
                    f"text={avg['text_loss']:.4f} "
                    f"audio={avg['audio_loss']:.4f} "
                    f"contrastive={avg['contrastive']:.4f} "
                    f"lr={lr:.2e} "
                    f"elapsed={elapsed:.0f}s"
                )
                running   = {k: 0.0 for k in running}
                running_n = 0

            # ── Eval ──────────────────────────────────────────────────────
            if eval_loader is not None and step % eval_interval == 0:
                lm.eval()
                adapter.eval()
                eval_losses = {k: 0.0 for k in ("loss", "text_loss", "audio_loss")}
                eval_n = 0
                with torch.no_grad():
                    for eval_batch in eval_loader:
                        s = training_step(
                            batch=eval_batch,
                            lm=lm,
                            adapter=adapter,
                            tokenizer=tokenizer,
                            args=args,
                            rng=rng,
                            device=device,
                        )
                        for k in eval_losses:
                            eval_losses[k] += s[k]
                        eval_n += 1
                        if eval_n >= 50:   # limit eval batches
                            break
                avg_eval = {k: v / max(eval_n, 1) for k, v in eval_losses.items()}
                logger.info(
                    f"[eval] step {step} "
                    f"loss={avg_eval['loss']:.4f} "
                    f"text={avg_eval['text_loss']:.4f} "
                    f"audio={avg_eval['audio_loss']:.4f}"
                )
                lm.train()
                adapter.train()

            # ── Save ──────────────────────────────────────────────────────
            if step % save_interval == 0:
                save_checkpoint(
                    out_dir, step, lm, adapter, optimizer, scheduler,
                    stats={"step": step},
                )

    # Final save
    save_checkpoint(
        out_dir, step, lm, adapter, optimizer, scheduler,
        stats={"step": step, "final": True},
    )
    logger.info(f"training complete at step {step}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    ap.add_argument("--train-data",  required=True, help="train.jsonl manifest path")
    ap.add_argument("--eval-data",   default="",    help="eval.jsonl manifest path (optional)")
    ap.add_argument("--duration-sec", type=float, default=100.0)
    ap.add_argument("--batch-size",   type=int,   default=16)

    # Model
    ap.add_argument("--hf-repo",      default=loaders.DEFAULT_REPO)
    ap.add_argument("--moshi-weight", default=None)
    ap.add_argument("--mimi-weight",  default=None)
    ap.add_argument("--tokenizer",    default=None)
    ap.add_argument("--device",       default="cuda")

    # LoRA
    ap.add_argument("--lora-rank",    type=int,   default=128)
    ap.add_argument("--lora-scaling", type=float, default=2.0)
    ap.add_argument("--ft-embed",     action="store_true",
                    help="Also fine-tune text_emb (default: frozen)")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)

    # Optimiser
    ap.add_argument("--lr",           type=float, default=2e-6)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--max-steps",    type=int,   default=2000)
    ap.add_argument("--grad-accum",   type=int,   default=1,
                    help="Gradient accumulation steps (default: 1)")

    # Loss weights
    ap.add_argument("--first-codebook-weight", dest="first_codebook_weight_multiplier",
                    type=float, default=100.0)
    ap.add_argument("--text-padding-weight",   type=float, default=0.5)
    ap.add_argument("--ref-token-weight",      type=float, default=5.0)
    ap.add_argument("--lead-weight",           type=float, default=1.0)
    ap.add_argument("--filler-weight",         type=float, default=0.5)
    ap.add_argument("--body-weight",           type=float, default=2.0)
    ap.add_argument("--no-rag-weight",         type=float, default=1.0)
    ap.add_argument("--contrastive-weight",    type=float, default=0.3)
    ap.add_argument("--contrastive-margin",    type=float, default=0.3)

    # Checkpointing / logging
    ap.add_argument("--out-dir",    default="checkpoints/rag_lora")
    ap.add_argument("--resume",     default=None,
                    help="Path to checkpoint directory to resume from")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every",  type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=200)

    # DataArgs compatibility (required by build_data_loader)
    ap.add_argument("--shuffle", action="store_true", default=True)

    args = ap.parse_args()

    # Expose as DataArgs-compatible attributes
    args.train_data = args.train_data
    args.eval_data  = args.eval_data

    with torch.no_grad():
        pass   # not wrapping train() — we need gradients

    train(args)


if __name__ == "__main__":
    main()
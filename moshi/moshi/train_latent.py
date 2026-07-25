"""
train_rag_resampler.py
=======================
Perceiver-style resampler: compresses a reference of any length into a
fixed M continuous latent vectors, injected into the LM's text-channel
embeddings at reserved slot positions (placeholder token ids in `codes`,
overwritten with resampler output before the transformer forward).

Unlike literal <ref> text injection, this needs a real embed_codes-level
hook, since the injected content is continuous vectors, not real tokens.
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path
from collections import deque

import sentencepiece
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.checkpoint import checkpoint

from huggingface_hub import hf_hub_download

from .models import loaders
from .models.lm import LMModel
from .rag_loss import compute_rag_losses, SegmentMeta
from .dataset import build_data_loader
from .interleaver_latent import InterleavedTokenizer, Interleaver, Batch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Perceiver resampler
# ---------------------------------------------------------------------------

class ReferenceResampler(nn.Module):
    """Compresses a variable-length reference into `num_latents` fixed
    vectors via learned-query cross-attention. Cost is independent of
    reference length beyond this module."""
    def __init__(self, dim: int, text_emb: nn.Embedding, num_latents: int = 16,
                 num_layers: int = 2, num_heads: int = 8):
        super().__init__()
        self.text_emb = text_emb   # shared with LM, frozen — reuse existing token embeddings
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(dim, num_heads, batch_first=True),
                "cross_ln": nn.LayerNorm(dim),
                "self_attn": nn.MultiheadAttention(dim, num_heads, batch_first=True),
                "self_ln": nn.LayerNorm(dim),
                "ff": nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)),
                "ff_ln": nn.LayerNorm(dim),
            }) for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(dim)
        # start near-zero so early training doesn't shock the LM with an
        # unfamiliar signal — mirrors the out_scale trick from before
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, ref_token_ids: torch.Tensor, ref_mask: torch.Tensor) -> torch.Tensor:
        """ref_token_ids: [B, T_ref] padded; ref_mask: [B, T_ref] True=valid.
        Returns: [B, M, dim]"""
        B = ref_token_ids.shape[0]
        with torch.no_grad():
            ref_embeds = self.text_emb(ref_token_ids.clamp(min=0))   # [B, T_ref, dim], frozen table
        x = self.latents.unsqueeze(0).expand(B, -1, -1).contiguous()
        key_padding_mask = ~ref_mask

        for layer in self.layers:
            attn_out, _ = layer["cross_attn"](x, ref_embeds, ref_embeds, key_padding_mask=key_padding_mask)
            x = layer["cross_ln"](x + attn_out)
            self_out, _ = layer["self_attn"](x, x, x)
            x = layer["self_ln"](x + self_out)
            x = layer["ff_ln"](x + layer["ff"](x))
        return self.out_norm(x) * self.out_scale


# ---------------------------------------------------------------------------
# LoRA / checkpointing — identical to approach 1
# ---------------------------------------------------------------------------

def apply_lora(lm: LMModel, rank: int = 128, lora_alpha: float = 256.0,
                target_modules: list[str] | None = None) -> LMModel:
    from peft import LoraConfig, get_peft_model, TaskType
    if target_modules is None:
        target_modules = ["in_proj", "out_proj", "fc1", "fc2", "linear", "proj"]
    config = LoraConfig(r=rank, lora_alpha=lora_alpha, target_modules=target_modules,
                         lora_dropout=0.05, bias="none", task_type=TaskType.FEATURE_EXTRACTION,
                         modules_to_save=None)
    lm = get_peft_model(lm, config)
    lm.print_trainable_parameters()
    return lm


def enable_gradient_checkpointing(lm: nn.Module) -> None:
    if hasattr(lm, "enable_input_require_grads"):
        lm.enable_input_require_grads()
    base = lm.base_model if hasattr(lm, "base_model") else lm
    for name in ("transformer", "depformer"):
        module = getattr(base, name, None)
        if module is not None and hasattr(module, "gradient_checkpointing_enable"):
            module.gradient_checkpointing_enable()
        elif module is not None and hasattr(module, "layers"):
            module.gradient_checkpointing = True


# ---------------------------------------------------------------------------
# Forward pass with latent-slot injection
# ---------------------------------------------------------------------------

def embed_codes_split(lm: LMModel, sequence: torch.Tensor):
    audio_sum = None
    for cb_index in range(lm.num_audio_codebooks):
        audio_emb = lm.emb[cb_index](sequence[:, cb_index + lm.audio_offset])
        audio_sum = audio_emb if audio_sum is None else audio_sum + audio_emb
    text_emb = lm.text_emb(sequence[:, 0])
    return audio_sum, text_emb


def forward_train_with_latents(
    self: LMModel,
    codes: torch.Tensor,
    seg_metas: list[SegmentMeta | None],
    resampler: ReferenceResampler,
    ref_token_ids: torch.Tensor,   # [B, T_ref] padded
    ref_mask: torch.Tensor,        # [B, T_ref]
):
    from .models.lm import _delay_sequence, _undelay_sequence, LMOutput

    B, K, T = codes.shape
    initial = self._get_initial_token().expand(B, -1, -1)
    delayed = _delay_sequence(self.delays, codes, initial)
    delayed = torch.cat([initial, delayed], dim=2)

    audio_sum, text_emb = embed_codes_split(self, delayed[:, :, :-1])
    embedded = audio_sum + text_emb

    latents = resampler(ref_token_ids, ref_mask)   # [B, M, dim]
    for meta in seg_metas:
        if meta is None or meta.n_ref_tokens == 0:
            continue
        b = meta.batch_idx
        s = meta.filler_end + 1
        e = s + meta.n_ref_tokens
        if e <= embedded.shape[1]:
            embedded[b, s:e] = latents[b, : meta.n_ref_tokens]

    def _run_transformer(x):
        return self.forward_embeddings(x)
    transformer_out, text_logits = checkpoint(_run_transformer, embedded, use_reentrant=False)

    def _run_depformer(delayed_slice, t_out):
        return self.forward_depformer_training(delayed_slice, t_out)
    logits = checkpoint(_run_depformer, delayed[:, :, 1:], transformer_out, use_reentrant=False)

    logits, logits_mask = _undelay_sequence(
        self.delays[self.audio_offset: self.audio_offset + self.dep_q],
        logits, fill_value=float("NaN"),
    )
    logits_mask &= (codes[:, self.audio_offset: self.audio_offset + self.dep_q] != self.zero_token_id)
    text_logits, text_logits_mask = _undelay_sequence(self.delays[:1], text_logits, fill_value=float("NaN"))
    text_logits_mask &= codes[:, :1] != self.zero_token_id

    return LMOutput(logits, logits_mask, text_logits, text_logits_mask)


# ---------------------------------------------------------------------------
# Checkpoint save / load — now includes resampler state
# ---------------------------------------------------------------------------

def save_checkpoint(out_dir: Path, step: int, lm: nn.Module, resampler: ReferenceResampler,
                     optimizer, scheduler, stats: dict) -> None:
    ckpt_dir = out_dir / f"step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(lm, "save_pretrained"):
        lm.save_pretrained(ckpt_dir / "lora")
    else:
        torch.save(lm.state_dict(), ckpt_dir / "lm_state.pt")
    torch.save(resampler.state_dict(), ckpt_dir / "resampler.pt")
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step, "stats": stats,
    }, ckpt_dir / "optim.pt")
    logger.info(f"[ckpt] saved step {step} → {ckpt_dir}")


def load_checkpoint(ckpt_dir: Path, lm: nn.Module, resampler: ReferenceResampler,
                     optimizer, scheduler) -> int:
    lora_path = ckpt_dir / "lora"
    if lora_path.exists() and hasattr(lm, "load_adapter"):
        lm.load_adapter(str(lora_path), adapter_name="default")
    elif (ckpt_dir / "lm_state.pt").exists():
        lm.load_state_dict(torch.load(ckpt_dir / "lm_state.pt"))
    resampler_path = ckpt_dir / "resampler.pt"
    if resampler_path.exists():
        resampler.load_state_dict(torch.load(resampler_path))
    optim_path = ckpt_dir / "optim.pt"
    if optim_path.exists():
        state = torch.load(optim_path)
        optimizer.load_state_dict(state["optimizer"])
        if scheduler and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        return state["step"]
    return 0


# ---------------------------------------------------------------------------
# Reference batching helper — pads references to a common T_ref for the batch
# ---------------------------------------------------------------------------

def encode_references_batch(references: list[str], tokenizer, device, max_tokens: int = 256):
    """Returns (ref_token_ids [B, T_ref], ref_mask [B, T_ref])."""
    encoded = [tokenizer.encode(r)[:max_tokens] if r else [] for r in references]
    max_len = max((len(e) for e in encoded), default=1) or 1
    B = len(references)
    ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
    mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
    for i, e in enumerate(encoded):
        if e:
            ids[i, :len(e)] = torch.tensor(e, device=device)
            mask[i, :len(e)] = True
    return ids, mask


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def training_step(batch: Batch, lm, resampler, tokenizer, args, device):
    codes = batch.codes.to(device)
    seg_metas = batch.segment_metas
    B = codes.shape[0]
    base_lm = lm.base_model if hasattr(lm, "base_model") else lm

    references = [
        (ca.get("reference", "") if isinstance(ca, dict) else "")
        for ca in (batch.condition_attributes or [None] * B)
    ]
    ref_ids, ref_mask = encode_references_batch(references, tokenizer, device, max_tokens=args.max_ref_source_tokens)

    output = forward_train_with_latents(
        base_lm, codes=codes, seg_metas=seg_metas,
        resampler=resampler, ref_token_ids=ref_ids, ref_mask=ref_mask,
    )

    filtered_metas = [m for m in seg_metas if m is not None]
    text_loss, audio_loss = compute_rag_losses(
        model=base_lm, output=output, codes=codes, seg_metas=filtered_metas, args=args,
    )
    loss = text_loss + audio_loss
    loss.backward()

    return {"loss": loss.item(), "text_loss": text_loss.item(), "audio_loss": audio_loss.item()}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(args.device)

    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)

    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, device)
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad_(False)

    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    lm_base: LMModel = loaders.get_moshi_lm(args.moshi_weight, device=device, cpu_offload=False)
    lm_base.eval()
    model_dtype = next(lm_base.parameters()).dtype
    lm_base = lm_base.to(dtype=model_dtype)

    lm = apply_lora(lm_base, rank=args.lora_rank, lora_alpha=args.lora_rank * args.lora_scaling)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(lm)
    lm.train()

    resampler = ReferenceResampler(
        dim=lm_base.dim, text_emb=lm_base.text_emb, num_latents=args.num_latents,
    ).to(device=device, dtype=model_dtype)
    logger.info(f"ReferenceResampler: {sum(p.numel() for p in resampler.parameters()):,} params")

    trainable_params = [p for p in lm.parameters() if p.requires_grad]
    resampler_params = list(resampler.parameters())
    logger.info(f"trainable: {sum(p.numel() for p in trainable_params):,} LM + "
                f"{sum(p.numel() for p in resampler_params):,} resampler")

    optimizer = AdamW([
        {"params": trainable_params, "lr": args.lr},
        {"params": resampler_params, "lr": args.lr * 5},
    ], weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=args.lr * 0.1)

    interleaver = Interleaver(
        tokenizer=tokenizer, audio_frame_rate=mimi.frame_rate,
        text_padding=lm_base.text_padding_token_id,
        end_of_text_padding=lm_base.end_of_text_padding_id,
        zero_padding=lm_base.zero_token_id,
        keep_main_only=False, use_bos_eos=True, main_speaker_label="MODEL", device=str(device),
    )
    # duration_sec here uses PLACEHOLDER slot frames (fixed = num_latents),
    # not literal reference text — see interleaver-side note below.
    instruct_tokenizer = InterleavedTokenizer(
        mimi=mimi, interleaver=interleaver, duration_sec=args.duration_sec,
        num_latent_slots=args.num_latents,
    )

    train_loader = build_data_loader(instruct_tokenizer=instruct_tokenizer, args=args,
                                      batch_size=args.batch_size, seed=42, rank=0, world_size=1, is_eval=False)
    eval_loader = build_data_loader(instruct_tokenizer=instruct_tokenizer, args=args,
                                     batch_size=args.batch_size, seed=0, rank=0, world_size=1, is_eval=True) \
        if args.eval_data else None

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(Path(args.resume), lm, resampler, optimizer, scheduler)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = start_step
    running = {k: 0.0 for k in ("loss", "text_loss", "audio_loss")}
    running_n = 0
    t0 = time.time()

    optimizer.zero_grad()
    accum_steps = 0

    for batch in train_loader:
        if step >= args.max_steps:
            break
        scalars = training_step(batch=batch, lm=lm, resampler=resampler, tokenizer=tokenizer, args=args, device=device)
        accum_steps += 1
        for k, v in scalars.items():
            running[k] += v
        running_n += 1

        if accum_steps >= args.grad_accum:
            seen, all_params = set(), []
            for p in trainable_params + resampler_params:
                if id(p) not in seen:
                    seen.add(id(p)); all_params.append(p)
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            accum_steps = 0

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                avg = {k: running[k] / max(running_n, 1) for k in running}
                logger.info(f"step {step:5d}/{args.max_steps} loss={avg['loss']:.4f} "
                            f"text={avg['text_loss']:.4f} audio={avg['audio_loss']:.4f} "
                            f"lr={scheduler.get_last_lr()[0]:.2e} elapsed={elapsed:.0f}s")
                running = {k: 0.0 for k in running}
                running_n = 0

            if eval_loader is not None and step % args.eval_every == 0:
                lm.eval(); resampler.eval()
                eval_losses = {k: 0.0 for k in ("loss", "text_loss", "audio_loss")}
                eval_n = 0
                with torch.no_grad():
                    for eval_batch in eval_loader:
                        s = training_step(batch=eval_batch, lm=lm, resampler=resampler,
                                           tokenizer=tokenizer, args=args, device=device)
                        for k in eval_losses:
                            eval_losses[k] += s[k]
                        eval_n += 1
                        if eval_n >= 50:
                            break
                avg_eval = {k: v / max(eval_n, 1) for k, v in eval_losses.items()}
                logger.info(f"[eval] step {step} loss={avg_eval['loss']:.4f} "
                            f"text={avg_eval['text_loss']:.4f} audio={avg_eval['audio_loss']:.4f}")
                lm.train(); resampler.train()

            if step % args.save_every == 0:
                save_checkpoint(out_dir, step, lm, resampler, optimizer, scheduler, stats={"step": step})

    save_checkpoint(out_dir, step, lm, resampler, optimizer, scheduler, stats={"step": step, "final": True})
    logger.info(f"training complete at step {step}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--eval-data", default="")
    ap.add_argument("--duration-sec", type=float, default=100.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-latents", type=int, default=64)
    ap.add_argument("--max-ref-source-tokens", type=int, default=256,
                     help="cap on reference length fed INTO the resampler (pre-compression)")

    ap.add_argument("--hf-repo", default=loaders.DEFAULT_REPO)
    ap.add_argument("--moshi-weight", default=None)
    ap.add_argument("--mimi-weight", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--lora-rank", type=int, default=128)
    ap.add_argument("--lora-scaling", type=float, default=2.0)
    ap.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true", default=True)
    ap.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")

    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--grad-accum", type=int, default=1)

    ap.add_argument("--first-codebook-weight", dest="first_codebook_weight_multiplier", type=float, default=100.0)
    ap.add_argument("--text-padding-weight", type=float, default=0.5)
    ap.add_argument("--ref-token-weight", type=float, default=5.0)
    ap.add_argument("--ref-context-weight", type=float, default=0.0,   # slots carry no real text loss
                     help="should stay 0 — latent slot frames are masked out of text CE (see rag_loss.py)")
    ap.add_argument("--lead-weight", type=float, default=1.0)
    ap.add_argument("--filler-weight", type=float, default=0.5)
    ap.add_argument("--body-weight", type=float, default=2.0)
    ap.add_argument("--no-rag-weight", type=float, default=1.0)

    ap.add_argument("--out-dir", default="./rag_resampler")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--shuffle", action="store_true", default=True)

    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
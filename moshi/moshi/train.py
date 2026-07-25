"""
train_rag.py
============
Fine-tuning script for RAG-aware Moshi LM — literal <ref> text injection.

<ref> is treated exactly like <system>: plain text through the ordinary
tokenizer, no vocabulary surgery, no new embedding row. The reference is
wrapped as "<ref> ... <ref>", tokenized normally, and spliced into the
text codebook at the point where the model should start reading it
(right after the filler segment, before the body/grounding segment).

Loss weighting (rag_loss.py, unchanged interface) upweights the body
segment so gradient teaches the model to actually use what it read.
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path
from functools import partial
from collections import deque

import sentencepiece
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.checkpoint import checkpoint

from huggingface_hub import hf_hub_download

from .models import loaders
from .models.lm import LMModel
from .rag_loss_new import compute_rag_losses, SegmentMeta
from .dataset import build_data_loader
from .interleaver import InterleavedTokenizer, Interleaver, Batch

logger = logging.getLogger(__name__)

_lookup_diag_state = {"correct": 0, "total": 0}

def diagnose_lookup_predictions(
    text_logits: torch.Tensor,   # [B, 1, T, card], raw with NaN at masked positions
    codes: torch.Tensor,         # [B, K, T]
    seg_metas: list,
    tokenizer,
    step: int,
    max_print: int = 2,
):
    """Prints what the model actually predicts at the frame immediately
    before the <lookup> token begins — i.e. the frame whose logits are
    the model's real 'do I need context' decision, before any teacher
    forcing pushes it past that point. Also tracks running top-1 accuracy
    at that specific frame across steps so you can see if it's moving."""
    printed = 0
    T = text_logits.shape[2]
    for meta in seg_metas:
        lookup_spans = meta.spans_of("lookup")
        if not lookup_spans:
            continue
        b = meta.batch_idx
        lookup_start = lookup_spans[0].start_frame

        # pos = lookup_start - 1 is the frame whose NEXT-token prediction
        # target is the first <lookup> token, under a standard shift-by-one
        # LM convention. VERIFY this offset against forward_train_checkpointed's
        # actual delay/undelay convention in your codebase — if predictions
        # look nonsensical (e.g. target never equals the true next code at
        # this position for OBVIOUS cases like ordinary words), try pos=lookup_start
        # instead. This is exactly the off-by-one I flagged as the most likely
        # silent bug last time — these prints are how you settle it empirically.
        pos = lookup_start - 1
        if pos < 0 or pos >= T:
            continue

        logits_bt = torch.nan_to_num(text_logits[b, 0, pos].float(), nan=-1e9, posinf=-1e9, neginf=-1e9)
        probs = torch.softmax(logits_bt, dim=-1)
        topk = torch.topk(probs, 5)
        target_id = codes[b, 0, lookup_start].item()
        pred_id = topk.indices[0].item()
        match = pred_id == target_id

        _lookup_diag_state["total"] += 1
        _lookup_diag_state["correct"] += int(match)

        if printed < max_print:
            top5_str = ", ".join(
                f"{tokenizer.id_to_piece(int(i))!r}:{p:.3f}"
                for i, p in zip(topk.indices, topk.values)
            )
            logger.info(
                f"[diag-lookup] step {step} batch_idx={b} pos={pos}: "
                f"target={tokenizer.id_to_piece(target_id)!r}({target_id}) "
                f"top1={tokenizer.id_to_piece(pred_id)!r}({pred_id}) p={topk.values[0]:.3f} "
                f"top5=[{top5_str}] match={'YES' if match else 'no'}"
            )
            printed += 1

    if step % 50 == 0 and _lookup_diag_state["total"] > 0:
        acc = _lookup_diag_state["correct"] / _lookup_diag_state["total"]
        logger.info(
            f"[diag-lookup] step {step} running top1 accuracy at lookup-decision frame: "
            f"{acc:.3f} ({_lookup_diag_state['correct']}/{_lookup_diag_state['total']})"
        )

        
class ReferenceBuffer:
    """Holds recently-seen RAG reference strings so wrong-reference sampling
    isn't limited to the current (tiny) micro-batch."""
    def __init__(self, maxlen: int = 128):
        self.buf: deque[str] = deque(maxlen=maxlen)

    def add(self, refs: list[str]):
        self.buf.extend(r for r in refs if r)

    def sample_wrong(self, references: list[str], seg_metas, rng: random.Random) -> list[str]:
        wrong = [""] * len(references)
        pool = list(self.buf)
        for i, meta in enumerate(seg_metas):
            if meta is None or not references[i]:
                continue
            candidates = [r for r in pool if r != references[i]]
            if candidates:
                wrong[i] = rng.choice(candidates)
        return wrong


# ---------------------------------------------------------------------------
# LoRA application
# ---------------------------------------------------------------------------

def apply_lora(
    lm: LMModel,
    rank: int = 128,
    lora_alpha: float = 256.0,
    target_modules: list[str] | None = None,
) -> LMModel:
    """Apply LoRA to the main transformer and depformer inside LMModel.
    text_emb / text_linear are NOT touched — <ref> is plain text, no
    vocabulary surgery, so nothing special needs unfreezing there."""
    from peft import LoraConfig, get_peft_model, TaskType

    if target_modules is None:
        target_modules = ["in_proj", "out_proj", "fc1", "fc2", "linear", "proj"]

    config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        modules_to_save=None,
    )
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
            logger.info(f"gradient checkpointing enabled on lm.{name}")
        elif module is not None and hasattr(module, "layers"):
            module.gradient_checkpointing = True
            logger.info(f"set gradient_checkpointing=True on lm.{name}")


# ---------------------------------------------------------------------------
# Forward pass — plain forward_train, no injection needed. <ref> content is
# already IN `codes` (spliced by InterleavedTokenizer at data-prep time),
# so this is just the model's own forward_train with checkpointing added.
# No monkeypatch of embed_codes / streaming_sum needed for this approach.
# ---------------------------------------------------------------------------

def forward_train_checkpointed(self: LMModel, codes: torch.Tensor):
    from .models.lm import _delay_sequence, _undelay_sequence, LMOutput

    B, K, T = codes.shape
    initial = self._get_initial_token().expand(B, -1, -1)
    delayed = _delay_sequence(self.delays, codes, initial)
    delayed = torch.cat([initial, delayed], dim=2)

    def _run_transformer(seq):
        return self.forward_codes(seq)
    transformer_out, text_logits = checkpoint(_run_transformer, delayed[:, :, :-1], use_reentrant=False)

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
# Checkpoint save / load — simple again, no vocab-surgery artifact needed
# ---------------------------------------------------------------------------

def save_checkpoint(out_dir: Path, step: int, lm: nn.Module,
                     optimizer: torch.optim.Optimizer, scheduler, stats: dict) -> None:
    ckpt_dir = out_dir / f"step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(lm, "save_pretrained"):
        lm.save_pretrained(ckpt_dir / "lora")
    else:
        torch.save(lm.state_dict(), ckpt_dir / "lm_state.pt")
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step, "stats": stats,
    }, ckpt_dir / "optim.pt")
    logger.info(f"[ckpt] saved step {step} → {ckpt_dir}")


def load_checkpoint(ckpt_dir: Path, lm: nn.Module,
                     optimizer: torch.optim.Optimizer, scheduler) -> int:
    lora_path = ckpt_dir / "lora"
    if lora_path.exists() and hasattr(lm, "load_adapter"):
        lm.load_adapter(str(lora_path), adapter_name="default")
        logger.info(f"[ckpt] loaded LoRA from {lora_path}")
    elif (ckpt_dir / "lm_state.pt").exists():
        lm.load_state_dict(torch.load(ckpt_dir / "lm_state.pt"))

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
# Training step
# ---------------------------------------------------------------------------

def training_step(batch: Batch, lm, args, rng, device, tokenizer, step: int = 0, diag_every: int = 20):
    codes = batch.codes.to(device)
    seg_metas = batch.segment_metas
    base_lm = lm.base_model if hasattr(lm, "base_model") else lm

    output = forward_train_checkpointed(base_lm, codes=codes)

    filtered_metas = [m for m in seg_metas if m is not None]

    if step % diag_every == 0:
        diagnose_lookup_predictions(
            text_logits=output.text_logits, codes=codes,
            seg_metas=filtered_metas, tokenizer=tokenizer, step=step,
        )
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
        p.requires_grad_(False)

    logger.info("loading moshi LM")
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    lm_base: LMModel = loaders.get_moshi_lm(args.moshi_weight, device=device, cpu_offload=False)
    lm_base.eval()
    model_dtype = next(lm_base.parameters()).dtype
    lm_base = lm_base.to(dtype=model_dtype)

    # NOTE: no vocab surgery. text_emb / text_linear are untouched — <ref>
    # is plain text through the existing tokenizer, exactly like <system>.

    lm = apply_lora(lm_base, rank=args.lora_rank, lora_alpha=args.lora_rank * args.lora_scaling)
    if args.ft_embed:
        # Only if you explicitly want the WHOLE embedding table trainable —
        # not needed for <ref> to work, kept as a general escape hatch.
        lm_base.text_emb.weight.requires_grad_(True)
        lm_base.text_linear.weight.requires_grad_(True)
        if lm_base.text_linear.bias is not None:
            lm_base.text_linear.bias.requires_grad_(True)

    if args.gradient_checkpointing:
        enable_gradient_checkpointing(lm)
    lm.train()

    trainable_params = [p for p in lm.parameters() if p.requires_grad]
    logger.info(f"trainable: {sum(p.numel() for p in trainable_params):,} params")

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay,
                       betas=(0.9, 0.95), eps=1e-8)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=args.lr * 0.1)

    interleaver = Interleaver(
        tokenizer=tokenizer,
        audio_frame_rate=mimi.frame_rate,
        text_padding=lm_base.text_padding_token_id,
        end_of_text_padding=lm_base.end_of_text_padding_id,
        zero_padding=lm_base.zero_token_id,
        keep_main_only=False,
        use_bos_eos=True,
        main_speaker_label="MODEL",
        device=str(device),
    )
    # max_ref_tokens caps the injected <ref> block length — bounds sequence
    # growth regardless of how long a retrieved reference is.
    instruct_tokenizer = InterleavedTokenizer(
        mimi=mimi, interleaver=interleaver, duration_sec=args.duration_sec,
        max_ref_tokens=args.max_ref_tokens,
    )

    train_loader = build_data_loader(
        instruct_tokenizer=instruct_tokenizer, args=args, batch_size=args.batch_size,
        seed=42, rank=0, world_size=1, is_eval=False,
    )
    eval_loader = build_data_loader(
        instruct_tokenizer=instruct_tokenizer, args=args, batch_size=args.batch_size,
        seed=0, rank=0, world_size=1, is_eval=True,
    ) if args.eval_data else None

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(Path(args.resume), lm, optimizer, scheduler)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    step = start_step
    running = {k: 0.0 for k in ("loss", "text_loss", "audio_loss")}
    running_n = 0
    t0 = time.time()
    logger.info(f"starting training from step {step}")

    optimizer.zero_grad()
    accum_steps = 0

    for batch in train_loader:
        if step >= args.max_steps:
            break

        scalars = training_step(batch=batch, lm=lm, args=args, rng=rng, device=device, tokenizer=tokenizer, step=step )
        accum_steps += 1
        for k, v in scalars.items():
            running[k] += v
        running_n += 1

        if accum_steps >= args.grad_accum:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            accum_steps = 0

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                avg = {k: running[k] / max(running_n, 1) for k in running}
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"step {step:5d}/{args.max_steps} loss={avg['loss']:.4f} "
                    f"text={avg['text_loss']:.4f} audio={avg['audio_loss']:.4f} "
                    f"lr={lr:.2e} elapsed={elapsed:.0f}s"
                )
                running = {k: 0.0 for k in running}
                running_n = 0

            if eval_loader is not None and step % args.eval_every == 0:
                lm.eval()
                eval_losses = {k: 0.0 for k in ("loss", "text_loss", "audio_loss")}
                eval_n = 0
                with torch.no_grad():
                    for eval_batch in eval_loader:
                        s = training_step(batch=eval_batch, lm=lm, args=args, rng=rng, device=device)
                        for k in eval_losses:
                            eval_losses[k] += s[k]
                        eval_n += 1
                        if eval_n >= 50:
                            break
                avg_eval = {k: v / max(eval_n, 1) for k, v in eval_losses.items()}
                logger.info(f"[eval] step {step} loss={avg_eval['loss']:.4f} "
                            f"text={avg_eval['text_loss']:.4f} audio={avg_eval['audio_loss']:.4f}")
                lm.train()

            if step % args.save_every == 0:
                save_checkpoint(out_dir, step, lm, optimizer, scheduler, stats={"step": step})

    save_checkpoint(out_dir, step, lm, optimizer, scheduler, stats={"step": step, "final": True})
    logger.info(f"training complete at step {step}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("--train-data", required=True)
    ap.add_argument("--eval-data", default="")
    ap.add_argument("--duration-sec", type=float, default=100.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-ref-tokens", type=int, default=500)

    ap.add_argument("--hf-repo", default=loaders.DEFAULT_REPO)
    ap.add_argument("--moshi-weight", default=None)
    ap.add_argument("--mimi-weight", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--lora-rank", type=int, default=128)
    ap.add_argument("--lora-scaling", type=float, default=2.0)
    ap.add_argument("--ft-embed", action="store_true",
                     help="Fine-tune the whole text_emb/text_linear table (not required for <ref>)")
    ap.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true", default=True)
    ap.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")

    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--grad-accum", type=int, default=1)

    ap.add_argument("--first-codebook-weight", dest="first_codebook_weight_multiplier", type=float, default=100.0)
    ap.add_argument("--text-padding-weight", type=float, default=0.5)
    ap.add_argument("--ref-token-weight", type=float, default=5.0)
    ap.add_argument("--ref-context-weight", type=float, default=1.0)
    ap.add_argument("--lead-weight", type=float, default=1.0)
    ap.add_argument("--lookup-weight", type=float, default=5.0)   # NEW — rare, high-signal token, same treatment as ref
    ap.add_argument("--filler-weight", type=float, default=1.0)   # WAS 0.5 — parity with lead, per earlier finding
    ap.add_argument("--body-weight", type=float, default=2.0)
    ap.add_argument("--no-rag-weight", type=float, default=1.0)
    

    ap.add_argument("--out-dir", default="checkpoints/rag_lora")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=200)

    ap.add_argument("--shuffle", action="store_true", default=True)

    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
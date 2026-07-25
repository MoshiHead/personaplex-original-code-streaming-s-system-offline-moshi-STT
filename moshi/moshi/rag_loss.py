"""
Extended loss computation for RAG fine-tuning — literal <ref> text approach
and Perceiver-resampler approach both use this module identically.

Builds on the base text/audio CE by adding:
  - per-segment text weights (lead / ref-token-frame / filler / ref-context / body)
  - NaN-safe handling of undelayed logits (masked positions come back as
    real NaN from _undelay_sequence's fill_value, not zero — must be
    sanitized BEFORE cross_entropy, not after, or backward can propagate
    NaN gradients even though the forward-masked value looks fine)

No contrastive loss in this version — dropped along with the streaming_sum
adapter path. If you reintroduce a contrastive objective later, it should
compare against a forward pass with a genuinely wrong <ref> BLOCK spliced
into codes (literal approach) or a wrong reference fed to the resampler
(latent approach) — not the old streaming_sum-vector mechanism.
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Segment metadata — one per example in the batch
# ---------------------------------------------------------------------------

@dataclass
class SegmentMeta:
    ref_frame:    int
    filler_start: int
    filler_end:   int
    body_start:   int
    n_ref_tokens: int = 0     # length of the injected <ref> block / latent-slot count
    batch_idx:    int = 0


def make_segment_weight_tensor(
    B: int, T: int, seg_metas: list[SegmentMeta],
    device: torch.device,
    ref_weight: float = 5.0,
    lead_weight: float = 1.0,
    filler_weight: float = 0.5,
    ref_context_weight: float = 1.0,
    body_weight: float = 2.0,
    no_rag_weight: float = 1.0,
) -> torch.Tensor:
    """
    Build a [B, T] float tensor of per-frame text loss weights.

    Weighting is purely POSITIONAL (frame index within each segment's
    boundaries) — there is no special token id to key off of, since <ref>
    is literal text (no vocab surgery) and the latent-slot approach uses
    placeholder ids that are never predicted anyway.

    ref_context_weight covers the span [filler_end+1, body_start):
      - literal <ref> approach: this is the actual injected reference
        text — worth a real (but modest) text-loss signal, since teaching
        the model to correctly "read back" reference tokens is a mild
        useful signal, though NOT the primary grounding objective.
      - latent-slot approach: this span holds placeholder ids with no
        real semantic content (overwritten at the embedding level, not
        the token level) — ref_context_weight MUST be 0 there, since
        there is nothing meaningful to predict. Set via
        --ref-context-weight 0.0 in train_rag_resampler.py.
    """
    weights = torch.full((B, T), no_rag_weight, dtype=torch.float32, device=device)

    for meta in seg_metas:
        b = meta.batch_idx

        # Lead: frames 0 .. ref_frame-1
        if meta.ref_frame > 0:
            weights[b, :meta.ref_frame] = lead_weight

        # <ref> tag's first token frame (or, for latent-slot mode, simply
        # the frame position where the model should have started signaling
        # intent to retrieve) — single frame, heavy upweight.
        if meta.ref_frame < T:
            weights[b, meta.ref_frame] = ref_weight

        # Filler: ref_frame+1 .. filler_end
        f_s = meta.filler_start
        f_e = min(meta.filler_end + 1, T)
        if f_s < T:
            weights[b, f_s:f_e] = filler_weight

        # Injected block: [filler_end+1, body_start)
        rc_s = min(meta.filler_end + 1, T)
        rc_e = min(meta.body_start, T)
        if rc_s < rc_e:
            weights[b, rc_s:rc_e] = ref_context_weight

        # Body: body_start .. end — the actual grounding signal
        b_s = meta.body_start
        if b_s < T:
            weights[b, b_s:] = body_weight

    return weights


# ---------------------------------------------------------------------------
# Text loss with segment weights
# ---------------------------------------------------------------------------

def compute_text_loss_rag(
    text_logits: torch.Tensor,   # [B, 1, T, text_card] — contains real NaN at masked positions
    text_codes:  torch.Tensor,   # [B, 1, T]
    text_mask:   torch.Tensor,   # [B, 1, T]
    seg_weights: torch.Tensor,   # [B, T]
    text_padding_weight: float = 1.0,
    text_padding_ids: set[int] | None = None,
) -> torch.Tensor:
    """
    Extended text CE with per-frame segment weights.

    IMPORTANT: text_logits comes from _undelay_sequence(..., fill_value=NaN).
    Masked-out positions contain ACTUAL NaN, not zero. NaN must be
    sanitized BEFORE cross_entropy — NaN * 0 == NaN in IEEE754, so
    multiplying by a zero weight AFTER the fact does not clean it, and
    can leak NaN gradients into the backward graph even though the
    forward value looks correct via torch.where.
    """
    mask_2d = text_mask[:, 0, :].float()
    weights = seg_weights * mask_2d   # [B, T]

    if text_padding_ids is not None:
        codes_2d = text_codes[:, 0, :]
        for pad_id in text_padding_ids:
            weights[codes_2d == pad_id] *= text_padding_weight

    logits_flat = text_logits[:, 0].reshape(-1, text_logits.size(-1)).float()
    logits_flat = torch.nan_to_num(logits_flat, nan=0.0, posinf=0.0, neginf=0.0)  # sanitize BEFORE CE

    target_flat = text_codes[:, 0].reshape(-1)
    weights_flat = weights.reshape(-1)

    target_safe = torch.where(
        weights_flat > 0, target_flat, torch.zeros_like(target_flat),
    )

    loss_per_tok = F.cross_entropy(logits_flat, target_safe, reduction="none")
    loss_per_tok = loss_per_tok * weights_flat   # safe now: no NaN operand anywhere
    return loss_per_tok.sum() / weights_flat.sum().clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Audio loss
# ---------------------------------------------------------------------------

def compute_audio_loss(
    logits: torch.Tensor,        # [B, dep_q, T, card] — also contains real NaN at masked positions
    audio_codes: torch.Tensor,   # [B, dep_q, T]
    mask: torch.Tensor,          # [B, dep_q, T]
    first_codebook_weight_multiplier: float = 1.0,
) -> torch.Tensor:
    """Same NaN-before-CE fix as compute_text_loss_rag — logits come from the
    same _undelay_sequence(..., fill_value=NaN) call."""
    target  = torch.where(mask, audio_codes, torch.zeros_like(audio_codes))
    weights = mask.float()
    weights[:, 0] *= first_codebook_weight_multiplier

    logits_flat = logits.reshape(-1, logits.size(-1)).float()
    logits_flat = torch.nan_to_num(logits_flat, nan=0.0, posinf=0.0, neginf=0.0)  # sanitize BEFORE CE

    target_flat  = target.reshape(-1)
    weights_flat = weights.reshape(-1)

    loss_per_tok = F.cross_entropy(logits_flat, target_flat, reduction="none")
    loss_per_tok = loss_per_tok * weights_flat
    return loss_per_tok.sum() / weights_flat.sum().clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Combined RAG loss — 2-tuple return, matches both train_rag.py and
# train_rag_resampler.py's `text_loss, audio_loss = compute_rag_losses(...)`
# ---------------------------------------------------------------------------

def compute_rag_losses(
    model,
    output,
    codes: torch.Tensor,
    seg_metas: list[SegmentMeta],
    args,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, K, T = codes.shape
    device = codes.device
    text_padding_ids = {model.text_padding_token_id, model.end_of_text_padding_id}

    seg_weights = make_segment_weight_tensor(
        B=B, T=T, seg_metas=seg_metas, device=device,
        ref_weight=getattr(args, "ref_token_weight", 5.0),
        lead_weight=getattr(args, "lead_weight", 1.0),
        filler_weight=getattr(args, "filler_weight", 0.5),
        ref_context_weight=getattr(args, "ref_context_weight", 1.0),
        body_weight=getattr(args, "body_weight", 2.0),
        no_rag_weight=getattr(args, "no_rag_weight", 1.0),
    )

    text_loss = compute_text_loss_rag(
        text_logits=output.text_logits,
        text_codes=codes[:, :model.audio_offset],
        text_mask=output.text_mask,
        seg_weights=seg_weights,
        text_padding_weight=getattr(args, "text_padding_weight", 1.0),
        text_padding_ids=text_padding_ids,
    )

    audio_loss = compute_audio_loss(
        logits=output.logits,
        audio_codes=codes[:, model.audio_offset : model.audio_offset + model.dep_q],
        mask=output.mask,
        first_codebook_weight_multiplier=getattr(args, "first_codebook_weight_multiplier", 1.0),
    )

    return text_loss, audio_loss
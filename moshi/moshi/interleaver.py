"""
interleaver.py  (modified for RAG fine-tuning — literal <ref> text approach)
=============================================================================
<ref> is treated exactly like <system>: plain text through the ordinary
tokenizer, wrapped as "<ref> ... <ref>", tokenized normally, and spliced
into the text codebook at filler_end+1 (right after the model's filler
segment, before the body/grounding segment). No vocabulary surgery, no
new embedding row — the model learns the convention purely from training
exposure, same as it already learned <system>.

Changes from original
----------------------
1. `Sample` carries an optional `segment_meta: SegmentMeta | None` field.
2. `InterleavedTokenizer.__call__` reads segment_meta + reference text from
   the sidecar JSON, splices a <ref>...<ref> block at filler_end+1, and
   shifts body_start by the spliced block's length.
3. `Batch` carries `segment_metas: list[SegmentMeta | None]`.
4. Reference length is capped via `max_ref_tokens` at construction time —
   bounds worst-case sequence growth regardless of source reference length.
"""

from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import math
from collections import deque
from dataclasses import dataclass, field
from functools import reduce
from typing import Optional

import numpy as np
import sentencepiece
import torch
from .conditioners import ConditionAttributes

from .models.lm import SILENCE_TOKENS, SINE_TOKENS, AUDIO_TOKENS_PER_STREAM

import logging
logger = logging.getLogger(__name__)

class _DiagCounter:
    """Rate-limited diagnostic logging: prints full detail for the first
    `verbose_n` samples of each category, then just counts silently so
    training isn't spammed once you've confirmed the pipeline is sane."""
    def __init__(self, verbose_n: int = 5):
        self.verbose_n = verbose_n
        self.counts: dict[str, int] = {}

    def hit(self, category: str) -> bool:
        n = self.counts.get(category, 0)
        self.counts[category] = n + 1
        return n < self.verbose_n

    def summary(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items()))


_diag = _DiagCounter()

def wrap_with_system_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def wrap_with_ref_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<ref>") and cleaned.endswith("<ref>"):
        return cleaned
    return f"<ref> {cleaned} <ref>"

def wrap_with_lookup_tags(text: str = "") -> str:
    return "<lookup>"


# ---------------------------------------------------------------------------
# SegmentMeta
# ---------------------------------------------------------------------------

@dataclass
class TurnSpan:
    kind: str          # "regular" | "lead" | "lookup" | "filler" | "ref" | "body"
    start_frame: int
    end_frame: int


@dataclass
class SegmentMeta:
    turns: list[TurnSpan] = field(default_factory=list)
    n_ref_tokens: int = 0     # length of spliced <ref> block, in frames
    n_lookup_tokens: int = 0  # length of spliced <lookup> block, in frames
    batch_idx: int = 0

    def spans_of(self, kind: str) -> list[TurnSpan]:
        return [t for t in self.turns if t.kind == kind]


# ---------------------------------------------------------------------------
# Sample and Batch
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    codes: torch.Tensor                           # [1, K, T]
    condition_attributes: ConditionAttributes | None = None
    segment_meta: SegmentMeta | None = None       # None for non-RAG turns


@dataclass
class Batch:
    codes: torch.Tensor
    condition_attributes: list[ConditionAttributes] | None = None
    segment_metas: list[SegmentMeta | None] = field(default_factory=list)

    @classmethod
    def collate(cls, batch: list[Sample], pad_value: int) -> "Batch":
        """Samples can have different T (ref/system-prompt-bearing samples
        are longer), so shorter ones are right-padded to the batch max
        before concatenation. pad_value is the sentinel compute_audio_loss /
        compute_text_loss_rag treat as 'ignore this frame'."""
        max_T = max(b.codes.shape[-1] for b in batch)
        padded_codes = []
        for b in batch:
            T = b.codes.shape[-1]
            if T < max_T:
                pad = torch.full(
                    (1, b.codes.shape[1], max_T - T),
                    pad_value, dtype=b.codes.dtype, device=b.codes.device,
                )
                padded_codes.append(torch.cat([b.codes, pad], dim=-1))
            else:
                padded_codes.append(b.codes)
        codes = torch.cat(padded_codes, dim=0)

        if any(b.condition_attributes is not None for b in batch):
            condition_attributes = [b.condition_attributes for b in batch]
        else:
            condition_attributes = None

        segment_metas: list[SegmentMeta | None] = []
        for i, sample in enumerate(batch):
            if sample.segment_meta is not None:
                sample.segment_meta.batch_idx = i
            segment_metas.append(sample.segment_meta)

        return Batch(codes, condition_attributes, segment_metas)

    @property
    def rag_segment_metas(self) -> list[SegmentMeta]:
        return [m for m in self.segment_metas if m is not None]


# ---------------------------------------------------------------------------
# Tokenisation utilities (unchanged)
# ---------------------------------------------------------------------------

Alignment = tuple[str, tuple[float, float], str]
TokenizedAlignment = tuple[list[int], tuple[float, float], str]


def tokenize(
    tokenizer: sentencepiece.SentencePieceProcessor,
    text: str,
    bos: bool = True,
    alpha: float | None = None,
):
    nl_piece = tokenizer.encode("\n")[-1]
    if alpha is not None:
        tokens = tokenizer.encode(
            text.split("\n"), enable_sampling=True, alpha=alpha, nbest_size=-1
        )
    else:
        tokens = tokenizer.encode(text.split("\n"))
    tokens = reduce(lambda a, b: [*a, nl_piece, *b], tokens)
    if bos:
        tokens = [tokenizer.bos_id(), *tokens]
    return tokens


# ---------------------------------------------------------------------------
# Interleaver (unchanged)
# ---------------------------------------------------------------------------

class Interleaver:
    def __init__(
        self,
        tokenizer: sentencepiece.SentencePieceProcessor,
        audio_frame_rate: float,
        text_padding: int,
        end_of_text_padding: int,
        zero_padding: int,
        in_word_padding: int | None = None,
        keep_main_only: bool = False,
        main_speaker_label: str = "SPEAKER_MAIN",
        use_bos_eos: bool = False,
        keep_and_shift: bool = False,
        audio_delay: float = 0.0,
        proba: float = 1.0,
        device: str | torch.device = "cuda",
    ):
        self.tokenizer = tokenizer
        self.audio_frame_rate = audio_frame_rate
        self.text_padding = text_padding
        self.end_of_text_padding = end_of_text_padding
        self.zero_padding = zero_padding
        self.in_word_padding = (
            self.text_padding if in_word_padding is None else in_word_padding
        )
        self.keep_main_only = keep_main_only
        self.main_speaker_label = main_speaker_label
        self.use_bos_eos = use_bos_eos
        self.keep_and_shift = keep_and_shift
        self.audio_delay = audio_delay
        self.proba = proba
        self.device = device

    @property
    def special_tokens(self) -> set[int]:
        return {
            self.text_padding, self.end_of_text_padding,
            self.tokenizer.bos_id(), self.tokenizer.eos_id(),
            self.zero_padding, self.in_word_padding,
        }

    def _tokenize(self, alignments: list[Alignment]) -> list[TokenizedAlignment]:
        out = []
        for word, ts, speaker in alignments:
            toks = tokenize(self.tokenizer, word.strip(), bos=False)
            out.append((toks, ts, speaker))
        return out

    def _keep_main_only(self, alignments, main_speaker):
        return [a for a in alignments if a[2] == main_speaker]

    def _keep_those_with_duration(self, alignments):
        return [a for a in alignments if a[1][0] < a[1][1]]

    def _add_delay(self, alignments):
        return [
            (a[0], (a[1][0] - self.audio_delay, a[1][1] - self.audio_delay), a[2])
            for a in alignments if a[1][1] > self.audio_delay
        ]

    def _insert_bos_eos(self, alignments, main_speaker):
        out: list[TokenizedAlignment] = []
        last_speaker = None
        for toks, ts, speaker in alignments:
            toks = list(toks)
            if speaker == last_speaker:
                pass
            elif speaker == main_speaker:
                toks.insert(0, self.tokenizer.bos_id())
            elif last_speaker == main_speaker:
                assert out
                toks.insert(0, self.tokenizer.eos_id())
            last_speaker = speaker
            out.append((toks, ts, speaker))
        return out

    def build_token_stream(self, alignments, segment_duration: float) -> torch.Tensor:
        T = math.ceil(segment_duration * self.audio_frame_rate)
        if alignments is None:
            text_tokens = [self.zero_padding] * T
        else:
            text_tokens = [self.text_padding] * T
            i = 0
            to_append_stack: deque = deque()
            last_word_end = -1
            for t in range(T):
                while (
                    i < len(alignments)
                    and alignments[i][1][0] * self.audio_frame_rate < t + 1
                ):
                    tokenized = alignments[i][0]
                    last_word_end = int(alignments[i][1][1] * self.audio_frame_rate)
                    if self.keep_and_shift:
                        to_append_stack.extend(tokenized)
                    else:
                        to_append_stack = deque(tokenized)
                    i += 1
                if to_append_stack:
                    if t > 0 and text_tokens[t - 1] in [self.text_padding, self.in_word_padding]:
                        text_tokens[t - 1] = self.end_of_text_padding
                    next_token = to_append_stack.popleft()
                    text_tokens[t] = next_token
                elif t <= last_word_end:
                    text_tokens[t] = self.in_word_padding
        if self.audio_delay < 0:
            prefix_length = int(self.audio_frame_rate * -self.audio_delay)
            text_tokens[:prefix_length] = [self.zero_padding] * prefix_length
        return torch.tensor(text_tokens, device=self.device).view(1, 1, -1)

    def prepare_item(self, alignments, segment_duration: float, main_speaker: str | None = None):
        if alignments is None:
            tokenized = None
        else:
            tokenized = self._tokenize(sorted(alignments, key=lambda x: x[1][0]))
            if self.keep_main_only:
                main_speaker = main_speaker or self.main_speaker_label
                tokenized = self._keep_main_only(tokenized, main_speaker)
            elif self.use_bos_eos:
                main_speaker = main_speaker or self.main_speaker_label
                tokenized = self._insert_bos_eos(tokenized, main_speaker)
            tokenized = self._keep_those_with_duration(tokenized)
            if self.audio_delay != 0:
                tokenized = self._add_delay(tokenized)
        return self.build_token_stream(tokenized, segment_duration)


def dicho(alignment, val, i=0, j=None):
    if j is None:
        j = len(alignment)
    if i == j:
        return i
    k = (i + j) // 2
    if alignment[k][1][0] < val:
        return dicho(alignment, val, k + 1, j)
    else:
        return dicho(alignment, val, i, k)


# ---------------------------------------------------------------------------
# InterleavedTokenizer
# ---------------------------------------------------------------------------

class InterleavedTokenizer:
    def __init__(
        self,
        mimi,
        interleaver: Interleaver,
        duration_sec: float,
        max_ref_tokens: int | None = 150,
    ):
        self.mimi = mimi
        self.interleaver = interleaver
        self.duration_sec = duration_sec
        self.max_ref_tokens = max_ref_tokens
        self.num_audio_frames = math.ceil(duration_sec * mimi.frame_rate)

    def _sec_to_frame(self, sec: float) -> int:
        return int(sec * self.mimi.frame_rate)

    def _build_forced_text_frames(self, wrapped_text: str, n_q: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared by system-prompt and <ref>-block injection: one text token
        per frame, agent audio codebooks forced silent, user codebooks sine.
        This is the ONLY place this construction lives — do not duplicate it."""
        tokens = self.interleaver.tokenizer.encode(wrapped_text)
        n = len(tokens)
        device = self.interleaver.device
        text = torch.tensor(tokens, device=device).view(1, 1, n)

        silence = torch.as_tensor(SILENCE_TOKENS, dtype=torch.long, device=device)
        sine    = torch.as_tensor(SINE_TOKENS,    dtype=torch.long, device=device)
        agent_frames = silence.view(1, AUDIO_TOKENS_PER_STREAM, 1).expand(-1, -1, n)
        user_frames  = sine.view(1, AUDIO_TOKENS_PER_STREAM, 1).expand(-1, -1, n)
        remaining = n_q - 2 * AUDIO_TOKENS_PER_STREAM
        if remaining > 0:
            pad = torch.full((1, remaining, n), self.interleaver.zero_padding, dtype=torch.long, device=device)
            audio = torch.cat([agent_frames, user_frames, pad], dim=1)
        else:
            audio = torch.cat([agent_frames, user_frames], dim=1)
        return text, audio

    def _build_system_prompt_frames(self, system_prompt: str, n_q: int):
        return self._build_forced_text_frames(wrap_with_system_tags(system_prompt), n_q)

    def _build_ref_block_frames(self, reference: str, n_q: int):
        """Caps reference length at max_ref_tokens BEFORE wrapping+encoding,
        so the closing <ref> tag is never truncated away, and the encoded
        wrapped block always matches exactly what inference will inject."""
        content = reference.strip()
        if self.max_ref_tokens is not None:
            ids = self.interleaver.tokenizer.encode(content)
            if len(ids) > self.max_ref_tokens:
                ids = ids[: self.max_ref_tokens]
                content = self.interleaver.tokenizer.decode(ids)
        return self._build_forced_text_frames(wrap_with_ref_tags(content), n_q)

    def _parse_segment_meta(self, raw_meta: dict, chunk_start_sec: float) -> SegmentMeta | None:
        """Consumes the new {'turns': [{kind, start_sec, end_sec}, ...]} format.
        Turns are clipped to the current [chunk_start_sec, chunk_end_sec) window;
        turns entirely outside it are dropped. Frame indices are relative to the
        chunk, matching build_token_stream's own indexing."""
        turns_raw = raw_meta.get("turns")
        if not turns_raw:
            return None
    
        chunk_end_sec = chunk_start_sec + self.duration_sec
    
        def to_frame(sec: float) -> int:
            return self._sec_to_frame(max(0.0, min(sec, chunk_end_sec) - chunk_start_sec))
    
        spans: list[TurnSpan] = []
        for t in turns_raw:
            t_start, t_end = t["start_sec"], t["end_sec"]
            if t_end <= chunk_start_sec or t_start >= chunk_end_sec:
                continue  # entirely outside this chunk
            spans.append(TurnSpan(
                kind=t["kind"],
                start_frame=to_frame(max(t_start, chunk_start_sec)),
                end_frame=to_frame(min(t_end, chunk_end_sec)),
            ))
    
        if not spans:
            return None
    
        # sanity: RAG-turn sub-kinds must be internally ordered lead < lookup 
        # filler < ref < body if present at all, since splicing logic below
        # depends on it.
        order = {"lead": 0, "lookup": 1, "filler": 2, "ref": 3, "body": 4}
        rag_spans = sorted(
            (s for s in spans if s.kind in order), key=lambda s: order[s.kind]
        )
        for a, b in zip(rag_spans, rag_spans[1:]):
            if a.end_frame > b.start_frame:
                logger.warning(
                    f"segment_meta turn ordering looks off: {a.kind} ends at "
                    f"{a.end_frame} but {b.kind} starts at {b.start_frame}"
                )
    
        return SegmentMeta(turns=spans)

    def __call__(self, wav: np.ndarray, start_sec: float, path: str) -> Sample:
        with torch.no_grad():
            audio_tensor = torch.Tensor(wav).cuda()
            audio_tokens = self.mimi.encode(audio_tensor[:, None])
            audio_tokens = audio_tokens[..., : self.num_audio_frames]
            this_num_audio_frames = audio_tokens.shape[-1]
            audio_tokens = torch.nn.functional.pad(
                audio_tokens[..., : self.num_audio_frames],
                (0, self.num_audio_frames - this_num_audio_frames),
                value=self.interleaver.zero_padding,
            )
            audio_tokens = audio_tokens.view(1, -1, self.num_audio_frames)

            info_file = os.path.splitext(path)[0] + ".new.combined.json"
            with open(info_file) as f:
                data = json.load(f)

            alignments = data["alignments"]
            start_alignment = dicho(alignments, start_sec)
            end_alignment   = dicho(alignments, start_sec + self.duration_sec)
            alignments = [
                (a[0], (a[1][0] - start_sec, a[1][1] - start_sec), a[2])
                for a in alignments[start_alignment:end_alignment]
            ]

            text_tokens = self.interleaver.prepare_item(alignments, this_num_audio_frames)
            text_tokens = torch.nn.functional.pad(
                text_tokens,
                (0, self.num_audio_frames - text_tokens.shape[-1]),
                value=self.interleaver.zero_padding,
            )

            condition_attributes = data.get("text_conditions", None)
            reference_text = ""
            if isinstance(condition_attributes, dict):
                reference_text = condition_attributes.get("reference", "") or ""

            raw_meta = data.get("segment_meta")
            segment_meta: SegmentMeta | None = None
            if raw_meta is not None:
                segment_meta = self._parse_segment_meta(raw_meta, start_sec)

            condition_attributes = data.get("text_conditions", None)
            reference_text = ""
            if isinstance(condition_attributes, dict):
                reference_text = condition_attributes.get("reference", "") or ""

            # ── DIAGNOSTIC: what kind of sample is this, before injection ──
            has_regular = segment_meta is not None and segment_meta.spans_of("regular")
            has_rag_spans = segment_meta is not None and segment_meta.spans_of("lookup") and segment_meta.spans_of("filler") and segment_meta.spans_of("body")

            if segment_meta is None:
                if _diag.hit("no_segment_meta"):
                    logger.info(f"[diag] {path}@{start_sec:.1f}s: no segment_meta (raw)")
            elif not has_rag_spans and has_regular:
                if _diag.hit("regular_only"):
                    kinds = [s.kind for s in segment_meta.turns]
                    logger.info(f"[diag] {path}@{start_sec:.1f}s: REGULAR-ONLY sample, {len(kinds)} spans {kinds[:6]}...")
            elif has_rag_spans and not reference_text:
                if _diag.hit("rag_spans_no_reference"):
                    logger.warning(f"[diag] {path}@{start_sec:.1f}s: has lookup/filler/body spans "
                                    f"but NO reference_text — injection will be SKIPPED silently")

            injected_lookup = False
            injected_ref = False

            if segment_meta is not None and reference_text:
                lookup_spans = segment_meta.spans_of("lookup")
                filler_spans = segment_meta.spans_of("filler")
                body_spans   = segment_meta.spans_of("body")

                if lookup_spans and filler_spans and body_spans:
                    lookup_span = lookup_spans[0]
                    filler_span = filler_spans[0]
                    body_span   = body_spans[0]

                    if filler_span.end_frame < this_num_audio_frames:
                        n_q = audio_tokens.shape[1]

                        lookup_text, lookup_audio = self._build_forced_text_frames(
                            wrap_with_lookup_tags(), n_q=n_q
                        )
                        n_lookup = lookup_text.shape[-1]
                        insert_at = lookup_span.start_frame
                        text_tokens = torch.cat([text_tokens[..., :insert_at], lookup_text, text_tokens[..., insert_at:]], dim=-1)
                        audio_tokens = torch.cat([audio_tokens[..., :insert_at], lookup_audio, audio_tokens[..., insert_at:]], dim=-1)
                        segment_meta = self._shift_spans_from(segment_meta, insert_at, n_lookup)
                        segment_meta.n_lookup_tokens = n_lookup
                        injected_lookup = True

                        filler_span = segment_meta.spans_of("filler")[0]
                        body_span   = segment_meta.spans_of("body")[0]

                        ref_text, ref_audio = self._build_ref_block_frames(reference_text, n_q=n_q)
                        n_ref = ref_text.shape[-1]
                        insert_at = filler_span.end_frame + 1
                        text_tokens = torch.cat([text_tokens[..., :insert_at], ref_text, text_tokens[..., insert_at:]], dim=-1)
                        audio_tokens = torch.cat([audio_tokens[..., :insert_at], ref_audio, audio_tokens[..., insert_at:]], dim=-1)
                        segment_meta = self._shift_spans_from(segment_meta, insert_at, n_ref)
                        segment_meta.n_ref_tokens = n_ref
                        injected_ref = True

                        if _diag.hit("rag_injected"):
                            logger.info(
                                f"[diag] {path}@{start_sec:.1f}s: INJECTED "
                                f"<lookup>({n_lookup}tok) + <ref>({n_ref}tok, "
                                f"ref_text={reference_text[:60]!r}...)"
                            )
                    else:
                        if _diag.hit("filler_past_chunk_end"):
                            logger.warning(
                                f"[diag] {path}@{start_sec:.1f}s: filler_span.end_frame "
                                f"({filler_span.end_frame}) >= chunk frames "
                                f"({this_num_audio_frames}) — injection SKIPPED, "
                                f"chunk boundary likely split the RAG turn"
                            )
                elif reference_text:
                    if _diag.hit("reference_no_spans"):
                        logger.warning(
                            f"[diag] {path}@{start_sec:.1f}s: has reference_text but missing "
                            f"one of lookup/filler/body spans — injection SKIPPED"
                        )

            system_prompt = data.get("metadata", {}).get("system_prompt")
            n_prompt = 0
            if system_prompt and start_sec == 0.0:
                sys_text, sys_audio = self._build_system_prompt_frames(system_prompt, n_q=audio_tokens.shape[1])
                n_prompt = sys_text.shape[-1]
                text_tokens  = torch.cat([sys_text,  text_tokens],  dim=-1)
                audio_tokens = torch.cat([sys_audio, audio_tokens], dim=-1)
                if segment_meta is not None and n_prompt:
                    segment_meta = self._shift_spans_from(segment_meta, 0, n_prompt)

            codes = torch.cat([text_tokens, audio_tokens], dim=1)
            return Sample(codes=codes, condition_attributes=condition_attributes, segment_meta=segment_meta)

    @staticmethod
    def _shift_spans_from(meta: SegmentMeta, insert_at: int, n_inserted: int) -> SegmentMeta:
        """Every span boundary at or after insert_at shifts right by
        n_inserted frames. Spans that straddle insert_at get their end
        pushed but not their start (they contain the insertion)."""
        new_turns = []
        for t in meta.turns:
            start = t.start_frame + n_inserted if t.start_frame >= insert_at else t.start_frame
            end   = t.end_frame   + n_inserted if t.end_frame   >= insert_at else t.end_frame
            new_turns.append(TurnSpan(kind=t.kind, start_frame=start, end_frame=end))
        meta.turns = new_turns
        return meta
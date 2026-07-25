# rag_index.py
"""
Loads a pre-built index and provides:
    - retrieve(): cb0 histogram similarity against user utterance
    - inject_as_text_prompt(): mirrors _step_text_prompt_core exactly
    - context reporting: shows what was injected and its window status
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from .models.lm import SILENCE_TOKENS, SINE_TOKENS, FRAME_RATE_HZ


@dataclass
class SemanticChunk:
    id: int
    text: str
    source: str
    text_tokens: list[int]        # Moshi tokenizer IDs — for text injection
    semantic_codes: np.ndarray    # (F,) cb0 only — for retrieval
    similarity_score: float = 0.0 # filled in at retrieval time


@dataclass
class InjectionRecord:
    """Records what was injected and when, for context window reporting."""
    chunk_id: int
    text: str
    source: str
    n_tokens: int
    injected_at_frame: int        # lm_gen offset when injection started
    similarity_score: float


class SemanticRAGIndex:
    def __init__(
        self,
        index_dir: str,
        vocab_size: int = 2048,
        token_budget: int = 150,          # max tokens injected per retrieval
        context_window_frames: int = 750,
    ):
        self.vocab_size = vocab_size
        self.token_budget = token_budget
        self.context_window_frames = context_window_frames

        self.chunks: list[SemanticChunk] = []
        self._histograms: list[np.ndarray] = []
        self.injection_log: list[InjectionRecord] = []

        self._load(index_dir)

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self, index_dir: str):
        manifest_path = os.path.join(index_dir, "manifest.json")
        chunks_path   = os.path.join(index_dir, "chunks.npz")

        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        arrays = np.load(chunks_path)

        for entry in manifest["chunks"]:
            cid = entry["id"]
            chunk = SemanticChunk(
                id=cid,
                text=entry["text"],
                source=entry["source"],
                text_tokens=entry["text_tokens"],
                semantic_codes=arrays[f"cb0_{cid}"],
            )
            self.chunks.append(chunk)
            self._histograms.append(self._to_histogram(chunk.semantic_codes))

        print(f"[RAGIndex] loaded {len(self.chunks)} chunks from {index_dir}")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _to_histogram(self, codes: np.ndarray) -> np.ndarray:
        SILENCE_CB0 = 948
        filtered = codes[codes != SILENCE_CB0]
        if len(filtered) == 0:
            return np.zeros(self.vocab_size, dtype=np.float32)
        hist = np.bincount(
            filtered.astype(np.int64), minlength=self.vocab_size
        ).astype(np.float32)
        norm = hist.sum()
        if norm > 0:
            hist /= norm
        return hist

    def retrieve(self, query_cb0: np.ndarray, top_k: int = 2) -> list[SemanticChunk]:
        if not self.chunks:
            return []
    
        # Filter out silence tokens before building query histogram
        # Silence token 948 dominates and dilutes the signal
        SILENCE_CB0 = 948
        filtered_query = query_cb0[query_cb0 != SILENCE_CB0]
    
        if len(filtered_query) == 0:
            log("info", "[RAGIndex] query is all silence, no retrieval")
            return []
    
        query_hist = self._to_histogram(filtered_query)
    
        scores = np.array([
            np.minimum(query_hist, h).sum()
            for h in self._histograms
        ])
    
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = self.chunks[idx]
                chunk.similarity_score = float(scores[idx])
                results.append(chunk)
    
        return results

    # ── Injection — mirrors _step_text_prompt_core exactly ────────────────────

    def inject_as_text_prompt(
    self,
    lm_gen,
    chunks: list[SemanticChunk],
    current_frame: int,
    tokenizer=None,    # pass in for live decode confirmation
):
        device = lm_gen.lm_model.device
        silence = torch.as_tensor(
            SILENCE_TOKENS, dtype=torch.long, device=device
        ).view(1, 8, 1)
        sine = torch.as_tensor(
            SINE_TOKENS, dtype=torch.long, device=device
        ).view(1, 8, 1)
    
        injected_total = 0
    
        for chunk in chunks:
            tokens_to_inject = chunk.text_tokens
            space = self.token_budget - injected_total
            if space <= 0:
                break
            tokens_to_inject = tokens_to_inject[:space]
    
            # Decode exactly what is going into the model
            if tokenizer is not None:
                injected_text = tokenizer.decode(tokens_to_inject)
            else:
                injected_text = chunk.text
    
            print(
                f"[RAG INJECT] chunk [{chunk.id}]  "
                f"score={chunk.similarity_score:.3f}  "
                f"tokens={len(tokens_to_inject)}\n"
                f'  text → "{injected_text}"'
            )
    
            for token_id in tokens_to_inject:
                lm_gen.step(
                    moshi_tokens=silence,
                    text_token=token_id,
                    input_tokens=sine,
                )
    
            injected_total += len(tokens_to_inject)
    
            self.injection_log.append(InjectionRecord(
                chunk_id=chunk.id,
                text=injected_text,    # store decoded text not original
                source=chunk.source,
                n_tokens=len(tokens_to_inject),
                injected_at_frame=current_frame,
                similarity_score=chunk.similarity_score,
            ))
    
        return injected_total

    # ── Context window reporting ───────────────────────────────────────────────

    def context_report(self, current_frame: int) -> str:
        if not self.injection_log:
            return "[RAGIndex] no context injected yet"
    
        # oldest frame still inside the attention window
        oldest_in_window = max(0, current_frame - self.context_window_frames)
    
        lines = [
            "┌─ RAG CONTEXT LOG " + "─" * 51,
            f"│  current frame : {current_frame}",
            f"│  window oldest : {oldest_in_window}",
            f"│  window size   : {self.context_window_frames} frames "
            f"({self.context_window_frames / FRAME_RATE_HZ:.0f}s)",
            "│",
        ]
    
        for rec in self.injection_log:
            # The chunk occupies frames [injected_at_frame, injected_at_frame + n_tokens)
            # It exits the window when injected_at_frame + n_tokens < oldest_in_window
            inject_start = rec.injected_at_frame
            inject_end   = rec.injected_at_frame + rec.n_tokens
    
            in_window = inject_end > oldest_in_window
    
            # How many more live frames until this chunk exits the window:
            # chunk exits when current_frame - context_window >= inject_end
            # i.e. current_frame >= inject_end + context_window
            # frames remaining = (inject_end + context_window) - current_frame
            frames_left = max(0, (inject_end + self.context_window_frames) - current_frame)
    
            status = (
                f"✓ in window  ({frames_left} frames / "
                f"{frames_left / FRAME_RATE_HZ:.1f}s left)"
                if in_window else "✗ EXPIRED"
            )
    
            lines += [
                f"│  chunk [{rec.chunk_id}]  score={rec.similarity_score:.3f}  "
                f"source={rec.source}",
                f"│    injected at frame {inject_start}  "
                f"tokens={rec.n_tokens}  {status}",
                f'│    "{rec.text[:80]}{"..." if len(rec.text) > 80 else ""}"',
                "│",
            ]
    
        lines.append("└" + "─" * 69)
        return "\n".join(lines)
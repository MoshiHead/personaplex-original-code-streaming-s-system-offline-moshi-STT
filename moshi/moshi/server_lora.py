# server.py — RAG-aware inference server with mid-stream injection
#
# Key differences from the original:
#   1. Loads LoRA fine-tuned Moshi + ReferenceAdapter at startup.
#      LoRA weights are merged into the base model for zero-overhead inference.
#   2. RAG is triggered by the model emitting <ref> token in its text stream
#      (learned behavior from fine-tuning), NOT by VAD reset.
#   3. Injection is mid-stream: a background task retrieves → compresses with
#      a 1B LLM → encodes with ReferenceAdapter → patches embed_codes() output.
#      The model's KV cache is never reset during a session.
#   4. A 1B parameter LLM (e.g. Qwen2.5-1.5B-Instruct or SmolLM2-1.7B) runs
#      on CPU/secondary GPU and compresses retrieved chunks + conversation
#      context into a 1-2 sentence grounding statement in ~300-500ms.
#   5. VAD still detects end-of-utterance for transcript accumulation, but
#      never triggers a model reset.
#   6. The original RAG-via-reset path is completely removed.

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import random
import os
from pathlib import Path
import tarfile
import time
import secrets
import gc
import sys
import threading
from typing import Literal, Optional
import json

import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import sphn
import torch
import torch.nn as nn
import torch.nn.functional as F

from .client_utils import make_log, colorize
from .models import loaders, MimiModel, LMModel, LMGen
from .utils.connection import create_ssl_context, get_lan_ip
from .utils.logging import setup_logger, ColorizedLog
from .build_index import text_to_audio
from moshi.models.loaders import CheckpointInfo

logger = setup_logger(__name__)
DeviceString = Literal["cuda"] | Literal["cpu"]


def torch_auto_device(requested: Optional[DeviceString] = None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def wrap_with_system_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


# ── RAG helpers ───────────────────────────────────────────────────────────────

def load_rag_index(index_dir: str):
    manifest_path = os.path.join(index_dir, "manifest.json")
    chunks_path   = os.path.join(index_dir, "chunks.npz")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest.get("mode") == "text", (
        f"Index at {index_dir} has mode='{manifest.get('mode')}'. "
        f"Rebuild with --mode text."
    )
    arrays = np.load(chunks_path)
    embeddings = arrays["text_embeddings"].astype(np.float32)
    chunks = manifest["chunks"]
    embedding_model_name = manifest.get("embedding_model", "all-MiniLM-L6-v2")
    return chunks, embeddings, embedding_model_name


def retrieve_chunks(
    transcript: str,
    chunks: list,
    embeddings: np.ndarray,
    embedding_model,
    top_k: int,
    min_score: float = 0.45,
) -> list[dict]:
    if not transcript.strip():
        return []
    query_vec = embedding_model.encode(
        transcript, normalize_embeddings=True
    ).astype(np.float32)
    scores = embeddings @ query_vec
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score < min_score:
            continue
        chunk = dict(chunks[idx])
        chunk["similarity_score"] = score
        results.append(chunk)
    return results


def retrieve_chunks_hierarchical(
    transcript: str,
    chunks: list,
    embeddings: np.ndarray,
    embedding_model,
    top_k: int,
    min_score: float = 0.45,
) -> list[dict]:
    """Retrieve top-k chunks then expand each hit to include its immediate
    next neighbor from the same source, recovering context that fixed-size
    chunking may have split across boundaries."""
    base_hits = retrieve_chunks(
        transcript, chunks, embeddings, embedding_model, top_k, min_score
    )
    if not base_hits:
        return base_hits

    # Build per-source ordered list: source -> [(global_idx, chunk), ...]
    by_source: dict[str, list] = {}
    for i, c in enumerate(chunks):
        by_source.setdefault(c["source"], []).append((i, c))

    expanded: list[dict] = []
    seen_ids: set = set()
    for hit in base_hits:
        src = hit["source"]
        siblings = by_source[src]
        # Find this chunk's position within its source
        pos = next(j for j, (_, c) in enumerate(siblings) if c["id"] == hit["id"])
        # Include the hit and the immediately following sibling
        for p in [pos, pos + 1]:
            if p >= len(siblings):
                continue
            _, c = siblings[p]
            if c["id"] not in seen_ids:
                merged = dict(c)
                merged["similarity_score"] = hit["similarity_score"]
                expanded.append(merged)
                seen_ids.add(c["id"])
    return expanded


def summarize_context(
    transcript: str,
    retrieved_chunks: list[dict],
    embedding_model,
    max_sentences: int = 3,
    max_chars: int = 400,
) -> str:
    """Extractive summarization: score each sentence by cosine similarity to
    the query, keep the top-N in original document order, and truncate
    cleanly at a sentence boundary rather than mid-word."""
    import re
    full_text = " ".join(c["text"] for c in retrieved_chunks)
    sentences = re.split(r'(?<=[.!?])\s+', full_text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    if not sentences:
        return full_text[:max_chars]

    query_vec = embedding_model.encode(
        transcript, normalize_embeddings=True
    ).astype(np.float32)
    sent_vecs = embedding_model.encode(
        sentences, normalize_embeddings=True
    ).astype(np.float32)
    scores = sent_vecs @ query_vec

    # Preserve original order for readability
    top_indices = sorted(np.argsort(scores)[::-1][:max_sentences])
    summary = " ".join(sentences[i] for i in top_indices)
    return summary[:max_chars]


# ── STT helpers ───────────────────────────────────────────────────────────────

def decode_stt_tokens(
    text_tokens: list[torch.Tensor],
    tokenizer,
    padding_token_id: int,
) -> str:
    """
    Decode accumulated STT text tokens into a string.
    Filters out padding and special tokens.
    """
    if not text_tokens:
        return ""
    all_tokens = torch.cat(text_tokens, dim=-1)    # (1, 1, T)
    all_tokens = all_tokens.cpu().view(-1)
    # Filter padding
    valid = all_tokens[all_tokens > padding_token_id]
    if valid.numel() == 0:
        return ""
    return tokenizer.decode(valid.tolist())


# ---------------------------------------------------------------------------
# ReferenceAdapter — must match train_rag.py exactly
# ---------------------------------------------------------------------------

class ReferenceAdapter(nn.Module):
    """
    Converts a compressed reference string into a [dim] conditioning vector.
    Architecture and init must match train_rag.py exactly for weight loading.
    """
    def __init__(self, dim: int, text_emb: nn.Embedding):
        super().__init__()
        self.text_emb  = text_emb
        self.norm_in   = nn.LayerNorm(dim)
        self.fc1       = nn.Linear(dim, 1024, bias=True)
        self.fc2       = nn.Linear(1024, dim, bias=True)
        self.out_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.clamp(0, self.text_emb.num_embeddings - 1)
        embeds = self.text_emb(token_ids.unsqueeze(0))
        pooled = embeds.mean(dim=1).squeeze(0)
        x = self.norm_in(pooled)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x * self.out_scale


# ---------------------------------------------------------------------------
# StreamingSumInjector
# Thread-safe holder for the pending conditioning vector.
# Patched into lm_model.embed_codes so injection happens inside lm_gen.step()
# without any structural change to LMGen or LMModel.
# ---------------------------------------------------------------------------

class StreamingSumInjector:
    """
    Wraps LMModel.embed_codes to add a conditioning vector to the TEXT
    channel only (matching train_rag.py's embed_codes_split /
    inject_streaming_sum_text_channel), using persistent in-place buffers.

    set()/clear() must always be called from the same thread that runs
    lm_gen.step() (server.py's gpu_executor) to avoid a cross-stream race
    with the CUDA graph replay — see call sites below.
    """

    def __init__(self, lm_model: LMModel):
        self._raw_lm = lm_model
        # Resolve past a peft wrapper if present.
        core = getattr(lm_model, "base_model", lm_model)
        core = getattr(core, "model", core)
        self._lm = core

        dim = self._lm.dim
        param = next(self._lm.parameters())
        device = param.device
        dtype  = param.dtype
        
        self._buf_vec    = torch.zeros(dim, device=device, dtype=dtype)
        self._buf_active = torch.zeros(1, device=device, dtype=dtype)

        self._original_embed_codes = lm_model.embed_codes
        lm_model.embed_codes = self._patched_embed_codes

    def _patched_embed_codes(self, sequence: torch.Tensor) -> torch.Tensor:
        lm = self._lm
        audio_sum = None
        for cb_index in range(lm.num_audio_codebooks):
            audio_emb = lm.emb[cb_index](sequence[:, cb_index + lm.audio_offset])
            audio_sum = audio_emb if audio_sum is None else audio_sum + audio_emb
        text_emb = lm.text_emb(sequence[:, 0])

        # Text-channel-only injection — tensor multiply, not a Python `if`,
        # so it survives CUDA graph replay correctly.
        text_emb = text_emb + self._buf_vec.view(1, 1, -1) * self._buf_active.view(1, 1, 1)

        return audio_sum + text_emb

    def set(self, vec: torch.Tensor) -> None:
        """Must be called on the gpu_executor thread — see call sites."""
        self._buf_vec.copy_(vec.detach().to(dtype=self._buf_vec.dtype))
        self._buf_active.fill_(1.0)

    def clear(self) -> None:
        """Must be called on the gpu_executor thread — see call sites."""
        self._buf_active.fill_(0.0)

    def restore(self) -> None:
        self._raw_lm.embed_codes = self._original_embed_codes
        self.clear()
 
 
# ---------------------------------------------------------------------------
# 1B LLM compressor
# ---------------------------------------------------------------------------
 
class ContextCompressor:
    """
    Lightweight LLM (~1B params) that compresses retrieved chunks + conversation
    history into a 1-2 sentence grounding statement for the ReferenceAdapter.
 
    Runs on a separate thread so it doesn't block the GPU step loop.
    Inference: ~300-500ms on CPU, ~100ms on a secondary GPU.
 
    Supported model families (any ~1B instruction-tuned model works):
        Qwen/Qwen2.5-1.5B-Instruct      (recommended — strong at compression)
        HuggingFaceTB/SmolLM2-1.7B-Instruct
        TinyLlama/TinyLlama-1.1B-Chat-v1.0
        google/gemma-3-1b-it
    """
 
    COMPRESS_PROMPT = (
        "You are a concise grounding assistant. Given a user's question, "
        "recent conversation, and retrieved reference passages, write a single "
        "1-2 sentence factual summary that directly answers the question. "
        "Use only the provided passages. Never add outside knowledge. "
        "If the passages do not answer the question, output: NO_CONTEXT\n\n"
        "Recent conversation:\n{history}\n\n"
        "User question: {question}\n\n"
        "Retrieved passages:\n{passages}\n\n"
        "Grounding summary (1-2 sentences):"
    )
 
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 60,          # reduced from 80 — tighter budget
        quantize_4bit: bool = True,
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        logger.info(f"[compressor] loading {model_name} on {device} (4bit={quantize_4bit}) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))

        if quantize_4bit and device != "cpu":
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device,
                token=os.getenv("HF_TOKEN"),
            )
        else:
            # bitsandbytes 4-bit requires CUDA — CPU quantization isn't
            # meaningfully supported, so CPU always falls back to fp32,
            # which is exactly what produced the 35s compression above.
            if quantize_4bit and device == "cpu":
                logger.warning(
                    "[compressor] quantize_4bit requested but device=cpu — "
                    "bitsandbytes 4-bit needs CUDA. Falling back to fp32 on "
                    "CPU, which will be slow. Set --compressor-device to a "
                    "CUDA device to actually get 4-bit + real throughput."
                )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map=device,
                token=os.getenv("HF_TOKEN"),
            )
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="compressor")
        logger.info(f"[compressor] ready — {sum(p.numel() for p in self.model.parameters())/1e9:.2f}B params")
 
    def compress(
        self,
        question: str,
        history: list[tuple[str, str]],    # [(user, moshi), ...]
        chunks: list[dict],
    ) -> str:
        """
        Synchronous compression — call from a thread pool only.
 
        Returns compressed grounding string, or "" if no context.
        """
        if not chunks:
            return ""
 
        history_text = "\n".join(
            f"User: {u}\nMoshi: {m}" for u, m in history[-3:]
        ) if history else "None"
 
        passages = "\n".join(
            f"[{i+1}] {c['text'][:300]}" for i, c in enumerate(chunks[:2])
        )
 
        prompt = self.COMPRESS_PROMPT.format(
            history=history_text,
            question=question,
            passages=passages,
        )
 
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]
 
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,          # greedy — faster and more deterministic
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0, prompt_len:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
 
        if "NO_CONTEXT" in result or not result:
            return ""
        return result
 
    async def compress_async(
        self,
        question: str,
        history: list[tuple[str, str]],
        chunks: list[dict],
        loop: asyncio.AbstractEventLoop,
    ) -> str:
        """Non-blocking wrapper — runs compress() on the compressor's own thread."""
        return await loop.run_in_executor(
            self._executor,
            self.compress, question, history, chunks,
        )
 
 
# ---------------------------------------------------------------------------
# ServerState
# ---------------------------------------------------------------------------
 
@dataclass
class ServerState:
    def __init__(
        self,
        mimi: MimiModel,
        other_mimi: MimiModel,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        lm: LMModel,
        mimi_weight: str,
        device: str | torch.device,
        voice_prompt_dir: Optional[str] = None,
        rag_index_dir: Optional[str] = None,
        rag_top_k: int = 2,
        rag_min_score: float = 0.40,
        stt_hf_repo: str = "kyutai/stt-1b-en_fr-candle",
        vad_threshold: float = 0.5,
        compressor_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        compressor_device: str = "cpu",
    ):
        self.mimi       = mimi
        self.other_mimi = other_mimi
        self.text_tokenizer = text_tokenizer
        self.device     = device
        self.voice_prompt_dir = voice_prompt_dir
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
 
        self.lm_gen = LMGen(
            lm,
            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
            sample_rate=self.mimi.sample_rate,
            device=device,
            frame_rate=self.mimi.frame_rate,
            save_voice_prompt_embeddings=False,
        )
 
        # StreamingSumInjector patches embed_codes on the LM model
        self.injector = StreamingSumInjector(lm)
 
        # ReferenceAdapter — weights loaded separately after LoRA merge
        self.adapter: Optional[ReferenceAdapter] = None
 
        # RAG token ID — set after vocabulary surgery during load
        self.rag_token_id: int = -1
 
        self.tts_mimi = loaders.get_mimi(mimi_weight, device)
        self.tts_mimi.streaming_forever(1)
 
        self.lock = asyncio.Lock()
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        # self.lm_gen.streaming_forever(1)
 
        self.vad_threshold = vad_threshold
        self.gpu_executor  = ThreadPoolExecutor(max_workers=1)
 
        # RAG + STT
        self.rag_enabled   = False
        self.rag_top_k     = rag_top_k
        self.rag_min_score = rag_min_score
        self.rag_chunks:    list = []
        self.rag_embeddings = None
        self.rag_embedding_model = None
        self.stt_lm_gen    = None
        self.stt_tokenizer = None
        self.stt_padding_token_id = 3
        self.compressor: Optional[ContextCompressor] = None
 
        if rag_index_dir is not None:
            self._init_rag(
                rag_index_dir, stt_hf_repo,
                compressor_model, compressor_device,
            )
 
    def _init_rag(
        self,
        rag_index_dir: str,
        stt_hf_repo: str,
        compressor_model: str,
        compressor_device: str,
    ) -> None:
        import moshi.models
 
        # Load vector index
        try:
            chunks, embeddings, emb_model_name = load_rag_index(rag_index_dir)
            self.rag_chunks     = chunks
            self.rag_embeddings = embeddings
            self.rag_embedding_model_name = emb_model_name
            logger.info(f"[RAG] index loaded: {len(self.rag_chunks)} chunks")
        except Exception as e:
            logger.error(f"[RAG] disabled — index load failed: {e}")
            return
 
        # Sentence-transformer encoder
        try:
            from sentence_transformers import SentenceTransformer
            self.rag_embedding_model = SentenceTransformer(
                self.rag_embedding_model_name, device="cpu"
            )
        except Exception as e:
            logger.error(f"[RAG] disabled — embedder failed: {e}")
            return
 
        # STT model
        try:
            stt_info = CheckpointInfo.from_hf_repo(stt_hf_repo)
            self.stt_mimi = stt_info.get_mimi(device=self.device)
            stt_lm = stt_info.get_moshi(device=self.device, dtype=torch.bfloat16)
            stt_lm.eval()
            self.stt_lm_gen = moshi.models.LMGen(stt_lm, temp=0, temp_text=0.0)
            self.stt_lm_gen.streaming_forever(1)
            self.stt_mimi.streaming_forever(1)
            self.stt_tokenizer = stt_info.get_text_tokenizer()
            self.stt_padding_token_id = stt_info.raw_config.get("text_padding_token_id", 3)
            logger.info("[RAG] STT loaded")
        except Exception as e:
            logger.error(f"[RAG] disabled — STT failed: {e}")
            return
 
        # 1B LLM compressor
        try:
            self.compressor = ContextCompressor(
                model_name=compressor_model,
                device="cuda",
                quantize_4bit = True,
            )
        except Exception as e:
            logger.error(f"[RAG] compressor failed (RAG still works without compression): {e}")
            self.compressor = None
 
        self.rag_enabled = True
        logger.info(f"[RAG] fully enabled, compressor={'yes' if self.compressor else 'no'}")
 
    def load_lora_and_adapter(
        self,
        checkpoint_dir: str,
        merge_lora: bool = True,
    ) -> None:
        """
        Load LoRA weights and ReferenceAdapter from a training checkpoint.
 
        Order of operations is critical:
          1. Read rag_token_id from checkpoint config (or infer it).
          2. Perform vocabulary surgery on the BASE model to match the
             checkpoint's weight shapes (text_linear, text_emb both grew
             by one row during training).  Loading LoRA onto an unexpanded
             model causes an immediate shape mismatch error.
          3. Load LoRA weights onto the now-correctly-shaped base model.
          4. Optionally merge LoRA deltas and discard peft wrapper.
          5. Load ReferenceAdapter state dict.
 
        checkpoint_dir should be e.g. 'checkpoints/rag_lora/step_2000'.
        It must contain:
            lora/          — peft LoRA adapter weights
            adapter.pt     — ReferenceAdapter state dict
        """
        ckpt = Path(checkpoint_dir)
        lm   = self.lm_gen.lm_model   # base LMModel, not yet wrapped by peft
 
        # ── Step 1: read rag_token_id from checkpoint ─────────────────────
        # Stored in lora/config.json if train_rag.py wrote it there,
        # otherwise we infer it after surgery below.
        rag_token_id = -1
        config_path  = ckpt / "lora" / "adapter_config.json"
        if not config_path.exists():
            config_path = ckpt / "lora" / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                if "rag_token_id" in cfg:
                    rag_token_id = int(cfg["rag_token_id"])
                    logger.info(f"[ckpt] rag_token_id={rag_token_id} (from config)")
            except Exception as e:
                logger.warning(f"[ckpt] could not read config: {e}")
 
        # ── Step 2: vocabulary surgery — expand base model to match ckpt ──
        # Training added one row to text_linear (output projection) and one
        # row to text_emb (input embedding).  We must do the same here
        # before loading any weights, or peft will raise a size mismatch.
        old_card = lm.text_card   # e.g. 32000
 
        # If rag_token_id not in config, it's old_card (the new last token)
        if rag_token_id < 0:
            rag_token_id = old_card
            logger.info(f"[ckpt] rag_token_id inferred as {rag_token_id}")
 
        new_card = rag_token_id + 1   # = old_card + 1
 
        if lm.text_linear.out_features == new_card:
            logger.info("[ckpt] base model already has correct vocab size — skipping surgery")
        else:
            logger.info(
                f"[ckpt] expanding vocab {old_card} → {new_card} "
                f"(rag_token_id={rag_token_id})"
            )
 
            # text_linear: [old_card, dim] → [new_card, dim]
            old_lin = lm.text_linear
            new_lin = nn.Linear(
                old_lin.in_features,
                new_card,
                bias=(old_lin.bias is not None),
                device=self.device,
                dtype=old_lin.weight.dtype,
            )
            with torch.no_grad():
                new_lin.weight[:old_card] = old_lin.weight
                # New row initialised to mean — same as training, so the
                # loaded LoRA delta will land on the right base value
                new_lin.weight[old_card]  = old_lin.weight.mean(dim=0)
                if old_lin.bias is not None:
                    new_lin.bias[:old_card] = old_lin.bias
                    new_lin.bias[old_card]  = old_lin.bias.mean()
            lm.text_linear = new_lin
 
            # text_emb (ScaledEmbedding): [old_card+1, dim] → [new_card+1, dim]
            # (+1 because ScaledEmbedding reserves one slot for zero_idx)
            old_emb = lm.text_emb
            new_emb = type(old_emb)(
                new_card + 1,
                old_emb.embedding_dim,
                norm=(old_emb.norm is not None),
                zero_idx=old_emb.zero_idx,
                device=self.device,
            ).to(dtype=old_emb.weight.dtype)
            with torch.no_grad():
                new_emb.weight[: old_card + 1] = old_emb.weight
                new_emb.weight[old_card + 1]   = old_emb.weight.mean(dim=0)
            lm.text_emb  = new_emb
            lm.text_card = new_card
            lm.text_emb  = lm.text_emb.to(dtype=old_emb.weight.dtype)
 
            logger.info("[ckpt] vocabulary surgery complete")
 
        self.rag_token_id = rag_token_id
 
        # ── Step 3: load LoRA weights onto the expanded base model ─────────
        lora_path = ckpt / "lora"
        if lora_path.exists():
            try:
                from peft import PeftModel
                logger.info(f"[ckpt] loading LoRA from {lora_path}")
                peft_model = PeftModel.from_pretrained(lm, str(lora_path))
 
                # ── Step 4: optionally merge LoRA deltas ──────────────────
                if merge_lora:
                    logger.info("[ckpt] merging LoRA into base weights")
                    merged = peft_model.merge_and_unload()
                    self.lm_gen.lm_model = merged
                    # Re-patch embed_codes on the now-merged model
                    self.injector.restore()
                    self.injector = StreamingSumInjector(merged)
                    logger.info("[ckpt] LoRA merged and unloaded — peft wrapper removed")
                else:
                    self.lm_gen.lm_model = peft_model
                    self.injector.restore()
                    self.injector = StreamingSumInjector(peft_model)
                    logger.info("[ckpt] LoRA loaded (not merged — peft wrapper kept)")
 
            except Exception as e:
                logger.error(f"[ckpt] LoRA load failed: {e}")
                import traceback; logger.error(traceback.format_exc())
        else:
            logger.warning(f"[ckpt] no lora/ directory at {lora_path} — skipping LoRA")
 
        # ── Step 5: load ReferenceAdapter ─────────────────────────────────
        adapter_path = ckpt / "adapter.pt"
        if adapter_path.exists():
            # Get the final LM (merged or peft-wrapped) for dim and text_emb
            final_lm = self.lm_gen.lm_model
            base     = getattr(final_lm, "base_model", final_lm)
            dim      = base.dim
 
            self.adapter = ReferenceAdapter(dim=dim, text_emb=base.text_emb)
            state_dict   = torch.load(adapter_path, map_location=self.device)
            self.adapter.load_state_dict(state_dict)
            self.adapter.to(self.device)
            adapter_dtype = next(self.adapter.parameters()).dtype
            lm_dtype = next(final_lm.parameters()).dtype
            if adapter_dtype != lm_dtype:
                logger.warning(f"[ckpt] adapter dtype {adapter_dtype} != lm dtype {lm_dtype}; forcing adapter to {lm_dtype}")
                self.adapter = self.adapter.to(dtype=lm_dtype)
            self.adapter.eval()
            logger.info(f"[ckpt] ReferenceAdapter loaded (dim={dim})")
        else:
            logger.warning(f"[ckpt] no adapter.pt at {adapter_path}")
 
    def warmup(self):
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            _ = self.other_mimi.encode(chunk)
            if self.rag_enabled and self.stt_lm_gen is not None:
                stt_codes = self.stt_mimi.encode(chunk)
                _ = self.stt_lm_gen.step_with_extra_heads(stt_codes)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:9])
                _ = self.other_mimi.decode(tokens[:, 1:9])
        if self.device.type == "cuda":
            torch.cuda.synchronize()
 
    # ------------------------------------------------------------------
    # Per-session RAG injection pipeline (called from handle_chat)
    # ------------------------------------------------------------------
 
    async def run_rag_injection(
        self,
        question: str,
        conversation_history: list[tuple[str, str]],
        clog: ColorizedLog,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """
        Background task triggered when <ref> token is detected.
 
        Pipeline:
          1. Retrieve relevant chunks from vector index (CPU, ~10ms)
          2. Compress with 1B LLM (CPU/secondary GPU, ~300-500ms)
          3. Encode with ReferenceAdapter (main GPU, ~1ms)
          4. Activate StreamingSumInjector
             → from this point forward, every lm_gen.step() adds the
               conditioning vector to embed_codes() output
 
        The model keeps generating filler speech throughout steps 1-3.
        Step 4 fires before the filler ends (the filler is ~1.5s, total
        pipeline is ~300-600ms), so the model transitions to grounded
        body speech without any gap.
        """
        t0 = time.monotonic()
 
        # Step 1: retrieve
        def _retrieve():
            return retrieve_chunks_hierarchical(
                question,
                self.rag_chunks,
                self.rag_embeddings,
                self.rag_embedding_model,
                self.rag_top_k,
                min_score=self.rag_min_score,
            )
        chunks = await loop.run_in_executor(None, _retrieve)
        t_retrieve = time.monotonic()
        clog.log("info",
            f"[RAG] retrieved {len(chunks)} chunks in "
            f"{(t_retrieve-t0)*1000:.0f}ms"
        )
 
        if not chunks:
            clog.log("info", "[RAG] no chunks above threshold — injector not activated")
            return
 
        # Step 2: compress with 1B LLM
        if self.compressor is not None:
            compressed = await self.compressor.compress_async(
                question, conversation_history, chunks, loop
            )
        else:
            # Fallback: concatenate chunk texts with truncation
            compressed = " ".join(c["text"][:200] for c in chunks[:2])
 
        t_compress = time.monotonic()
        clog.log("info",
            f"[RAG] compressed in {(t_compress-t_retrieve)*1000:.0f}ms: "
            f"{compressed[:100]!r}"
        )
 
        if not compressed:
            clog.log("info", "[RAG] compressor returned empty — no injection")
            return
 
        # Step 3: encode with ReferenceAdapter
        if self.adapter is None:
            clog.log("warning", "[RAG] no adapter loaded — cannot inject")
            return
 
        def _encode_ref():
            ids = self.text_tokenizer.encode(compressed)
            if not ids:
                return None
            id_tensor = torch.tensor(ids, dtype=torch.long, device=self.device)
            with torch.no_grad():
                vec = self.adapter(id_tensor)  # [dim]
            return vec
 
        vec = await loop.run_in_executor(self.gpu_executor, _encode_ref)
        t_encode = time.monotonic()
 
        if vec is None:
            return
 
        # Step 4: activate injector — takes effect on next lm_gen.step()
        await loop.run_in_executor(self.gpu_executor, self.injector.set, vec)
        clog.log("info",
            f"[RAG] injector activated in {(t_encode-t0)*1000:.0f}ms total "
            f"(retrieve={1000*(t_retrieve-t0):.0f}ms "
            f"compress={1000*(t_compress-t_retrieve):.0f}ms "
            f"encode={1000*(t_encode-t_compress):.0f}ms)"
        )
 
    # ------------------------------------------------------------------
    # WebSocket handler
    # ------------------------------------------------------------------
 
    async def handle_chat(self, request):
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(request)
        clog = ColorizedLog.randomize()
        peer = request.remote
        clog.log("info", f"connection from {peer}")
 
        # Voice prompt
        voice_prompt_path = None
        if self.voice_prompt_dir is not None:
            vpf = request.query["voice_prompt"]
            vpp = os.path.join(self.voice_prompt_dir, vpf)
            if not os.path.exists(vpp):
                raise FileNotFoundError(f"Voice prompt not found: {vpp}")
            voice_prompt_path = vpp
 
        if self.lm_gen.voice_prompt != voice_prompt_path:
            if voice_prompt_path and voice_prompt_path.endswith(".pt"):
                self.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
            elif voice_prompt_path:
                self.lm_gen.load_voice_prompt(voice_prompt_path)
 
        base_text_prompt = request.query.get("text_prompt", "")
        seed = int(request.query["seed"]) if "seed" in request.query else None
 
        # ── Per-session state ─────────────────────────────────────────────
 
        # Conversation history for the compressor: [(user_text, moshi_text), ...]
        conversation_history: list[tuple[str, str]] = []
        current_user_text:    str = ""
        current_moshi_parts:  list[str] = []
 
        # RAG phase tracking
        # States: idle → ref_detected → injecting → body
        # idle       : normal generation, injector off
        # ref_detected: <ref> token just seen, background task started
        # injecting  : background task running, model speaking filler
        # body       : injector active, model grounded on reference
        rag_phase = "idle"
        rag_task:  Optional[asyncio.Task] = None
 
        # STT state
        stt_token_buffer:        list = []
        stt_in_utterance:        bool = False
        stt_silence_frame_count: int  = 0
        stt_last_vad_end:        bool = False
        VAD_SILENCE_FRAMES = 12
 
        model_consecutive_pads = 0
        MODEL_IDLE_PAD_THRESHOLD = 25
 
        BARGE_IN_FRAMES = 4
        barge_in_frame_count = 0
        assistant_is_speaking = False
 
        close = False
        loop  = asyncio.get_event_loop()
 
        # ── Helpers ───────────────────────────────────────────────────────
 
        def _on_response_end():
            """Called when the model finishes a response (PAD threshold reached)."""
            nonlocal rag_phase, current_moshi_parts, current_user_text, assistant_is_speaking
            nonlocal stt_token_buffer, stt_in_utterance, stt_silence_frame_count, stt_last_vad_end
            moshi_text = "".join(current_moshi_parts).strip()
            if current_user_text and moshi_text:
                conversation_history.append((current_user_text, moshi_text))
                if len(conversation_history) > 10:
                    conversation_history.pop(0)
            current_moshi_parts = []
            current_user_text   = ""
            assistant_is_speaking = False
            
            stt_token_buffer = []
            stt_in_utterance = False
            stt_silence_frame_count = 0
            stt_last_vad_end = False
            loop.run_in_executor(self.gpu_executor, self.injector.clear)
            rag_phase = "idle"
 
        async def _trigger_rag_injection(question: str):
            """Fire the RAG pipeline as a background task."""
            nonlocal rag_phase, rag_task
            if rag_phase != "idle":
                return
            rag_phase = "injecting"
            clog.log("info", f"[RAG] <ref> detected — starting injection for: {question!r}")
            rag_task = asyncio.create_task(
                self.run_rag_injection(question, conversation_history, clog, loop)
            )
 
        # ── Recv loop ─────────────────────────────────────────────────────
 
        async def recv_loop():
            nonlocal close
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        break
                    if message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE):
                        break
                    if message.type != aiohttp.WSMsgType.BINARY:
                        continue
                    data = message.data
                    if not isinstance(data, bytes) or not data:
                        continue
                    if data[0] == 1:
                        opus_reader.append_bytes(data[1:])
            finally:
                close = True
 
        # ── Opus loop ─────────────────────────────────────────────────────
 
        async def opus_loop():
            nonlocal close, rag_phase, barge_in_frame_count
            nonlocal model_consecutive_pads
            nonlocal stt_token_buffer, stt_in_utterance
            nonlocal stt_silence_frame_count, stt_last_vad_end
            nonlocal current_user_text, current_moshi_parts
            nonlocal assistant_is_speaking
 
            all_pcm_data = None
            frame_budget = self.frame_size / self.mimi.sample_rate
 
            try:
                while True:
                    if close:
                        return
                    await asyncio.sleep(0.001)
 
                    pcm = opus_reader.read_pcm()
                    if pcm is None or pcm.shape[-1] == 0:
                        continue
                    all_pcm_data = pcm if all_pcm_data is None else np.concatenate((all_pcm_data, pcm))
 
                    while all_pcm_data.shape[-1] >= self.frame_size:
                        chunk_pcm    = all_pcm_data[: self.frame_size]
                        all_pcm_data = all_pcm_data[self.frame_size:]
                        chunk_tensor = (
                            torch.from_numpy(chunk_pcm)
                            .to(device=self.device)[None, None]
                        )
 
                        # ── GPU step ──────────────────────────────────────
                        def _step():
                            input_codes = self.mimi.encode(chunk_tensor)
                            _ = self.other_mimi.encode(chunk_tensor)
 
                            stt_res = None
                            if self.rag_enabled and self.stt_lm_gen is not None:
                                stt_codes = self.stt_mimi.encode(chunk_tensor)
                                stt_res   = self.stt_lm_gen.step_with_extra_heads(stt_codes)
 
                            # lm_gen.step → internally calls embed_codes (patched)
                            # StreamingSumInjector adds conditioning when active
                            tokens = self.lm_gen.step(input_codes)
                            main_pcm = None
                            if tokens is not None:
                                main_pcm = self.mimi.decode(tokens[:, 1:9])
                                _ = self.other_mimi.decode(tokens[:, 1:9])
                                main_pcm = main_pcm.cpu()
                            return tokens, main_pcm, input_codes, stt_res
 
                        t0 = time.monotonic()
                        tokens, main_pcm, input_codes, stt_result = (
                            await loop.run_in_executor(self.gpu_executor, _step)
                        )
                        dt = time.monotonic() - t0
                        if dt > frame_budget:
                            clog.log("warning", f"STEP OVERRUN: {dt*1000:.1f}ms")
 
                        # ── STT / VAD ─────────────────────────────────────
                        vad_score = 0.0
                        if stt_result is not None:
                            stt_tokens, vad_heads = stt_result
                            if vad_heads and len(vad_heads) > 2:
                                vad_score = float(vad_heads[2][0, 0, 0].cpu().item())
                            if stt_tokens is not None:
                                stt_token_buffer.append(stt_tokens[:, :1, :].cpu())
                                tt = stt_tokens[0, 0, 0].item()
                                if tt not in (0, 3):
                                    stt_in_utterance = True
                                    stt_last_vad_end = False
 
                            if vad_score > self.vad_threshold:
                                stt_silence_frame_count += 1
                            else:
                                stt_silence_frame_count = 0
 
                            vad_fired = (
                                stt_silence_frame_count >= VAD_SILENCE_FRAMES
                                and not stt_last_vad_end
                            )
                            stt_last_vad_end = stt_silence_frame_count >= VAD_SILENCE_FRAMES
 
                            if vad_fired and stt_in_utterance and stt_token_buffer:
                                transcript = decode_stt_tokens(
                                    stt_token_buffer,
                                    self.stt_tokenizer,
                                    self.stt_padding_token_id,
                                )
                                stt_in_utterance  = False
                                stt_silence_frame_count = 0
                                if transcript.strip():
                                    current_user_text = transcript.strip()
                                    clog.log("info", f"[STT] transcript so far: {current_user_text!r}")

                                    if self.rag_enabled and self.adapter is not None:
                                        await _trigger_rag_injection(current_user_text)
 
                        # ── Barge-in detection ────────────────────────────
                        speech_frame = False
                        if stt_result is not None and stt_result[0] is not None:
                            tt = stt_result[0][0, 0, 0].item()
                            if tt not in (0, 3) and vad_score < self.vad_threshold:
                                speech_frame = True
 
                        barge_in_frame_count = barge_in_frame_count + 1 if speech_frame else 0
 
                        if barge_in_frame_count >= BARGE_IN_FRAMES and assistant_is_speaking :
                            clog.log("info", "[barge-in] user interrupted")
                            _on_response_end()
                            barge_in_frame_count = 0
                            model_consecutive_pads = 0
                            # Cancel any in-flight RAG task
                            if rag_task and not rag_task.done():
                                rag_task.cancel()
                            opus_writer.append_pcm(
                                np.zeros(self.frame_size, dtype=np.float32)
                            )
                            continue
 
                        # ── Audio output ──────────────────────────────────
                        if main_pcm is not None:
                            opus_writer.append_pcm(main_pcm[0, 0].numpy())
                        else:
                            opus_writer.append_pcm(
                                np.zeros(self.frame_size, dtype=np.float32)
                            )
 
                        if tokens is None:
                            continue
 
                        # ── Text token handling ───────────────────────────
                        text_token = int(tokens[0, 0, 0].item())
 
                        # ── <ref> token detection ─────────────────────────
                        if (
                            self.rag_enabled
                            and self.adapter is not None
                            and text_token == self.rag_token_id
                            and rag_phase == "idle"
                            and current_user_text
                        ):
                            await _trigger_rag_injection(current_user_text)
                            print(text_token)
 
                        elif text_token not in (0, 3):
                            # Normal text token — collect for history
                            model_consecutive_pads = 0
                            assistant_is_speaking = True
                            _text = (
                                self.text_tokenizer.id_to_piece(text_token)
                                .replace("▁", " ")
                            )
                            current_moshi_parts.append(_text)
                            await ws.send_bytes(b"\x02" + _text.encode("utf-8"))
 
                        else:
                            # PAD token
                            model_consecutive_pads += 1
                            if model_consecutive_pads >= MODEL_IDLE_PAD_THRESHOLD:
                                _on_response_end()
                                model_consecutive_pads = 0
                                barge_in_frame_count   = 0
                                if self.stt_lm_gen is not None:
                                    self.stt_lm_gen.reset_streaming()
                                clog.log("info", "response finished — listening")
 
            except Exception:
                import traceback
                clog.log("error", "opus_loop crashed:\n" + traceback.format_exc())
                raise
 
        # ── Send loop ─────────────────────────────────────────────────────
 
        async def send_loop():
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if msg:
                    await ws.send_bytes(b"\x01" + msg)
 
        # ── Session setup ─────────────────────────────────────────────────
 
        clog.log("info", f"accepted — RAG={'on' if self.rag_enabled else 'off'}")
        if base_text_prompt:
            clog.log("info", f"text prompt: {base_text_prompt[:80]}")
 
        self.lm_gen.text_prompt_tokens = (
            self.text_tokenizer.encode(wrap_with_system_tags(base_text_prompt))
            if base_text_prompt else None
        )
 
        async with self.lock:
            if seed is not None and seed != -1:
                seed_all(seed)
 
            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
 
            self.mimi.reset_streaming()
            self.other_mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            await loop.run_in_executor(self.gpu_executor, self.injector.clear)    # ensure injector is off at session start
 
            if self.stt_lm_gen is not None:
                self.stt_lm_gen.reset_streaming()
 
            # System prompts — voice + silence + text + silence
            def _system_prompts():
                self.lm_gen.step_system_prompts(self.mimi)
            await loop.run_in_executor(self.gpu_executor, _system_prompts)
            self.mimi.reset_streaming()
 
            # Handshake
            if not close and not ws.closed:
                await ws.send_bytes(b"\x00")
                clog.log("info", "handshake sent")
 
                # Greeting
                GREETING_MAX   = int(8 * self.mimi.frame_rate)
                GREETING_PADS  = 15
                silence_chunk  = torch.zeros(
                    1, 1, self.frame_size, dtype=torch.float32, device=self.device
                )
                g_pads = 0
 
                for _ in range(GREETING_MAX):
                    if close or ws.closed:
                        break
 
                    def _greet():
                        inp = self.mimi.encode(silence_chunk)
                        _ = self.other_mimi.encode(silence_chunk)
                        tok = self.lm_gen.step(inp)
                        if tok is not None:
                            pcm = self.mimi.decode(tok[:, 1:9])
                            _ = self.other_mimi.decode(tok[:, 1:9])
                            return tok, pcm.cpu()
                        return None, None
 
                    g_tok, g_pcm = await loop.run_in_executor(self.gpu_executor, _greet)
                    if g_tok is None:
                        continue
                    opus_writer.append_pcm(g_pcm[0, 0].numpy())
                    gt = g_tok[0, 0, 0].item()
                    if gt not in (0, 3):
                        g_pads = 0
                        _t = self.text_tokenizer.id_to_piece(gt).replace("▁", " ")
                        await ws.send_bytes(b"\x02" + _t.encode("utf-8"))
                    else:
                        g_pads += 1
                        if g_pads >= GREETING_PADS:
                            break
 
                clog.log("info", "greeting done — listening")
 
                tasks = [
                    asyncio.create_task(recv_loop()),
                    asyncio.create_task(send_loop()),
                    asyncio.create_task(opus_loop()),
                ]
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for t in pending:
                    t.cancel()
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass
                if rag_task and not rag_task.done():
                    rag_task.cancel()
                # Clean up injector
                await loop.run_in_executor(self.gpu_executor, self.injector.clear)
                await ws.close()
 
        return ws


def _get_voice_prompt_dir(voice_prompt_dir: Optional[str], hf_repo: str) -> Optional[str]:
    if voice_prompt_dir is not None:
        return voice_prompt_dir
    logger.info("retrieving voice prompts")
    voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
    voices_tgz = Path(voices_tgz)
    voices_dir = voices_tgz.parent / "voices"
    if not voices_dir.exists():
        logger.info(f"extracting {voices_tgz} to {voices_dir}")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)
    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a 'voices/' directory")
    return str(voices_dir)


def _get_static_path(static: Optional[str]) -> Optional[str]:
    if static is None:
        logger.info("retrieving the static content")
        dist_tgz = hf_hub_download("nvidia/personaplex-7b-v1", "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        return str(dist)
    elif static != "none":
        return static
    return None
# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG-aware Moshi server (mid-stream injection)")
    parser.add_argument("--host",             default="localhost")
    parser.add_argument("--port",             type=int, default=8998)
    parser.add_argument("--static",           type=str)
    parser.add_argument("--tokenizer",        type=str)
    parser.add_argument("--moshi-weight",     type=str)
    parser.add_argument("--mimi-weight",      type=str)
    parser.add_argument("--hf-repo",          type=str, default=loaders.DEFAULT_REPO)
    parser.add_argument("--device",           type=str, default="cuda")
    parser.add_argument("--cpu-offload",      action="store_true")
    parser.add_argument("--voice-prompt-dir", type=str)
    parser.add_argument("--ssl",              type=str)

    # LoRA + adapter checkpoint
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Path to training checkpoint (e.g. checkpoints/rag_lora/step_2000). "
             "Must contain lora/ and adapter.pt. Omit to run base model."
    )
    parser.add_argument(
        "--no-merge-lora", action="store_true",
        help="Keep peft wrapper instead of merging LoRA into base weights."
    )

    # RAG
    parser.add_argument("--rag-index-dir",    type=str, default=None)
    parser.add_argument("--rag-top-k",        type=int,   default=2)
    parser.add_argument("--rag-min-score",    type=float, default=0.40)
    parser.add_argument("--stt-hf-repo",      type=str,
                        default="kyutai/stt-1b-en_fr-candle")
    parser.add_argument("--vad-threshold",    type=float, default=0.5)

    # 1B compressor
    parser.add_argument(
        "--compressor-model", type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HF model ID for the 1B context compressor LLM."
    )
    parser.add_argument(
        "--compressor-device", type=str, default="cpu",
        help="Device for the compressor LLM (default: cpu). "
             "Use 'cuda:1' if you have a second GPU."
    )

    parser.add_argument("--gradio-tunnel",       action="store_true")
    parser.add_argument("--gradio-tunnel-token", type=str)

    args = parser.parse_args()

    args.voice_prompt_dir = _get_voice_prompt_dir(args.voice_prompt_dir, args.hf_repo)
    static_path = _get_static_path(args.static)
    args.device = torch_auto_device(args.device)

    seed_all(42424242)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    hf_hub_download(args.hf_repo, "config.json")

    logger.info("loading mimi")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi       = loaders.get_mimi(args.mimi_weight, args.device)
    other_mimi = loaders.get_mimi(args.mimi_weight, args.device)

    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)

    logger.info("loading moshi")
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    torch.cuda.empty_cache()
    lm = loaders.get_moshi_lm(
        args.moshi_weight, device=args.device, cpu_offload=args.cpu_offload
    )
    lm.eval()

    state = ServerState(
        mimi=mimi,
        other_mimi=other_mimi,
        text_tokenizer=text_tokenizer,
        lm=lm,
        mimi_weight=args.mimi_weight,
        device=args.device,
        voice_prompt_dir=args.voice_prompt_dir,
        rag_index_dir=args.rag_index_dir,
        rag_top_k=args.rag_top_k,
        rag_min_score=args.rag_min_score,
        stt_hf_repo=args.stt_hf_repo,
        vad_threshold=args.vad_threshold,
        compressor_model=args.compressor_model,
        compressor_device=args.compressor_device,
    )

    # Load LoRA + adapter if checkpoint provided
    if args.checkpoint_dir:
        state.load_lora_and_adapter(
            args.checkpoint_dir,
            merge_lora=not args.no_merge_lora,
        )
    else:
        logger.info("no checkpoint — running base model (RAG injection disabled)")

    
    assert state.lm_gen._streaming_state is None, (
        "CUDA graph appears already captured before StreamingSumInjector "
        "was attached — injection would be silently ignored. Check that "
        "load_lora_and_adapter() (which re-patches the injector after any "
        "LoRA merge) runs before this warmup() call."
    )
    state.lm_gen.streaming_forever(1)
    logger.info("warming up")
    state.gpu_executor.submit(state.warmup).result()

    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)

    if static_path is not None:
        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))
        logger.info(f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static(
            "/", path=static_path, follow_symlinks=True, name="static"
        )

    ssl_context = None
    protocol    = "http"
    if args.ssl:
        ssl_context, protocol = create_ssl_context(args.ssl)

    host_ip = (
        args.host if args.host not in ("0.0.0.0", "::", "localhost")
        else __import__("moshi.utils.connection", fromlist=["get_lan_ip"]).get_lan_ip()
    )
    logger.info(f"serving at {protocol}://{host_ip}:{args.port}")

    if args.gradio_tunnel:
        try:
            from gradio import networking
            token = args.gradio_tunnel_token or secrets.token_urlsafe(32)
            tunnel = networking.setup_tunnel("localhost", args.port, token, None)
            logger.info(f"tunnel: {tunnel}")
        except ImportError:
            logger.error("pip install gradio for tunnel support")

    web.run_app(app, port=args.port, ssl_context=ssl_context)


with torch.no_grad():
    main()
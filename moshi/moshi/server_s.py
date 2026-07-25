# server.py — mid-stream <ref> injection, no reset between turns
#
# Design change from the reset-and-replay version:
#   - <ref>-wrapped compressed context is injected via forced step() calls,
#     identical convention to _step_text_prompt_core (silence on agent
#     audio codebooks, sine on user audio codebooks), but WITHOUT ever
#     calling reset_streaming(). The transformer's KV-cache — and with it
#     the entire conversation including the just-heard query audio — stays
#     intact across turns.
#   - No audio replay needed: the query audio was never discarded, so
#     there is nothing to rebuild.
#   - STT/VAD pipeline, barge-in detection, and the single-threaded
#     gpu_executor CUDA-graph discipline are unchanged.

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import random
import os
from pathlib import Path
import tarfile
import time
import secrets
import sys
from typing import Literal, Optional
import json

import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import sphn
import torch

from .client_utils import make_log, colorize
from .models import loaders, MimiModel, LMModel, LMGen
from .utils.connection import create_ssl_context, get_lan_ip
from .utils.logging import setup_logger, ColorizedLog
import moshi.models


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


def wrap_with_ref_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<ref>") and cleaned.endswith("<ref>"):
        return cleaned
    return f"<ref> {cleaned} <ref>"


# ── RAG helpers ──────────────────────────────────────────────────────────

def load_rag_index(index_dir: str):
    manifest_path = os.path.join(index_dir, "manifest.json")
    chunks_path = os.path.join(index_dir, "chunks.npz")
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
    transcript: str, chunks: list, embeddings: np.ndarray, embedding_model,
    top_k: int, min_score: float = 0.45,
) -> list[dict]:
    if not transcript.strip():
        return []
    query_vec = embedding_model.encode(transcript, normalize_embeddings=True).astype(np.float32)
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
    transcript: str, chunks: list, embeddings: np.ndarray, embedding_model,
    top_k: int, min_score: float = 0.45,
) -> list[dict]:
    base_hits = retrieve_chunks(transcript, chunks, embeddings, embedding_model, top_k, min_score)
    if not base_hits:
        return base_hits

    by_source: dict[str, list] = {}
    for i, c in enumerate(chunks):
        by_source.setdefault(c["source"], []).append((i, c))

    expanded: list[dict] = []
    seen_ids: set = set()
    for hit in base_hits:
        src = hit["source"]
        siblings = by_source[src]
        pos = next(j for j, (_, c) in enumerate(siblings) if c["id"] == hit["id"])
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
    transcript: str, retrieved_chunks: list[dict], embedding_model,
    max_sentences: int = 3, max_chars: int = 400,
) -> str:
    """Extractive fallback summarizer — used only if the ContextCompressor
    LLM isn't loaded or fails."""
    import re
    full_text = " ".join(c["text"] for c in retrieved_chunks)
    sentences = re.split(r'(?<=[.!?])\s+', full_text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    if not sentences:
        return full_text[:max_chars]

    query_vec = embedding_model.encode(transcript, normalize_embeddings=True).astype(np.float32)
    sent_vecs = embedding_model.encode(sentences, normalize_embeddings=True).astype(np.float32)
    scores = sent_vecs @ query_vec

    top_indices = sorted(np.argsort(scores)[::-1][:max_sentences])
    summary = " ".join(sentences[i] for i in top_indices)
    return summary[:max_chars]


# ── STT helpers (unchanged) ─────────────────────────────────────────────

def decode_stt_tokens(text_tokens: list[torch.Tensor], tokenizer, padding_token_id: int) -> str:
    if not text_tokens:
        return ""
    all_tokens = torch.cat(text_tokens, dim=-1)
    all_tokens = all_tokens.cpu().view(-1)
    valid = all_tokens[all_tokens > padding_token_id]
    if valid.numel() == 0:
        return ""
    return tokenizer.decode(valid.tolist())


# ── Context compressor (small LLM: query + top-k hits -> 1-2 sentence grounding) ──

class ContextCompressor:
    """Lightweight LLM (~1B params) that compresses retrieved chunks + recent
    text history into a short grounding statement for <ref> injection.
    Runs on its own dedicated thread — separate from gpu_executor — so it
    never contends with the CUDA-graph-captured step loop."""

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
        max_new_tokens: int = 60,
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
                model_name, quantization_config=bnb_config, device_map=device,
                token=os.getenv("HF_TOKEN"),
            )
        else:
            if quantize_4bit and device == "cpu":
                logger.warning(
                    "[compressor] quantize_4bit requested but device=cpu — "
                    "bitsandbytes 4-bit needs CUDA. Falling back to fp32 on CPU."
                )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map=device, token=os.getenv("HF_TOKEN"),
            )
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="compressor")
        logger.info(f"[compressor] ready — {sum(p.numel() for p in self.model.parameters())/1e9:.2f}B params")

    def compress(self, question: str, history: list[tuple[str, str]], chunks: list[dict]) -> str:
        if not chunks:
            return ""
        history_text = "\n".join(f"User: {u}\nMoshi: {m}" for u, m in history[-3:]) if history else "None"
        passages = "\n".join(f"[{i+1}] {c['text'][:300]}" for i, c in enumerate(chunks[:2]))
        prompt = self.COMPRESS_PROMPT.format(history=history_text, question=question, passages=passages)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0, prompt_len:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if "NO_CONTEXT" in result or not result:
            return ""
        return result

    async def compress_async(
        self, question: str, history: list[tuple[str, str]], chunks: list[dict],
        loop: asyncio.AbstractEventLoop,
    ) -> str:
        return await loop.run_in_executor(self._executor, self.compress, question, history, chunks)


# ── ServerState ──────────────────────────────────────────────────────────

@dataclass
class ServerState:
    mimi: MimiModel
    other_mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    lock: asyncio.Lock

    def __init__(
        self,
        mimi: MimiModel,
        other_mimi: MimiModel,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        lm: LMModel,
        mimi_weight,
        device: str | torch.device,
        voice_prompt_dir: str | None = None,
        save_voice_prompt_embeddings: bool = False,
        rag_index_dir: Optional[str] = None,
        rag_top_k: int = 2,
        rag_min_score: float = 0.40,
        stt_hf_repo: str = "kyutai/stt-1b-en_fr-candle",
        vad_threshold: float = 0.5,
        compressor_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        compressor_device: str = "cuda",
        compressor_4bit: bool = True,
        max_ref_tokens: int = 120,
    ):
        self.mimi = mimi
        self.other_mimi = other_mimi
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.voice_prompt_dir = voice_prompt_dir
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lm_gen = LMGen(
            lm,
            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
            sample_rate=self.mimi.sample_rate,
            device=device,
            frame_rate=self.mimi.frame_rate,
            save_voice_prompt_embeddings=save_voice_prompt_embeddings,
        )
        self.tts_mimi = loaders.get_mimi(mimi_weight, device)
        self.tts_mimi.streaming_forever(1)
        self.lock = asyncio.Lock()
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

        self.vad_threshold = vad_threshold

        # Dedicated single-worker executor for ALL lm_gen.step() /
        # mimi.encode/decode calls — keeps CUDA graph capture/replay on a
        # single consistent thread. Unchanged from the reset/replay version.
        self.gpu_executor = ThreadPoolExecutor(max_workers=1)

        # RAG + STT state
        self.rag_enabled = rag_index_dir is not None
        self.rag_top_k = rag_top_k
        self.rag_min_score = rag_min_score
        self.rag_chunks = []
        self.rag_embeddings = None
        self.rag_embedding_model = None
        self.stt_lm_gen = None
        self.stt_tokenizer = None
        self.stt_padding_token_id = 3
        self.context_compressor = None
        self.max_ref_tokens = max_ref_tokens

        # Silence framing around the injected <ref> block — mirrors the
        # pre/post-prompt silence convention from the old text-prompt
        # refresh, but now surrounds a mid-stream injection instead of a
        # post-reset one.
        self.pre_ref_silence_frames = max(2, int(0.1 * self.mimi.frame_rate))
        self.post_ref_silence_frames = max(2, int(0.15 * self.mimi.frame_rate))

        if self.rag_enabled:
            try:
                logger.info(f"loading RAG index from {rag_index_dir}...")
                chunks, embeddings, emb_model_name = load_rag_index(rag_index_dir)
                self.rag_chunks = chunks
                self.rag_embeddings = embeddings
                self.rag_embedding_model_name = emb_model_name
                logger.info(f"RAG index loaded: {len(self.rag_chunks)} chunks")
            except Exception as e:
                logger.error(f"RAG disabled — index load failed: {e}")
                self.rag_enabled = False

        if self.rag_enabled:
            try:
                from sentence_transformers import SentenceTransformer
                self.rag_embedding_model = SentenceTransformer(self.rag_embedding_model_name, device="cpu")
                logger.info(f"embedding model loaded on CPU: {self.rag_embedding_model_name}")
            except ImportError:
                logger.error("RAG disabled — pip install sentence-transformers")
                self.rag_enabled = False
            except Exception as e:
                logger.error(f"RAG disabled — embedding model failed: {e}")
                self.rag_enabled = False

        if self.rag_enabled:
            try:
                logger.info(f"loading STT model from {stt_hf_repo}...")
                stt_info = moshi.models.loaders.CheckpointInfo.from_hf_repo(stt_hf_repo)
                self.stt_mimi = stt_info.get_mimi(device=device)
                stt_lm = stt_info.get_moshi(device=device, dtype=torch.bfloat16)
                stt_lm.eval()
                self.stt_lm_gen = moshi.models.LMGen(stt_lm, temp=0, temp_text=0.0)
                self.stt_lm_gen.streaming_forever(1)
                self.stt_mimi.streaming_forever(1)
                self.stt_tokenizer = stt_info.get_text_tokenizer()
                self.stt_padding_token_id = stt_info.raw_config.get("text_padding_token_id", 3)
                logger.info(
                    f"STT model loaded: "
                    f"params={sum(p.numel() for p in stt_lm.parameters()) / 1e9:.1f}B "
                    f"padding_id={self.stt_padding_token_id}"
                )
            except Exception as e:
                logger.error(f"RAG disabled — STT model load failed: {e}")
                self.rag_enabled = False

        if self.rag_enabled:
            try:
                self.context_compressor = ContextCompressor(
                    model_name=compressor_model_name,
                    device=compressor_device,
                    quantize_4bit=compressor_4bit,
                )
            except Exception as e:
                logger.error(f"context compressor disabled — load failed: {e} "
                              f"(falling back to extractive summarize_context)")
                self.context_compressor = None
    def load_lora_and_adapter(
        self,
        checkpoint_dir: str,
        merge_lora: bool = True,
    ) -> None:
        """
        Load LoRA weights from a training checkpoint.
    
        This version assumes the base model has NOT been expanded with a special
        token; LoRA was applied directly to the original LMModel.
    
        checkpoint_dir should contain:
            lora/          — peft LoRA adapter weights (adapter_config.json, adapter_model.*)
            adapter.pt     — (optional) ReferenceAdapter, but ignored in this pipeline.
    
        Args:
            checkpoint_dir: Path to the checkpoint directory.
            merge_lora: If True, merge LoRA deltas into the base model and discard the wrapper.
        """
        from pathlib import Path
        from peft import PeftModel
        import logging
    
        logger = logging.getLogger(__name__)
        ckpt = Path(checkpoint_dir)
        lm = self.lm_gen.lm_model   # base LMModel, not yet wrapped by peft
    
        # ── Load LoRA weights ────────────────────────────────────────────────
        lora_path = ckpt / "lora"
        if not lora_path.exists():
            logger.warning(f"No lora/ directory found at {lora_path} — skipping LoRA")
            return
    
        try:
            logger.info(f"Loading LoRA from {lora_path}")
            peft_model = PeftModel.from_pretrained(lm, str(lora_path))
    
            if merge_lora:
                logger.info("Merging LoRA into base weights")
                merged = peft_model.merge_and_unload()
                self.lm_gen.lm_model = merged
                # Update injector if it exists (e.g., for streaming generation)
                if hasattr(self, 'injector') and self.injector is not None:
                    self.injector.restore()
                    self.injector = StreamingSumInjector(merged)
                logger.info("LoRA merged and unloaded — peft wrapper removed")
            else:
                self.lm_gen.lm_model = peft_model
                if hasattr(self, 'injector') and self.injector is not None:
                    self.injector.restore()
                    self.injector = StreamingSumInjector(peft_model)
                logger.info("LoRA loaded (not merged — peft wrapper kept)")
    
        except Exception as e:
            logger.error(f"LoRA load failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
        # ── Optional ReferenceAdapter (ignored in this pipeline) ──────────
        adapter_path = ckpt / "adapter.pt"
        if adapter_path.exists():
            logger.info(
                f"Found adapter.pt at {adapter_path}, but this training pipeline "
                "does not use ReferenceAdapter. Skipping."
            )
            # If you ever need to load it, you can add logic here,
            # but it requires the corresponding `self.adapter` attribute.
        else:
            logger.debug("No adapter.pt found (expected for this pipeline).")

    def warmup(self):
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            _ = self.other_mimi.encode(chunk)

            stt_codes = self.stt_mimi.encode(chunk)
            _ = self.stt_lm_gen.step_with_extra_heads(stt_codes)

            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:9])
                _ = self.other_mimi.decode(tokens[:, 1:9])

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

    async def handle_chat(self, request):
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(request)
        clog = ColorizedLog.randomize()
        peer = request.remote
        peer_port = request.transport.get_extra_info("peername")[1]
        clog.log("info", f"Incoming connection from {peer}:{peer_port}")
        clog.log("info",
            f"RAG enabled={self.rag_enabled} chunks={len(self.rag_chunks)} "
            f"vad_threshold={self.vad_threshold}"
        )

        requested_voice_prompt_path = None
        voice_prompt_path = None
        if self.voice_prompt_dir is not None:
            voice_prompt_filename = request.query["voice_prompt"]
            if voice_prompt_filename:
                requested_voice_prompt_path = os.path.join(self.voice_prompt_dir, voice_prompt_filename)
            if (requested_voice_prompt_path is None
                    or not os.path.exists(requested_voice_prompt_path)):
                raise FileNotFoundError(
                    f"Voice prompt '{voice_prompt_filename}' not found in '{self.voice_prompt_dir}'"
                )
            voice_prompt_path = requested_voice_prompt_path

        if self.lm_gen.voice_prompt != voice_prompt_path:
            if voice_prompt_path.endswith('.pt'):
                self.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
            else:
                self.lm_gen.load_voice_prompt(voice_prompt_path)

        base_text_prompt = request.query.get("text_prompt", "")
        seed = int(request.query["seed"]) if "seed" in request.query else None

        # ── Per-session state ────────────────────────────────────────────
        listening = True
        generating = False
        injecting = False   # renamed from `resetting` — nothing is actually reset now
        model_consecutive_pads = 0
        MODEL_IDLE_PAD_THRESHOLD = 25

        stt_token_buffer = []
        stt_in_utterance = False
        stt_silence_frame_count = 0
        stt_last_vad_end = False
        VAD_SILENCE_FRAMES_REQUIRED = 12

        BARGE_IN_FRAMES_REQUIRED = 4
        barge_in_frame_count = 0

        # Conversation history for the compressor + response-text accumulation
        session_history: list[tuple[str, str]] = []
        current_transcript = ""
        response_text_buffer = ""

        gpu_executor = self.gpu_executor

        async def send_context_window(retrieved: list[dict], transcript: str):
            clog.log("info", "─" * 60)
            clog.log("info", "RAG CONTEXT WINDOW")
            clog.log("info", f"transcript: \"{transcript}\"")
            for i, c in enumerate(retrieved):
                clog.log("info", f"  [{i+1}] score={c['similarity_score']:.3f} source={c['source']}")
                clog.log("info", f"       \"{c['text'][:100]}\"")
            clog.log("info", "─" * 60)
            payload = {
                "transcript": transcript,
                "chunks": [
                    {"id": c["id"], "text": c["text"], "source": c["source"],
                     "score": round(c["similarity_score"], 3)}
                    for c in retrieved
                ],
            }
            msg = b"\x03" + json.dumps(payload).encode("utf-8")
            try:
                await ws.send_bytes(msg)
            except Exception:
                pass

        async def start_rag_turn(transcript: str):
            """Mid-stream injection — NO reset_streaming() anywhere in this
            function. The transformer's KV-cache (and therefore the whole
            conversation, including the just-heard query audio) stays
            intact throughout. Matches training: <ref>-wrapped content is
            forced into the text stream via per-frame step() calls, with
            agent-audio-silence / user-audio-sine, exactly like
            _step_text_prompt_core / _build_ref_block_frames."""
            nonlocal listening, generating, injecting, current_transcript

            if not listening:
                return

            clog.log("info", f"VAD end – injecting mid-stream: \"{transcript}\"")
            listening = False
            generating = False
            injecting = True

            loop = asyncio.get_event_loop()

            # 1. Retrieve top-k chunks
            def _retrieve():
                return retrieve_chunks_hierarchical(
                    transcript, self.rag_chunks, self.rag_embeddings,
                    self.rag_embedding_model, self.rag_top_k, min_score=self.rag_min_score,
                )
            hits = await loop.run_in_executor(None, _retrieve)
            await send_context_window(hits, transcript)

            # 2. Compress with the small LLM (question + recent text
            #    history + hits). No audio history needed — the model's
            #    own KV-cache already carries the audio.
            if hits and self.context_compressor is not None:
                grounding = await self.context_compressor.compress_async(
                    question=transcript, history=session_history[-3:], chunks=hits, loop=loop,
                )
            elif hits:
                grounding = await loop.run_in_executor(
                    None,
                    lambda: summarize_context(
                        transcript, hits, self.rag_embedding_model,
                        max_sentences=2, max_chars=300,
                    )
                )
            else:
                grounding = ""

            if grounding:
                ref_content = f"Use ONLY this information to answer: {grounding}"
            else:
                ref_content = (
                    "No relevant information was found. Say you don't have "
                    "information about that, and nothing else."
                )
            clog.log("info", f"Injecting <ref> block: {ref_content[:200]}")

            # 3. Tokenize + cap length — bounds injection latency regardless
            #    of compressor verbosity. wrap_with_ref_tags + ordinary
            #    tokenizer.encode — SAME construction as
            #    InterleavedTokenizer._build_ref_block_frames on the
            #    training side (cap-then-wrap, so the closing tag is never
            #    truncated away).
            content = grounding if grounding else ref_content
            # (ref_content above already includes the wrapping instruction
            #  text; keep the whole thing as the thing we wrap+cap)
            source_text = ref_content
            ids = self.text_tokenizer.encode(source_text.strip())
            if len(ids) > self.max_ref_tokens:
                ids = ids[: self.max_ref_tokens]
                source_text = self.text_tokenizer.decode(ids)
            ref_tokens = self.text_tokenizer.encode(wrap_with_ref_tags(source_text))

            # 4. Inject via forced text_token, one frame at a time —
            #    IDENTICAL convention to _step_text_prompt_core /
            #    _build_ref_block_frames (silence on agent audio codebooks,
            #    sine on user audio codebooks). NO reset_streaming().
            def _inject_ref_block_sync():
                for _ in range(self.pre_ref_silence_frames):
                    self.lm_gen.step(
                        moshi_tokens=self.lm_gen._encode_zero_frame(),
                        text_token=self.lm_gen.zero_text_code,
                        input_tokens=self.lm_gen._encode_sine_frame(),
                    )
                for tok in ref_tokens:
                    self.lm_gen.step(
                        moshi_tokens=self.lm_gen._encode_zero_frame(),
                        text_token=tok,
                        input_tokens=self.lm_gen._encode_sine_frame(),
                    )
                for _ in range(self.post_ref_silence_frames):
                    self.lm_gen.step(
                        moshi_tokens=self.lm_gen._encode_zero_frame(),
                        text_token=self.lm_gen.zero_text_code,
                        input_tokens=self.lm_gen._encode_sine_frame(),
                    )

            t0 = time.monotonic()
            await loop.run_in_executor(gpu_executor, _inject_ref_block_sync)
            clog.log("info", f"TIMING ref_injection={time.monotonic() - t0:.2f}s ({len(ref_tokens)} tokens)")

            # 5. Keep the client's audio stream alive during injection —
            #    exactly as many silent frames as were consumed above. No
            #    audio replay: nothing was discarded, there's nothing to
            #    rebuild.
            n_inject_frames = (
                self.pre_ref_silence_frames + len(ref_tokens) + self.post_ref_silence_frames
            )
            silent_frame = np.zeros(self.frame_size, dtype=np.float32)
            for _ in range(n_inject_frames):
                opus_writer.append_pcm(silent_frame)

            generating = True
            injecting = False
            current_transcript = transcript
            clog.log("info", "ref injected – now generating response")

        async def recv_loop():
            nonlocal close
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        clog.log("error", f"{ws.exception()}")
                        break
                    elif message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE):
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        continue
                    message = message.data
                    if not isinstance(message, bytes) or len(message) == 0:
                        continue
                    kind = message[0]
                    if kind == 1:
                        opus_reader.append_bytes(message[1:])
                    else:
                        clog.log("warning", f"unknown message kind {kind}")
            finally:
                close = True
                clog.log("info", "connection closed")

        async def opus_loop():
            nonlocal listening, generating, injecting
            nonlocal stt_token_buffer, stt_in_utterance, stt_silence_frame_count, stt_last_vad_end
            nonlocal model_consecutive_pads, barge_in_frame_count
            nonlocal response_text_buffer, current_transcript

            all_pcm_data = None
            loop = asyncio.get_event_loop()
            frame_budget_s = self.frame_size / self.mimi.sample_rate

            try:
                while True:
                    if close:
                        return
                    await asyncio.sleep(0.001)

                    # Drain mic input during injection so no backlog builds
                    # up and bursts once generation resumes — same drain
                    # behavior as before, just renamed from "resetting".
                    if injecting:
                        pcm = opus_reader.read_pcm()
                        silent_pcm = np.zeros(self.frame_size, dtype=np.float32)
                        opus_writer.append_pcm(silent_pcm)
                        continue

                    pcm = opus_reader.read_pcm()
                    if pcm is None or pcm.shape[-1] == 0:
                        continue
                    if all_pcm_data is None:
                        all_pcm_data = pcm
                    else:
                        all_pcm_data = np.concatenate((all_pcm_data, pcm))

                    while all_pcm_data.shape[-1] >= self.frame_size:
                        chunk_pcm = all_pcm_data[:self.frame_size]
                        all_pcm_data = all_pcm_data[self.frame_size:]
                        chunk_tensor = torch.from_numpy(chunk_pcm).to(device=self.device)[None, None]

                        if listening:
                            # STT/VAD pipeline — UNCHANGED from the
                            # reset/replay version.
                            def _listen_step():
                                codes = self.mimi.encode(chunk_tensor)
                                _ = self.other_mimi.encode(chunk_tensor)
                                stt_res = None
                                if self.rag_enabled and self.stt_lm_gen is not None:
                                    stt_codes = self.stt_mimi.encode(chunk_tensor)
                                    stt_res = self.stt_lm_gen.step_with_extra_heads(stt_codes)
                                return codes, stt_res

                            t_step0 = time.monotonic()
                            input_codes, stt_result = await loop.run_in_executor(gpu_executor, _listen_step)
                            t_step_dt = time.monotonic() - t_step0
                            if t_step_dt > frame_budget_s:
                                clog.log("warning",
                                    f"STEP OVERRUN (listen): {t_step_dt*1000:.1f}ms "
                                    f"(budget {frame_budget_s*1000:.1f}ms)")

                            # NOTE: no more session_user_audio_frames
                            # buffering — nothing replays it, since the
                            # model's own KV-cache already has this audio
                            # from the live step() call happening as part
                            # of normal generation once the turn resumes.
                            # (During `listening`, mimi.encode() runs here
                            # purely for STT/VAD purposes — the LM itself
                            # is not stepped while listening, matching the
                            # original design; the LM only sees this
                            # audio once generation begins after injection.)

                            if stt_result is not None:
                                stt_tokens, vad_heads = stt_result
                                vad_score = 0.0
                                if vad_heads and len(vad_heads) > 2:
                                    vad_score = float(vad_heads[2][0, 0, 0].cpu().item())
                                if stt_tokens is not None:
                                    stt_token_buffer.append(stt_tokens[:, :1, :].cpu())
                                    text_token = stt_tokens[0, 0, 0].item()
                                    if text_token not in (0, 3):
                                        stt_in_utterance = True
                                        stt_last_vad_end = False
                                if vad_score > self.vad_threshold:
                                    stt_silence_frame_count += 1
                                else:
                                    stt_silence_frame_count = 0
                                vad_fired = (
                                    stt_silence_frame_count >= VAD_SILENCE_FRAMES_REQUIRED
                                    and not stt_last_vad_end
                                )
                                stt_last_vad_end = stt_silence_frame_count >= VAD_SILENCE_FRAMES_REQUIRED
                                if vad_fired and stt_in_utterance and stt_token_buffer:
                                    transcript = decode_stt_tokens(
                                        stt_token_buffer, self.stt_tokenizer, self.stt_padding_token_id
                                    )
                                    stt_token_buffer = []
                                    stt_in_utterance = False
                                    stt_silence_frame_count = 0
                                    if transcript.strip():
                                        if self.stt_lm_gen is not None:
                                            self.stt_lm_gen.reset_streaming()
                                            self.stt_mimi.reset_streaming()
                                        asyncio.create_task(start_rag_turn(transcript))

                            silent_pcm = np.zeros(self.frame_size, dtype=np.float32)
                            opus_writer.append_pcm(silent_pcm)

                        elif generating:
                            def _step_and_decode():
                                input_codes = self.mimi.encode(chunk_tensor)
                                _ = self.other_mimi.encode(chunk_tensor)

                                stt_res = None
                                if self.rag_enabled and self.stt_lm_gen is not None:
                                    stt_codes = self.stt_mimi.encode(chunk_tensor)
                                    stt_res = self.stt_lm_gen.step_with_extra_heads(stt_codes)

                                tokens = self.lm_gen.step(input_codes)
                                if tokens is not None:
                                    main_pcm = self.mimi.decode(tokens[:, 1:9])
                                    _ = self.other_mimi.decode(tokens[:, 1:9])
                                    return tokens, main_pcm.cpu(), input_codes, stt_res
                                return None, None, input_codes, stt_res

                            t_step0 = time.monotonic()
                            tokens, main_pcm, input_codes, stt_result = await loop.run_in_executor(
                                gpu_executor, _step_and_decode
                            )
                            t_step_dt = time.monotonic() - t_step0
                            if t_step_dt > frame_budget_s:
                                clog.log("warning",
                                    f"STEP OVERRUN (generate): {t_step_dt*1000:.1f}ms "
                                    f"(budget {frame_budget_s*1000:.1f}ms)")

                            # Barge-in detection — UNCHANGED
                            speech_this_frame = False
                            if stt_result is not None:
                                stt_tokens, _vad_heads = stt_result
                                vad_score_bi = 0.0
                                if _vad_heads and len(_vad_heads) > 2:
                                    vad_score_bi = float(_vad_heads[2][0, 0, 0].cpu().item())
                                if stt_tokens is not None:
                                    stt_text_token = stt_tokens[0, 0, 0].item()
                                    if (stt_text_token not in (0, 3)
                                            and vad_score_bi < self.vad_threshold):
                                        speech_this_frame = True

                            if speech_this_frame:
                                barge_in_frame_count += 1
                            else:
                                barge_in_frame_count = 0

                            barge_in = barge_in_frame_count >= BARGE_IN_FRAMES_REQUIRED

                            if barge_in:
                                clog.log("info",
                                    f"Barge-in confirmed ({BARGE_IN_FRAMES_REQUIRED} "
                                    f"consecutive speech frames) — interrupting response")
                                barge_in_frame_count = 0
                                generating = False
                                listening = True
                                model_consecutive_pads = 0
                                stt_token_buffer = []
                                stt_in_utterance = True
                                stt_last_vad_end = False
                                stt_silence_frame_count = 0
                                response_text_buffer = ""

                                # Barge-in IS a legitimate discontinuity —
                                # the in-flight response is genuinely
                                # aborted, so resetting here (unlike the
                                # normal turn-transition path) is correct.
                                self.lm_gen.reset_streaming()
                                self.mimi.reset_streaming()
                                self.other_mimi.reset_streaming()
                                if self.stt_lm_gen is not None:
                                    self.stt_lm_gen.reset_streaming()
                                    self.stt_mimi.reset_streaming()
                                if self.device.type == 'cuda':
                                    torch.cuda.empty_cache()

                                silent_pcm = np.zeros(self.frame_size, dtype=np.float32)
                                opus_writer.append_pcm(silent_pcm)
                                continue

                            if tokens is not None:
                                opus_writer.append_pcm(main_pcm[0, 0].numpy())
                                text_token = tokens[0, 0, 0].item()
                                if text_token not in (0, 3):
                                    model_consecutive_pads = 0
                                    _text = self.text_tokenizer.id_to_piece(text_token)
                                    _text = _text.replace("▁", " ")
                                    response_text_buffer += _text
                                    msg = b"\x02" + bytes(_text, encoding="utf8")
                                    await ws.send_bytes(msg)
                                else:
                                    model_consecutive_pads += 1
                                    if model_consecutive_pads >= MODEL_IDLE_PAD_THRESHOLD:
                                        generating = False
                                        listening = True
                                        model_consecutive_pads = 0
                                        barge_in_frame_count = 0
                                        stt_token_buffer = []
                                        stt_in_utterance = False
                                        stt_silence_frame_count = 0
                                        if response_text_buffer.strip():
                                            session_history.append(
                                                (current_transcript, response_text_buffer.strip())
                                            )
                                            if len(session_history) > 6:
                                                session_history[:] = session_history[-6:]
                                        response_text_buffer = ""
                                        # STT model's own streaming state —
                                        # a separate model instance, safe
                                        # and correct to reset per-turn.
                                        if self.stt_lm_gen is not None:
                                            self.stt_lm_gen.reset_streaming()
                                            self.stt_mimi.reset_streaming()
                                        clog.log("info", "Response finished – back to listening")
            except Exception:
                import traceback
                clog.log("error", "opus_loop crashed")
                clog.log("error", traceback.format_exc())
                raise

        async def send_loop():
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await ws.send_bytes(b"\x01" + msg)

        clog.log("info", "accepted connection")
        if base_text_prompt:
            clog.log("info", f"text prompt: {base_text_prompt}")
        if voice_prompt_path:
            clog.log("info", f"voice prompt: {voice_prompt_path}")

        self.lm_gen.text_prompt_tokens = (
            self.text_tokenizer.encode(wrap_with_system_tags(base_text_prompt))
            if base_text_prompt else None
        )

        close = False
        async with self.lock:
            if seed is not None and seed != -1:
                seed_all(seed)

            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)

            # Session-start resets are the ONLY legitimate reset_streaming()
            # calls left in the normal flow (plus barge-in, above).
            self.mimi.reset_streaming()
            self.other_mimi.reset_streaming()
            self.lm_gen.reset_streaming()

            if self.stt_lm_gen is not None:
                self.stt_lm_gen.reset_streaming()
                self.stt_mimi.reset_streaming()

            async def is_alive():
                if close or ws.closed:
                    return False
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=0.01)
                    if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        return False
                except asyncio.TimeoutError:
                    return True
                except aiohttp.ClientConnectionError:
                    return False
                return True

            loop = asyncio.get_event_loop()

            async def _run_initial_system_prompts():
                def _sync_system_prompts():
                    self.lm_gen.step_system_prompts(self.mimi)
                await loop.run_in_executor(self.gpu_executor, _sync_system_prompts)

            await _run_initial_system_prompts()
            self.mimi.reset_streaming()
            clog.log("info", "done with system prompts")

            if await is_alive():
                await ws.send_bytes(b"\x00")
                clog.log("info", "sent handshake bytes")

                recv_task = asyncio.create_task(recv_loop())
                send_task = asyncio.create_task(send_loop())

                GREETING_MAX_FRAMES = int(8 * self.mimi.frame_rate)
                GREETING_PAD_THRESHOLD = 15

                silence_chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
                greeting_pads = 0
                clog.log("info", "generating greeting...")

                for _ in range(GREETING_MAX_FRAMES):
                    if close or ws.closed:
                        break

                    def _greeting_step():
                        inp = self.mimi.encode(silence_chunk)
                        _ = self.other_mimi.encode(silence_chunk)
                        tok = self.lm_gen.step(inp)
                        if tok is not None:
                            pcm = self.mimi.decode(tok[:, 1:9])
                            _ = self.other_mimi.decode(tok[:, 1:9])
                            return tok, pcm.cpu()
                        return None, None

                    g_tokens, g_pcm = await loop.run_in_executor(self.gpu_executor, _greeting_step)
                    if g_tokens is None:
                        continue

                    opus_writer.append_pcm(g_pcm[0, 0].numpy())
                    g_text_token = g_tokens[0, 0, 0].item()
                    if g_text_token not in (0, 3):
                        greeting_pads = 0
                        _t = self.text_tokenizer.id_to_piece(g_text_token)
                        _t = _t.replace("▁", " ")
                        await ws.send_bytes(b"\x02" + bytes(_t, encoding="utf8"))
                    else:
                        greeting_pads += 1
                        if greeting_pads >= GREETING_PAD_THRESHOLD:
                            break

                clog.log("info", "greeting complete — entering listening mode")

                opus_task = asyncio.create_task(opus_loop())
                tasks = [recv_task, send_task, opus_task]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                await ws.close()
                clog.log("info", "session closed")

        clog.log("info", "done with connection")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--gradio-tunnel", action='store_true')
    parser.add_argument("--gradio-tunnel-token", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--moshi-weight", type=str)
    parser.add_argument("--mimi-weight", type=str)
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--voice-prompt-dir", type=str)
    parser.add_argument("--ssl", type=str)
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Path to training checkpoint (e.g. checkpoints/rag_lora/step_2000). "
             "Must contain lora/ and adapter.pt. Omit to run base model."
    )
    parser.add_argument(
        "--no-merge-lora", action="store_true",
        help="Keep peft wrapper instead of merging LoRA into base weights."
    )
    parser.add_argument("--rag-index-dir", type=str, default=None,
        help="Path to RAG index (build_index.py --mode text). Omit to disable RAG.")
    parser.add_argument("--rag-top-k", type=int, default=2)
    parser.add_argument("--rag-min-score", type=float, default=0.25)
    parser.add_argument("--stt-hf-repo", type=str, default="kyutai/stt-1b-en_fr-candle")
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--compressor-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--compressor-device", type=str, default="cuda",
        help="Device for the context-compressor LLM. Use a CUDA device separate "
             "from the main model for real throughput with 4-bit quantization.")
    parser.add_argument("--compressor-4bit", action="store_true", default=True)
    parser.add_argument("--max-ref-tokens", type=int, default=500,
        help="Cap on injected <ref> block length, in tokens. Must match the "
             "training-time max_ref_tokens for train/inference consistency.")

    args = parser.parse_args()
    args.voice_prompt_dir = _get_voice_prompt_dir(args.voice_prompt_dir, args.hf_repo)
    if args.voice_prompt_dir is not None:
        assert os.path.exists(args.voice_prompt_dir), f"Directory missing: {args.voice_prompt_dir}"
    logger.info(f"voice_prompt_dir = {args.voice_prompt_dir}")

    static_path = _get_static_path(args.static)
    assert static_path is None or os.path.exists(static_path), f"Static path does not exist: {static_path}."
    logger.info(f"static_path = {static_path}")
    args.device = torch_auto_device(args.device)

    seed_all(42424242)

    setup_tunnel = None
    tunnel_token = ''
    if args.gradio_tunnel:
        try:
            from gradio import networking
        except ImportError:
            logger.error("Cannot find gradio. pip install gradio")
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        tunnel_token = args.gradio_tunnel_token if args.gradio_tunnel_token else secrets.token_urlsafe(32)

    hf_hub_download(args.hf_repo, "config.json")

    logger.info("loading mimi")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, args.device)
    other_mimi = loaders.get_mimi(args.mimi_weight, args.device)
    logger.info("mimi loaded")

    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)

    logger.info("loading moshi")
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)

    torch.cuda.empty_cache()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    lm = loaders.get_moshi_lm(args.moshi_weight, device=args.device, cpu_offload=args.cpu_offload)
    lm.eval()
    logger.info("moshi loaded")

    state = ServerState(
        mimi=mimi,
        other_mimi=other_mimi,
        text_tokenizer=text_tokenizer,
        lm=lm,
        mimi_weight=args.mimi_weight,
        device=args.device,
        voice_prompt_dir=args.voice_prompt_dir,
        save_voice_prompt_embeddings=False,
        rag_index_dir=args.rag_index_dir,
        rag_top_k=args.rag_top_k,
        rag_min_score=args.rag_min_score,
        stt_hf_repo=args.stt_hf_repo,
        vad_threshold=args.vad_threshold,
        compressor_model_name=args.compressor_model,
        compressor_device=args.compressor_device,
        compressor_4bit=args.compressor_4bit,
        max_ref_tokens=args.max_ref_tokens,
    )

    if args.checkpoint_dir:
        state.load_lora_and_adapter(
            args.checkpoint_dir,
            merge_lora=not args.no_merge_lora,
        )
    else:
        logger.info("no checkpoint — running base model (RAG injection disabled)")
        
    logger.info("warming up the model")
    state.gpu_executor.submit(state.warmup).result()

    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)

    if static_path is not None:
        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))
        logger.info(f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static("/", path=static_path, follow_symlinks=True, name="static")

    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        ssl_context, protocol = create_ssl_context(args.ssl)

    host_ip = args.host if args.host not in ("0.0.0.0", "::", "localhost") else get_lan_ip()
    logger.info(f"Access the Web UI at {protocol}://{host_ip}:{args.port}")
    if setup_tunnel is not None:
        tunnel = setup_tunnel('localhost', args.port, tunnel_token, None)
        logger.info(f"Tunnel: {tunnel}")

    web.run_app(app, port=args.port, ssl_context=ssl_context)


with torch.no_grad():
    main()
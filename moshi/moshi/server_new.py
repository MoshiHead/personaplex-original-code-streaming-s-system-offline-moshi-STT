# server.py — fixed pipeline for synthetic audio RAG injection
#
# Latency/choppiness fixes applied:
#   1. Voice-prompt replay SKIPPED on RAG turns (only silence + text prompt
#      + silence are re-stepped) — the full voice prompt is only replayed
#      once, at session start.
#   2. User-audio replay during a RAG turn is batched into a SINGLE
#      gpu_executor call instead of one call per frame.
#   3. mimi/other_mimi encode + STT/VAD step during "listening" are moved
#      onto gpu_executor (batched per chunk) so they don't block the event
#      loop and starve send_loop/recv_loop, which was causing choppy audio.
#   4. opus_reader is drained (and discarded) during "resetting" so no
#      backlog of mic audio builds up and bursts once generation resumes.

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
import gc
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
import random

from .client_utils import make_log, colorize
from .models import loaders, MimiModel, LMModel, LMGen
from .utils.connection import create_ssl_context, get_lan_ip
from .utils.logging import setup_logger, ColorizedLog
import moshi.models
from .build_index import text_to_audio


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


# ── ServerState ───────────────────────────────────────────────────────────────

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
        rag_min_score: float = 0.25,
        stt_hf_repo: str = "kyutai/stt-1b-en_fr-candle",
        vad_threshold: float = 0.5,
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
        # single consistent thread.
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

        if self.rag_enabled:
            # Load RAG index
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
            # Load sentence-transformers on CPU
            try:
                from sentence_transformers import SentenceTransformer
                self.rag_embedding_model = SentenceTransformer(
                    self.rag_embedding_model_name,
                    device="cpu",
                )
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
                stt_info = moshi.models.loaders.CheckpointInfo.from_hf_repo(
                    stt_hf_repo,
                )
                self.stt_mimi = stt_info.get_mimi(device=device)
                stt_lm = stt_info.get_moshi(
                    device=device,
                    dtype=torch.bfloat16,
                )
                stt_lm.eval()
                self.stt_lm_gen = moshi.models.LMGen(
                    stt_lm,
                    temp=0,
                    temp_text=0.0,
                )
                self.stt_lm_gen.streaming_forever(1)
                self.stt_mimi.streaming_forever(1)
                self.stt_tokenizer = stt_info.get_text_tokenizer()
                self.stt_padding_token_id = stt_info.raw_config.get(
                    "text_padding_token_id", 3
                )
                logger.info(
                    f"STT model loaded: "
                    f"params={sum(p.numel() for p in stt_lm.parameters()) / 1e9:.1f}B "
                    f"padding_id={self.stt_padding_token_id}"
                )
            except Exception as e:
                logger.error(f"RAG disabled — STT model load failed: {e}")
                self.rag_enabled = False

    def warmup(self):
        for _ in range(4):
            chunk = torch.zeros(
                1, 1, self.frame_size, dtype=torch.float32, device=self.device
            )
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
            f"RAG enabled={self.rag_enabled} "
            f"chunks={len(self.rag_chunks)} "
            f"vad_threshold={self.vad_threshold}"
        )
    
        requested_voice_prompt_path = None
        voice_prompt_path = None
        if self.voice_prompt_dir is not None:
            voice_prompt_filename = request.query["voice_prompt"]
            if voice_prompt_filename:
                requested_voice_prompt_path = os.path.join(
                    self.voice_prompt_dir, voice_prompt_filename
                )
            if (requested_voice_prompt_path is None
                    or not os.path.exists(requested_voice_prompt_path)):
                raise FileNotFoundError(
                    f"Voice prompt '{voice_prompt_filename}' not found "
                    f"in '{self.voice_prompt_dir}'"
                )
            voice_prompt_path = requested_voice_prompt_path
    
        if self.lm_gen.voice_prompt != voice_prompt_path:
            if voice_prompt_path.endswith('.pt'):
                self.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
            else:
                self.lm_gen.load_voice_prompt(voice_prompt_path)
    
        base_text_prompt = request.query.get("text_prompt", "")
        seed = int(request.query["seed"]) if "seed" in request.query else None
    
        # ── Per‑session state ──────────────────────────────────────────────────────
        session_accumulated_chunks: list[dict] = []
        session_user_audio_frames: list[torch.Tensor] = []
        listening = True
        generating = False
        resetting = False
        model_consecutive_pads = 0
        MODEL_IDLE_PAD_THRESHOLD = 25  # ~2s of consecutive pad tokens (12.5Hz) before considering the response done
    
        stt_token_buffer = []
        stt_in_utterance = False
        stt_silence_frame_count = 0
        stt_last_vad_end = False
        VAD_SILENCE_FRAMES_REQUIRED = 12
    
        gpu_executor = self.gpu_executor
    
        async def send_context_window(retrieved: list[dict], transcript: str):
            clog.log("info", "─" * 60)
            clog.log("info", "RAG CONTEXT WINDOW")
            clog.log("info", f"transcript: \"{transcript}\"")
            for i, c in enumerate(retrieved):
                clog.log("info",
                    f"  [{i+1}] score={c['similarity_score']:.3f} "
                    f"source={c['source']}"
                )
                clog.log("info", f"       \"{c['text'][:100]}\"")
            clog.log("info", "─" * 60)
            payload = {
                "transcript": transcript,
                "chunks": [
                    {
                        "id": c["id"],
                        "text": c["text"],
                        "source": c["source"],
                        "score": round(c["similarity_score"], 3),
                    }
                    for c in retrieved
                ],
            }
            msg = b"\x03" + json.dumps(payload).encode("utf-8")
            try:
                await ws.send_bytes(msg)
            except Exception:
                pass
    
        async def start_rag_turn(transcript: str):
            nonlocal listening, generating, resetting
            nonlocal session_user_audio_frames, session_accumulated_chunks
            nonlocal model_consecutive_pads
    
            if not listening:
                return
    
            clog.log("info", f"VAD end – starting RAG turn: \"{transcript}\"")
            listening = False
            generating = False
            resetting = True
    
            # 1. Retrieve chunks
            def _retrieve():
                return retrieve_chunks(
                    transcript, self.rag_chunks, self.rag_embeddings,
                    self.rag_embedding_model, self.rag_top_k, min_score=self.rag_min_score,
                )
            loop = asyncio.get_event_loop()
            new_chunks = await loop.run_in_executor(None, _retrieve)
            await send_context_window(new_chunks, transcript)
    
            # Keep accumulated chunks for history (optional)
            existing_ids = {c["id"] for c in session_accumulated_chunks}
            for c in new_chunks:
                if c["id"] not in existing_ids:
                    session_accumulated_chunks.append(c)
                    existing_ids.add(c["id"])
            MAX_ACCUMULATED_CHUNKS = 2
            if len(session_accumulated_chunks) > MAX_ACCUMULATED_CHUNKS:
                session_accumulated_chunks[:] = session_accumulated_chunks[-MAX_ACCUMULATED_CHUNKS:]
    
            # 2. Build combined text prompt: base + context, with a
            #    conditional branch so the model is explicitly told to
            #    decline when no relevant context was retrieved (instead
            #    of falling back on outside knowledge). The redundant
            #    "Question: {transcript}" text is dropped — the audio
            #    replay in step 6 already gives the model the question via
            #    audio embeddings, so encoding it again as text only adds
            #    extra forward passes during the prompt refresh.
            if new_chunks:
                context_text = " ".join(c["text"] for c in new_chunks[:1])[:500]  # limit length
                combined_prompt = (
                    f"{base_text_prompt.strip()} "
                    f"Use only the following context to answer. If the "
                    f"context does not contain the answer, say you don't "
                    f"have information about that. "
                    f"Context: {context_text}"
                )
            else:
                combined_prompt = (
                    f"{base_text_prompt.strip()} "
                    f"No relevant context was found for this question. "
                    f"Respond only with a brief statement that you don't "
                    f"have information about that in your available "
                    f"context. Do not attempt to answer using outside "
                    f"knowledge."
                )
            clog.log("info", f"New text prompt: {combined_prompt[:200]}...")
    
            # 3. Reset all streaming states
            self.mimi.reset_streaming()
            self.other_mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
    
            # 4. Set the new text prompt
            self.lm_gen.text_prompt_tokens = self.text_tokenizer.encode(
                wrap_with_system_tags(combined_prompt)
            )
    
            # 5. Re-prime context WITHOUT replaying the (expensive) voice
            #    prompt audio again — that was already done once at session
            #    start in step_system_prompts_async below. Only silence +
            #    refreshed text prompt + silence are re-stepped, mirroring
            #    step_system_prompts_async minus _step_voice_prompt_async.
            #    Run as ONE batched sync call on gpu_executor.
            t_refresh_start = time.monotonic()
            def _refresh_prompt_sync():
                self.lm_gen._step_audio_silence()
                self.lm_gen._step_text_prompt()
                self.lm_gen._step_audio_silence()
            await loop.run_in_executor(gpu_executor, _refresh_prompt_sync)
            self.mimi.reset_streaming()
            t_refresh_end = time.monotonic()
    
            # 6. Replay the user's original audio (the query) — batched into
            #    a SINGLE gpu_executor call instead of one call per frame.
            #    Cap the number of frames replayed to bound worst-case
            #    latency and memory: only the most recent MAX_REPLAY_FRAMES
            #    are replayed (older audio is dropped from the front).
            MAX_REPLAY_FRAMES = int(10 * self.mimi.frame_rate)  # ~10s cap
            if len(session_user_audio_frames) > MAX_REPLAY_FRAMES:
                clog.log("info",
                    f"Capping replay: {len(session_user_audio_frames)} -> "
                    f"{MAX_REPLAY_FRAMES} frames"
                )
                # in-place slice assignment — keeps the same list object so
                # the nonlocal binding shared with opus_loop stays correct
                session_user_audio_frames[:] = session_user_audio_frames[-MAX_REPLAY_FRAMES:]
    
            n_frames = len(session_user_audio_frames)
            clog.log("info", f"Replaying {n_frames} user audio frames")
            frames_to_replay = list(session_user_audio_frames)
    
            def _replay_all():
                for codes in frames_to_replay:
                    tokens = self.lm_gen.step(codes)
                    if tokens is not None:
                        # Discard output – we don't want the client to hear this
                        _ = self.mimi.decode(tokens[:, 1:9])
                        _ = self.other_mimi.decode(tokens[:, 1:9])
    
            t_replay_start = time.monotonic()
            await loop.run_in_executor(gpu_executor, _replay_all)
            t_replay_end = time.monotonic()
    
            clog.log("info",
                f"TIMING prompt_refresh={t_refresh_end - t_refresh_start:.2f}s "
                f"audio_replay={t_replay_end - t_replay_start:.2f}s "
                f"({n_frames} frames)"
            )
    
            # Send silence to keep client alive for the whole replay duration.
            # append_pcm requires PCM length to be one of Opus's fixed frame
            # sizes, so write frame_size-sized chunks individually rather
            # than one large buffer.
            if n_frames > 0:
                silent_frame = np.zeros(self.frame_size, dtype=np.float32)
                for _ in range(n_frames):
                    opus_writer.append_pcm(silent_frame)
    
            # 7. Clear the user buffer — explicitly drop GPU tensor refs
            #    before clearing so the CUDA caching allocator can reclaim
            #    them, then empty the cache to reduce fragmentation/growth
            #    across turns.
            for t in frames_to_replay:
                del t
            frames_to_replay = None
            session_user_audio_frames.clear()
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                clog.log("info",
                    f"CUDA mem: alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
                    f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB"
                )
    
            # 8. Switch to generation mode
            generating = True
            resetting = False
            model_consecutive_pads = 0
            clog.log("info", "RAG turn ready – now generating response")
    
        async def recv_loop():
            nonlocal close
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        clog.log("error", f"{ws.exception()}")
                        break
                    elif message.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                    ):
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
            nonlocal listening, generating, resetting
            nonlocal session_user_audio_frames
            nonlocal stt_token_buffer, stt_in_utterance, stt_silence_frame_count, stt_last_vad_end
            nonlocal model_consecutive_pads
    
            all_pcm_data = None
            loop = asyncio.get_event_loop()
            # Real-time budget per frame at the model's frame rate — used to
            # detect when a GPU step takes longer than the audio it produces
            # represents (i.e. the pipeline is falling behind real-time,
            # which manifests as choppy audio regardless of network).
            frame_budget_s = self.frame_size / self.mimi.sample_rate
    
            try:
                while True:
                    if close:
                        return
                    await asyncio.sleep(0.001)
    
                    # If we are resetting, drain (and discard) mic input so no
                    # backlog accumulates, and send silence to the client.
                    if resetting:
                        _ = opus_reader.read_pcm()  # discard, keep buffer drained
                        silent_pcm = np.zeros(self.frame_size, dtype=np.float32)
                        opus_writer.append_pcm(silent_pcm)
                        continue
    
                    pcm = opus_reader.read_pcm()
                    if pcm.shape[-1] == 0:
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
                            # Encode (mimi + other_mimi) and run STT/VAD in a
                            # single batched gpu_executor call so these GPU
                            # forward passes don't block the event loop and
                            # starve send_loop (which was causing choppy
                            # audio during listening).
                            def _listen_step():
                                codes = self.mimi.encode(chunk_tensor)
                                _ = self.other_mimi.encode(chunk_tensor)
                                stt_res = None
                                if self.rag_enabled and self.stt_lm_gen is not None:
                                    stt_codes = self.stt_mimi.encode(chunk_tensor)
                                    stt_res = self.stt_lm_gen.step_with_extra_heads(stt_codes)
                                return codes, stt_res
    
                            t_step0 = time.monotonic()
                            input_codes, stt_result = await loop.run_in_executor(
                                gpu_executor, _listen_step
                            )
                            t_step_dt = time.monotonic() - t_step0
                            if t_step_dt > frame_budget_s:
                                clog.log("warning",
                                    f"STEP OVERRUN (listen): {t_step_dt*1000:.1f}ms "
                                    f"(budget {frame_budget_s*1000:.1f}ms)"
                                )
    
                            # Buffer user audio for later replay
                            session_user_audio_frames.append(input_codes.clone())
    
                            # STT + VAD (CPU-side bookkeeping, unchanged)
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
                                stt_last_vad_end = (
                                    stt_silence_frame_count >= VAD_SILENCE_FRAMES_REQUIRED
                                )
                                if vad_fired and stt_in_utterance and stt_token_buffer:
                                    transcript = decode_stt_tokens(
                                        stt_token_buffer, self.stt_tokenizer, self.stt_padding_token_id
                                    )
                                    stt_token_buffer = []
                                    stt_in_utterance = False
                                    stt_silence_frame_count = 0
                                    if transcript.strip():
                                        asyncio.create_task(start_rag_turn(transcript))
    
                            # Send silence during listening
                            silent_pcm = np.zeros(self.frame_size, dtype=np.float32)
                            opus_writer.append_pcm(silent_pcm)
    
                        elif generating:
                            # Run mimi encode/decode + lm_gen.step AND a
                            # speech-presence check via the STT model in one
                            # batched gpu_executor call, so we can detect
                            # the user starting to talk (barge-in) while the
                            # model is mid-response.
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
                                    f"(budget {frame_budget_s*1000:.1f}ms)"
                                )

                            # ── Barge-in detection ──────────────────────────
                            # If the STT model sees real speech (non-pad
                            # token) while we're generating, the user has
                            # started talking over the response. Abort the
                            # current response, reset streaming state, and
                            # go back to listening so the new utterance is
                            # captured from this frame onward.
                            barge_in = False
                            if stt_result is not None:
                                stt_tokens, _vad_heads = stt_result
                                if stt_tokens is not None:
                                    stt_text_token = stt_tokens[0, 0, 0].item()
                                    if stt_text_token not in (0, 3):
                                        barge_in = True

                            if barge_in:
                                clog.log("info", "Barge-in detected — interrupting response")
                                generating = False
                                listening = True
                                model_consecutive_pads = 0
                                stt_token_buffer = []
                                stt_in_utterance = True  # already mid-utterance
                                stt_last_vad_end = False
                                stt_silence_frame_count = 0

                                self.lm_gen.reset_streaming()
                                self.mimi.reset_streaming()
                                self.other_mimi.reset_streaming()
                                if self.stt_lm_gen is not None:
                                    self.stt_lm_gen.reset_streaming()
                                if self.device.type == 'cuda':
                                    torch.cuda.empty_cache()

                                # Start the new buffer with this frame's audio
                                session_user_audio_frames.clear()
                                session_user_audio_frames.append(input_codes.clone())

                                # Send silence for this frame instead of the
                                # (now-aborted) generated audio
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
                                    msg = b"\x02" + bytes(_text, encoding="utf8")
                                    await ws.send_bytes(msg)
                                else:
                                    model_consecutive_pads += 1
                                    if model_consecutive_pads >= MODEL_IDLE_PAD_THRESHOLD:
                                        # Response finished, back to listening
                                        generating = False
                                        listening = True
                                        model_consecutive_pads = 0
                                        session_user_audio_frames.clear()
                                        stt_token_buffer = []
                                        stt_in_utterance = False
                                        stt_silence_frame_count = 0
                                        if self.stt_lm_gen is not None:
                                            self.stt_lm_gen.reset_streaming()
                                        if self.device.type == 'cuda':
                                            torch.cuda.empty_cache()
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
    
        # Set initial text prompt (used until first RAG turn)
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
            self.mimi.reset_streaming()
            self.other_mimi.reset_streaming()
            self.lm_gen.reset_streaming()
    
            if self.stt_lm_gen is not None:
                self.stt_lm_gen.reset_streaming()
    
            async def is_alive():
                if close or ws.closed:
                    return False
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=0.01)
                    if msg.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        return False
                except asyncio.TimeoutError:
                    return True
                except aiohttp.ClientConnectionError:
                    return False
                return True
    
            # Initial system prompts (voice + text) — this is the ONLY place
            # the voice prompt audio is replayed for the whole session.
            # Run via gpu_executor for thread-consistency with all later
            # lm_gen.step() calls.
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
                tasks = [
                    asyncio.create_task(recv_loop()),
                    asyncio.create_task(opus_loop()),
                    asyncio.create_task(send_loop()),
                ]
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
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
    parser.add_argument("--tokenizer",    type=str)
    parser.add_argument("--moshi-weight", type=str)
    parser.add_argument("--mimi-weight",  type=str)
    parser.add_argument("--hf-repo",      type=str, default=loaders.DEFAULT_REPO)
    parser.add_argument("--device",       type=str, default="cuda")
    parser.add_argument("--cpu-offload",  action="store_true")
    parser.add_argument("--voice-prompt-dir", type=str)
    parser.add_argument("--ssl",          type=str)
    parser.add_argument(
        "--rag-index-dir", type=str, default=None,
        help="Path to RAG index (build_index.py --mode text). Omit to disable RAG."
    )
    parser.add_argument("--rag-top-k", type=int, default=2)
    parser.add_argument(
        "--rag-min-score", type=float, default=0.25,
        help="Minimum cosine similarity to accept a retrieved chunk (default 0.25)."
    )
    parser.add_argument(
        "--stt-hf-repo", type=str, default="kyutai/stt-1b-en_fr-candle",
        help="HF repo for the STT model used for transcription and VAD."
    )
    parser.add_argument(
        "--vad-threshold", type=float, default=0.5,
        help="VAD end-of-turn probability threshold (default 0.5)."
    )

    args = parser.parse_args()
    args.voice_prompt_dir = _get_voice_prompt_dir(args.voice_prompt_dir, args.hf_repo)
    if args.voice_prompt_dir is not None:
        assert os.path.exists(args.voice_prompt_dir), \
            f"Directory missing: {args.voice_prompt_dir}"
    logger.info(f"voice_prompt_dir = {args.voice_prompt_dir}")

    static_path = _get_static_path(args.static)
    assert static_path is None or os.path.exists(static_path), \
        f"Static path does not exist: {static_path}."
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
        tunnel_token = (
            args.gradio_tunnel_token
            if args.gradio_tunnel_token
            else secrets.token_urlsafe(32)
        )

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

    lm = loaders.get_moshi_lm(
        args.moshi_weight, device=args.device, cpu_offload=args.cpu_offload
    )
    lm.eval()
    logger.info("moshi loaded")

    state = ServerState(
        mimi=mimi,
        other_mimi=other_mimi,
        text_tokenizer=text_tokenizer,
        lm=lm,
        mimi_weight = args.mimi_weight,
        device=args.device,
        voice_prompt_dir=args.voice_prompt_dir,
        save_voice_prompt_embeddings=False,
        rag_index_dir=args.rag_index_dir,
        rag_top_k=args.rag_top_k,
        rag_min_score=args.rag_min_score,
        stt_hf_repo=args.stt_hf_repo,
        vad_threshold=args.vad_threshold,
    )
    logger.info("warming up the model")
    # Run warmup on the same dedicated GPU thread that will service all
    # per-connection lm_gen.step() calls, so CUDA graph capture happens on
    # that thread and later replays from the same thread remain valid.
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

    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        ssl_context, protocol = create_ssl_context(args.ssl)

    host_ip = (
        args.host
        if args.host not in ("0.0.0.0", "::", "localhost")
        else get_lan_ip()
    )
    logger.info(f"Access the Web UI at {protocol}://{host_ip}:{args.port}")
    if setup_tunnel is not None:
        tunnel = setup_tunnel('localhost', args.port, tunnel_token, None)
        logger.info(f"Tunnel: {tunnel}")

    web.run_app(app, port=args.port, ssl_context=ssl_context)


with torch.no_grad():
    main()
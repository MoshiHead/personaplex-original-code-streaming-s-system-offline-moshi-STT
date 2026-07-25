# lm.py
# Changes from original: 
#   1. ContextLog dataclass added after existing constants
#   2. LMGen.__init__ gains rag_tokens, context_window, tokenizer params
#   3. _step_voice_prompt_frame increments ctx_log.voice_prompt_frames
#   4. _step_audio_silence_core increments ctx_log.silence_frames
#   5. Everything else is identical to the original

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from os.path import splitext
import logging
import numpy as np
import sys
from typing import Optional, Union, List, Tuple, Callable, Iterator
import sphn
import torch
from tqdm.auto import tqdm

from ..utils.sampling import sample_token
from ..utils.compile import CUDAGraphed
from ..modules.streaming import StreamingStateDict, StreamingContainer, StreamingModule, load_streaming_state
from ..modules.transformer import (
    StreamingTransformer,
    create_norm_fn,
)

logger = logging.getLogger(__name__)

AUDIO_TOKENS_PER_STREAM = 8
FRAME_RATE_HZ = 12.5
SILENCE_TOKENS = np.array([948, 243, 1178, 546, 1736, 1030, 1978, 2008], dtype=np.int64)
SINE_TOKENS    = np.array([430, 1268, 381, 1611, 1095, 1495, 56, 472], dtype=np.int64)


# ── NEW ───────────────────────────────────────────────────────────────────────
@dataclass
class ContextLog:
    """
    Records every segment that has been force-fed into the model's KV cache.
    Updated incrementally by the prompt step methods so counts are always
    accurate at any point during or after step_system_prompts().
    """
    rag_tokens:    list[int] = field(default_factory=list)
    system_tokens: list[int] = field(default_factory=list)
    voice_prompt_frames: int = 0
    silence_frames:      int = 0
    live_frames:         int = 0
    context_window:      int = 750
    frame_rate:          float = FRAME_RATE_HZ

    # ── segment boundaries (computed from counts above) ───────────────────

    @property
    def voice_end(self) -> int:
        return self.voice_prompt_frames

    @property
    def silence1_end(self) -> int:
        # first silence block sits between voice and RAG
        return self.voice_end + self.silence_frames // 2

    @property
    def rag_start(self) -> int:
        return self.silence1_end

    @property
    def rag_end(self) -> int:
        return self.rag_start + len(self.rag_tokens)

    @property
    def system_start(self) -> int:
        return self.rag_end

    @property
    def system_end(self) -> int:
        return self.system_start + len(self.system_tokens)

    @property
    def live_start(self) -> int:
        return self.system_end + self.silence_frames // 2

    @property
    def total_frames(self) -> int:
        return self.live_start + self.live_frames

    # ── window arithmetic ─────────────────────────────────────────────────

    @property
    def oldest_frame(self) -> int:
        return max(0, self.total_frames - self.context_window)

    def _segment_in_window(self, start: int, end: int) -> bool:
        return end > self.oldest_frame

    def _frames_until_exit(self, end: int) -> int:
        return max(0, end - self.oldest_frame)

    # ── report ────────────────────────────────────────────────────────────

    def report(self, tokenizer=None) -> str:
        fr = self.frame_rate
        W  = self.context_window
        T  = self.total_frames
        O  = self.oldest_frame

        def status(start, end):
            if not self._segment_in_window(start, end):
                return "✗ EXPIRED"
            left = self._frames_until_exit(end)
            return f"✓ in window  ({left} frames / {left/fr:.1f}s left)"

        rag_text = ""
        if tokenizer and self.rag_tokens:
            decoded = tokenizer.decode(self.rag_tokens)
            rag_text = f'\n│    "{decoded[:100]}{"..." if len(decoded) > 100 else ""}"'

        sys_text = ""
        if tokenizer and self.system_tokens:
            decoded = tokenizer.decode(self.system_tokens)
            sys_text = f'\n│    "{decoded[:100]}"'

        pct = 100 * T / W if W else 0
        bar_filled = int(40 * min(T, W) / W)
        bar = "█" * bar_filled + "░" * (40 - bar_filled)

        lines = [
            "┌─ CONTEXT WINDOW " + "─" * 52,
            f"│  [{bar}] {pct:.0f}%",
            f"│  frames used : {T} / {W}  |  oldest in window : {O}",
            "│",
            f"│  [0 → {self.voice_end}]  "
            f"voice prompt  ({self.voice_prompt_frames} frames, "
            f"{self.voice_prompt_frames/fr:.1f}s)  "
            f"{status(0, self.voice_end)}",
            "│",
            f"│  [{self.rag_start} → {self.rag_end}]  "
            f"RAG context   ({len(self.rag_tokens)} tokens, "
            f"{len(self.rag_tokens)/fr:.1f}s)  "
            f"{status(self.rag_start, self.rag_end)}"
            + rag_text,
            "│",
            f"│  [{self.system_start} → {self.system_end}]  "
            f"system prompt ({len(self.system_tokens)} tokens, "
            f"{len(self.system_tokens)/fr:.1f}s)  "
            f"{status(self.system_start, self.system_end)}"
            + sys_text,
            "│",
            f"│  [{self.live_start} → {T}]  "
            f"conversation  ({self.live_frames} frames, "
            f"{self.live_frames/fr:.1f}s)",
            "└" + "─" * 69,
        ]
        return "\n".join(lines)
# ── END NEW ───────────────────────────────────────────────────────────────────


@dataclass
class LMOutput:
    logits: torch.Tensor      # [B, K, T, card]
    mask: torch.Tensor        # [B, K, T]
    text_logits: torch.Tensor # [B, 1, T, text_card]
    text_mask: torch.Tensor   # [B, 1, T]


def _delay_sequence(delays: List[int], tensor: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
    B, K, T = tensor.shape
    assert len(delays) == K, (len(delays), K)
    outs = []
    for k, delay in enumerate(delays):
        assert delay >= 0
        line = tensor[:, k].roll(delay, dims=1)
        if delay > 0:
            line[:, :delay] = padding[:, k]
        outs.append(line)
    return torch.stack(outs, dim=1)


def _undelay_sequence(delays: List[int], tensor: torch.Tensor,
                      fill_value: Union[int, float] = float('NaN')) -> Tuple[torch.Tensor, torch.Tensor]:
    B, K, T, *_ = tensor.shape
    assert len(delays) == K
    mask = torch.ones(B, K, T, dtype=torch.bool, device=tensor.device)
    outs = []
    if all([delay == 0 for delay in delays]):
        return tensor, mask
    for k, delay in enumerate(delays):
        assert delay >= 0
        line = tensor[:, k].roll(-delay, dims=1)
        if delay > 0:
            line[:, -delay:] = fill_value
            mask[:, k, -delay:] = 0
        outs.append(line)
    return torch.stack(outs, dim=1), mask


def create_sinewave(duration: float, sample_rate: int) -> np.ndarray:
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def normalize_audio(wav: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    import pyloudnorm as pyln
    if wav.ndim == 2 and wav.shape[0] == 1:
        wav = wav[0]
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(wav)
    return pyln.normalize.loudness(wav, loudness, target_lufs)


def load_audio(filepath: str, sample_rate: int):
    sample_pcm, sample_sr = sphn.read(filepath)
    sample_pcm = sphn.resample(sample_pcm, src_sample_rate=sample_sr, dst_sample_rate=sample_rate)
    return sample_pcm


def _iterate_audio(sample_pcm, sample_interval_size, max_len=sys.maxsize, pad=True):
    cnt = 0
    while sample_pcm.shape[-1] > 0 and cnt < max_len:
        sample = sample_pcm[:, :sample_interval_size]
        sample_pcm = sample_pcm[:, sample_interval_size:]
        if sample_pcm.shape[-1] == 0 and pad:
            sample = np.concatenate(
                [sample, np.zeros((sample.shape[0], sample_interval_size - sample.shape[-1]))],
                axis=1,
            )
        cnt += 1
        yield sample[0:1]


def encode_from_sphn(mimi, samples, max_batch=sys.maxsize):
    device = next(mimi.parameters()).device
    current_batch = []
    done_flag = False
    max_batch = 1
    while True:
        try:
            sample = next(samples)
            tensor = torch.tensor(sample, dtype=torch.float32, device=device).unsqueeze(0)
            current_batch.append(tensor)
        except StopIteration:
            done_flag = True
        if (not done_flag) and len(current_batch) < max_batch:
            continue
        if not current_batch:
            break
        batch = torch.cat(current_batch, dim=0)
        encoded = mimi.encode(batch)
        separated = torch.unbind(encoded, dim=0)
        reshaped = [x.unsqueeze(0) for x in separated]
        detached = [x.detach().clone() for x in reshaped]
        current_batch = []
        yield from detached
        if done_flag:
            break


class ScaledEmbedding(torch.nn.Embedding):
    def __init__(self, *args, norm: bool = False, zero_idx: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = None
        if norm:
            self.norm = create_norm_fn("layer_norm", self.embedding_dim)
        assert zero_idx < 0
        self.zero_idx = zero_idx

    def forward(self, input, *args, **kwargs):
        is_zero = input == self.zero_idx
        zero = torch.zeros(1, dtype=input.dtype, device=input.device)
        input = input.clamp(min=0)
        y = super().forward(input, *args, **kwargs)
        if self.norm is not None:
            y = self.norm(y)
        return torch.where(is_zero[..., None], zero, y)


class LMModel(StreamingContainer):
    def __init__(
        self,
        delays: List[int] = [0],
        n_q: int = 8,
        dep_q: int = 8,
        card: int = 1024,
        text_card: int = 32000,
        dim: int = 128,
        num_heads: int = 8,
        hidden_scale: int = 4,
        norm: str = "layer_norm",
        norm_emb: bool = False,
        bias_proj: bool = False,
        depformer_dim: int = 256,
        depformer_dim_feedforward: int | list[int] | None = None,
        depformer_multi_linear: bool = False,
        depformer_weights_per_step: bool = False,
        depformer_weights_per_step_schedule: list[int] | None = None,
        depformer_pos_emb: str = "sin",
        existing_text_padding_id: Optional[int] = None,
        context: Optional[int] = None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.n_q = n_q
        self.dep_q = dep_q
        self.card = card
        self.text_card = text_card
        assert len(delays) == self.num_codebooks
        self.delays = delays
        self.dim = dim
        self.existing_text_padding_id = existing_text_padding_id
        self.context = context
        self.depformer_weights_per_step_schedule = depformer_weights_per_step_schedule
        if depformer_weights_per_step_schedule is not None:
            assert len(depformer_weights_per_step_schedule) == dep_q
        kwargs["context"] = context
        EmbeddingFactory = partial(
            ScaledEmbedding, norm=norm_emb, device=device, dtype=dtype, zero_idx=self.zero_token_id,
        )
        self.EmbeddingFactory = EmbeddingFactory
        self.emb = torch.nn.ModuleList([EmbeddingFactory(self.card + 1, dim) for _ in range(n_q)])
        extra_text = self.existing_text_padding_id is None
        self.text_emb = EmbeddingFactory(text_card + 1, dim)
        self.text_linear = torch.nn.Linear(dim, text_card + extra_text, bias=bias_proj)
        depformer_prefix = "depformer_"
        main_kwargs = {k: v for k, v in kwargs.items() if not k.startswith(depformer_prefix)}
        self.transformer = StreamingTransformer(
            d_model=dim, num_heads=num_heads, dim_feedforward=int(hidden_scale * dim),
            norm=norm, device=device, dtype=dtype, **main_kwargs,
        )
        self.out_norm = create_norm_fn(norm, dim)
        self.depformer_multi_linear = depformer_multi_linear
        kwargs_dep = main_kwargs.copy()
        kwargs_dep.update({k.removeprefix(depformer_prefix): v for k, v in kwargs.items() if k.startswith(depformer_prefix)})
        kwargs_dep["positional_embedding"] = depformer_pos_emb
        kwargs_dep["context"] = None
        if depformer_weights_per_step:
            kwargs_dep["weights_per_step"] = dep_q
        if depformer_multi_linear:
            self.depformer_in = torch.nn.ModuleList([torch.nn.Linear(dim, depformer_dim, bias=False) for _ in range(dep_q)])
        else:
            self.depformer_in = torch.nn.ModuleList([torch.nn.Linear(dim, depformer_dim, bias=False)])
        self.depformer_emb = torch.nn.ModuleList([EmbeddingFactory(self.card + 1, depformer_dim) for _ in range(dep_q - 1)])
        self.depformer_text_emb = EmbeddingFactory(text_card + 1, depformer_dim)
        if depformer_dim_feedforward is None:
            depformer_dim_feedforward = int(hidden_scale * depformer_dim)
        self.depformer = StreamingTransformer(
            d_model=depformer_dim, dim_feedforward=depformer_dim_feedforward,
            norm=norm, device=device, dtype=dtype, **kwargs_dep,
        )
        self.depformer.set_streaming_propagate(False)
        dim = depformer_dim
        self.linears = torch.nn.ModuleList([torch.nn.Linear(dim, self.card, bias=bias_proj) for _ in range(dep_q)])

    @property
    def initial_token_id(self) -> int:
        return self.card

    @property
    def text_initial_token_id(self) -> int:
        return self.text_card

    @property
    def text_padding_token_id(self) -> int:
        return self.text_card if self.existing_text_padding_id is None else self.existing_text_padding_id

    @property
    def end_of_text_padding_id(self) -> int:
        return 0

    @property
    def zero_token_id(self) -> int:
        return -1

    @property
    def ungenerated_token_id(self) -> int:
        return -2

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def num_codebooks(self) -> int:
        return self.n_q + 1

    @property
    def num_audio_codebooks(self) -> int:
        return self.n_q

    @property
    def audio_offset(self) -> int:
        return 1

    def _get_initial_token(self) -> torch.Tensor:
        device = next(iter(self.parameters())).device
        zero = torch.full([1, 1, 1], self.zero_token_id, device=device, dtype=torch.long)
        special = torch.full_like(zero, self.initial_token_id)
        text_special = torch.full_like(zero, self.text_initial_token_id)
        audio_token = special.expand(-1, self.num_audio_codebooks, -1)
        return torch.cat([text_special, audio_token], dim=1)

    def embed_codes(self, sequence: torch.Tensor) -> torch.Tensor:
        B, K, S = sequence.shape
        assert K == self.num_codebooks
        input_ = None
        for cb_index in range(self.num_audio_codebooks):
            audio_emb = self.emb[cb_index](sequence[:, cb_index + self.audio_offset])
            input_ = audio_emb if input_ is None else input_ + audio_emb
        text_emb = self.text_emb(sequence[:, 0])
        return text_emb if input_ is None else input_ + text_emb

    def forward_codes(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_embeddings(self.embed_codes(sequence))

    def forward_embeddings(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        transformer_out = self.transformer(input)
        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)
        assert isinstance(transformer_out, torch.Tensor)
        text_logits = self.text_linear(transformer_out)
        return transformer_out, text_logits[:, None]

    def forward_depformer(self, depformer_cb_index: int, sequence: torch.Tensor, transformer_out: torch.Tensor) -> torch.Tensor:
        B, K, S = sequence.shape
        assert K == 1 and S == 1 and transformer_out.shape[1] == 1
        depformer_input = transformer_out
        if self.depformer_multi_linear:
            depformer_input = self.depformer_in[depformer_cb_index](depformer_input)
        else:
            depformer_input = self.depformer_in[0](depformer_input)
        if depformer_cb_index == 0:
            last_token_input = self.depformer_text_emb(sequence[:, 0])
        else:
            last_token_input = self.depformer_emb[depformer_cb_index - 1](sequence[:, 0])
        depformer_input = depformer_input + last_token_input
        dep_output = self.depformer(depformer_input)
        logits = self.linears[depformer_cb_index](dep_output)
        logits = logits[:, None]
        assert logits.dim() == 4
        return logits

    def forward_depformer_training(self, sequence: torch.Tensor, transformer_out: torch.Tensor) -> torch.Tensor:
        B, K, T = sequence.shape
        Ka = self.dep_q
        assert K == self.num_codebooks
        depformer_inputs = []
        for cb_index in range(Ka):
            if self.depformer_multi_linear:
                linear_index = cb_index
                if self.depformer_weights_per_step_schedule is not None:
                    linear_index = self.depformer_weights_per_step_schedule[cb_index]
                transformer_in = self.depformer_in[linear_index](transformer_out)
            else:
                transformer_in = self.depformer_in[0](transformer_out)
            if cb_index == 0:
                token_in = self.depformer_text_emb(sequence[:, 0])
            else:
                token_in = self.depformer_emb[cb_index - 1](sequence[:, cb_index + self.audio_offset - 1])
            depformer_inputs.append(token_in + transformer_in)
        depformer_input = torch.stack(depformer_inputs, 2).view(B * T, Ka, -1)
        depformer_output = self.depformer(depformer_input)
        all_logits = [self.linears[cb_index](depformer_output[:, cb_index]).view(B, T, -1) for cb_index in range(Ka)]
        logits = torch.stack(all_logits, 1)
        assert logits.dim() == 4
        return logits

    def forward_train(self, codes: torch.Tensor):
        B, K, T = codes.shape
        initial = self._get_initial_token().expand(B, -1, -1)
        delayed_codes = _delay_sequence(self.delays, codes, initial)
        delayed_codes = torch.cat([initial, delayed_codes], dim=2)
        transformer_out, text_logits = self.forward_codes(delayed_codes[:, :, :-1])
        logits = self.forward_depformer_training(delayed_codes[:, :, 1:], transformer_out)
        logits, logits_mask = _undelay_sequence(self.delays[self.audio_offset:self.audio_offset + self.dep_q], logits, fill_value=float('NaN'))
        logits_mask &= (codes[:, self.audio_offset: self.audio_offset + self.dep_q] != self.zero_token_id)
        text_logits, text_logits_mask = _undelay_sequence(self.delays[:1], text_logits, fill_value=float('NaN'))
        text_logits_mask &= (codes[:, :1] != self.zero_token_id)
        return LMOutput(logits, logits_mask, text_logits, text_logits_mask)


@dataclass
class _LMGenState:
    cache: torch.Tensor
    provided: torch.Tensor
    initial: torch.Tensor
    graphed_main: CUDAGraphed
    graphed_embeddings: CUDAGraphed
    graphed_depth: CUDAGraphed
    offset: int = 0

    def reset(self):
        self.offset = 0
        self.provided[:] = False


@torch.no_grad()
def create_loss_report(state_cache, lm_model, text_logits, audio_logits, target, sampled_text_token, sampled_audio_tokens, target_position):
    report = {}
    B = state_cache.shape[0]
    model_tokens = torch.zeros_like(state_cache[:, :, target_position])
    model_tokens[:, 0] = sampled_text_token
    model_tokens[:, 1 : lm_model.dep_q + 1] = sampled_audio_tokens
    report.update({"forced_tokens": torch.zeros((B, lm_model.dep_q + 1)), "model_tokens": torch.zeros((B, lm_model.dep_q + 1)), "ranks_of_forced": torch.zeros((B, lm_model.dep_q + 1)), "losses": torch.zeros((B, lm_model.dep_q+1))})
    report["model_tokens"] = model_tokens.clone()
    report["forced_tokens"] = target.clone()
    text_logits = text_logits.squeeze(dim=1).squeeze(dim=1)
    target = target[:, 0].squeeze(1).clone()
    text_probs = torch.softmax(text_logits, dim=-1)
    text_ranks = torch.argsort(text_probs, dim=-1, descending=True)
    for b in range(B):
        forced_token = target[b].item()
        try:
            rank = (text_ranks[b] == forced_token).nonzero().item()
        except RuntimeError:
            rank = lm_model.zero_token_id
        report["ranks_of_forced"][b, 0] = rank
    target[target == lm_model.text_initial_token_id] = -100
    report["losses"][:, 0] = torch.nn.functional.cross_entropy(text_logits, target, ignore_index=-100)
    for k in range(lm_model.dep_q):
        target = target[:, k+1].squeeze(1).clone()
        channel_logits = audio_logits[:, k, :]
        audio_probs = torch.softmax(channel_logits, dim=-1)
        audio_ranks = torch.argsort(audio_probs, dim=-1, descending=True)
        for b in range(B):
            forced_token = target[b].item()
            try:
                rank = (audio_ranks[b] == forced_token).nonzero().item()
            except RuntimeError:
                rank = lm_model.zero_token_id
            report["ranks_of_forced"][b, k + 1] = rank
        target[target == lm_model.initial_token_id] = -100
        report["losses"][:, k + 1] = torch.nn.functional.cross_entropy(channel_logits, target, ignore_index=-100)
    return report


class LMGen(StreamingModule[_LMGenState]):
    def __init__(
        self,
        lm_model: LMModel,
        device: str | torch.device,
        use_sampling: bool = True,
        temp: float = 0.8,
        temp_text: float = 0.7,
        top_k: int = 250,
        top_k_text: int = 25,
        check: bool = False,
        report_loss: bool = False,
        return_logits: bool = False,
        audio_silence_frame_cnt: int = 1,
        text_prompt_tokens: Optional[list[int]] = None,
        save_voice_prompt_embeddings: bool = False,
        sample_rate: int = 32000,
        frame_rate: int = FRAME_RATE_HZ,
        # ── NEW ──────────────────────────────────────────────────────────
        rag_tokens: Optional[list[int]] = None,
        context_window: int = 750,
        tokenizer=None,
        # ─────────────────────────────────────────────────────────────────
    ):
        assert not lm_model.training
        super().__init__()
        self.lm_model = lm_model
        self.use_sampling = use_sampling
        self.temp = temp
        self.temp_text = temp_text
        self.top_k = top_k
        self.top_k_text = top_k_text
        self.text_prompt_tokens = text_prompt_tokens
        self.audio_silence_frame_cnt = audio_silence_frame_cnt
        self.voice_prompt = None
        self.zero_text_code = 3
        self._frame_rate = frame_rate
        self._sample_rate = sample_rate
        self._frame_size = int(self._sample_rate / self._frame_rate)
        self._zero_frame = torch.zeros(1, 1, self._frame_size, device=device)
        duration = self._frame_size / self._sample_rate
        sine = create_sinewave(duration, self._sample_rate)
        self._sine_frame = torch.tensor(sine, device=device).unsqueeze(0).unsqueeze(0)
        self.check = check
        self.report_loss = report_loss
        if report_loss:
            return_logits = True
        self.return_logits = return_logits
        self.max_delay = max(lm_model.delays)
        self.delays_cuda = torch.tensor(lm_model.delays, device=lm_model.device, dtype=torch.long)
        self.save_voice_prompt_embeddings = save_voice_prompt_embeddings
        self.voice_prompt_audio: Optional[torch.Tensor] = None
        self.voice_prompt_cache: Optional[torch.Tensor] = None
        self.voice_prompt_embeddings: Optional[torch.Tensor] = None

        # ── NEW: build ContextLog ─────────────────────────────────────────
        self.ctx_log = ContextLog(
            rag_tokens=rag_tokens or [],
            system_tokens=text_prompt_tokens or [],
            context_window=context_window,
            frame_rate=float(frame_rate),
        )
        self._tokenizer = tokenizer
        self.pending_injection_tokens: list[int] = []
        self._injected_offsets: set[int] = set()
        # ─────────────────────────────────────────────────────────────────

    def _init_streaming_state(self, batch_size: int) -> _LMGenState:
        lm_model = self.lm_model
        initial = lm_model._get_initial_token()
        cache = torch.full(
            (batch_size, self.lm_model.num_codebooks, self.max_delay + 3),
            lm_model.ungenerated_token_id, device=lm_model.device, dtype=torch.long,
        )
        provided = torch.full(
            (batch_size, self.lm_model.num_codebooks, self.max_delay + 3),
            False, device=lm_model.device, dtype=torch.bool,
        )
        disable = lm_model.device.type != 'cuda'
        graphed_main = CUDAGraphed(lm_model.forward_codes, disable=disable)
        graphed_embeddings = CUDAGraphed(lm_model.forward_embeddings, disable=disable)
        graphed_depth = CUDAGraphed(self.depformer_step, disable=disable)
        return _LMGenState(cache, provided, initial, graphed_main, graphed_embeddings, graphed_depth)

    def queue_text_injection(self, tokens: list[int]):
        self.pending_injection_tokens.extend(tokens)

    def pop_injection_token(self) -> Optional[int]:
        if self.pending_injection_tokens:
            token = self.pending_injection_tokens.pop(0)
            state = self._streaming_state
            target_offset = state.offset + self.max_delay
            self._injected_offsets.add(target_offset)
            return token
        return None
    
    def is_current_output_injected(self) -> bool:
        state = self._streaming_state
        return (state.offset - 1) in self._injected_offsets
    
    @torch.no_grad()
    def prepare_step_input(self, input_tokens=None, moshi_tokens=None, text_token=None):
        state = self._streaming_state
        if state is None:
            raise RuntimeError("You should wrap those calls with a `with lm_gen.streaming(): ...`.")
        lm_model = self.lm_model
        needed_tokens = lm_model.num_codebooks - AUDIO_TOKENS_PER_STREAM - 1
        CT = state.cache.shape[2]

        if input_tokens is not None:
            assert input_tokens.dim() == 3
            B, Ki, S = input_tokens.shape
            assert S == 1
            assert Ki == needed_tokens
            for q_other in range(input_tokens.shape[1]):
                k = AUDIO_TOKENS_PER_STREAM + 1 + q_other
                delay = lm_model.delays[k]
                write_position = (state.offset + delay) % CT
                state.cache[:, k, write_position : write_position + 1] = input_tokens[:, q_other]
                state.provided[:, k, write_position : write_position + 1] = True

        if moshi_tokens is not None:
            assert moshi_tokens.dim() == 3
            B, Ki, S = moshi_tokens.shape
            assert S == 1
            assert Ki == needed_tokens
            for q_moshi in range(moshi_tokens.shape[1]):
                k = 1 + q_moshi
                delay = lm_model.delays[k]
                write_position = (state.offset + delay) % CT
                state.cache[:, k, write_position : write_position + 1] = moshi_tokens[:, q_moshi]
                state.provided[:, k, write_position : write_position + 1] = True

        if text_token is not None:
            write_position = (state.offset + lm_model.delays[0]) % CT
            state.cache[:, 0, write_position] = text_token
            state.provided[:, 0, write_position] = True

        for k, delay in enumerate(lm_model.delays):
            if state.offset <= delay:
                state.cache[:, k, state.offset % CT] = state.initial[:, k, 0]
                state.provided[:, k, state.offset % CT] = True

        if state.offset == 0:
            state.cache[:, :, 0] = state.initial[:, :, 0]
            state.offset += 1
            return None

        model_input_position = (state.offset - 1) % CT
        target_position = state.offset % CT
        input_ = state.cache[:, :, model_input_position : model_input_position + 1]
        target_ = state.cache[:, :, target_position : target_position + 1]
        provided_ = state.provided[:, :, target_position : target_position + 1]

        if self.check:
            assert not (input_ == lm_model.ungenerated_token_id).any()
            assert (input_[:, lm_model.audio_offset :] <= lm_model.card).all()
            assert (input_[:, :1] <= lm_model.text_card).all()
        return input_, provided_, target_, model_input_position, target_position

    @torch.no_grad()
    def step(self, input_tokens=None, moshi_tokens=None, text_token=None, return_embeddings=False):
        state = self._streaming_state
        lm_model = self.lm_model
        prepared_inputs = self.prepare_step_input(input_tokens, moshi_tokens, text_token)
        if prepared_inputs is None:
            return (None, None) if self.report_loss or self.return_logits else None
        input_, provided_, target_, model_input_position, target_position = prepared_inputs
        if self.check:
            assert not (input_ == lm_model.ungenerated_token_id).any()
            assert (input_[:, lm_model.audio_offset :] <= lm_model.card).all()
            assert (input_[:, :1] <= lm_model.text_card).all()
        embeddings = None
        if return_embeddings:
            embeddings = self.lm_model.embed_codes(input_)
        transformer_out, text_logits = state.graphed_main(input_)
        output = self.process_transformer_output(transformer_out, text_logits, provided_, target_, model_input_position, target_position)
        if return_embeddings:
            return output, embeddings
        return output

    @torch.no_grad()
    def step_embeddings(self, embeddings: torch.Tensor):
        state = self._streaming_state
        lm_model = self.lm_model
        needed_input_tokens = lm_model.num_codebooks - AUDIO_TOKENS_PER_STREAM - 1
        _dummy_audio_token = lm_model._get_initial_token()
        while True:
            prepared_inputs = self.prepare_step_input(
                input_tokens=_dummy_audio_token[:, 1:1+needed_input_tokens],
                moshi_tokens=_dummy_audio_token[:, 1+needed_input_tokens:],
                text_token=self.zero_text_code,
            )
            if prepared_inputs is not None:
                break
        _, provided_, target_, model_input_position, target_position = prepared_inputs
        transformer_out, text_logits = state.graphed_embeddings(embeddings)
        return self.process_transformer_output(transformer_out, text_logits, provided_, target_, model_input_position, target_position)

    @torch.no_grad()
    def process_transformer_output(self, transformer_out, text_logits, provided_, target_, model_input_position, target_position):
        state = self._streaming_state
        lm_model = self.lm_model
        sampled_text_token = sample_token(text_logits.float(), self.use_sampling, self.temp_text, self.top_k_text)
        assert sampled_text_token.dim() == 3 and sampled_text_token.shape[2] == 1 and sampled_text_token.shape[1] == 1
        sampled_text_token = sampled_text_token[:, 0, 0]
        next_text_token = torch.where(provided_[:, 0, 0], target_[:, 0, 0], sampled_text_token)
        if self.return_logits:
            sampled_audio_tokens, audio_logits = state.graphed_depth(next_text_token, transformer_out, target_[:,lm_model.audio_offset:,0], provided_[:,lm_model.audio_offset:,0])
        else:
            sampled_audio_tokens = state.graphed_depth(next_text_token, transformer_out, target_[:,lm_model.audio_offset:,0], provided_[:,lm_model.audio_offset:,0])
        state.provided[:, :, model_input_position] = False
        state.cache[:, 0, target_position] = torch.where(~state.provided[:, 0, target_position], sampled_text_token, state.cache[:, 0, target_position])
        state.cache[:, 1 : lm_model.dep_q + 1, target_position] = torch.where(~state.provided[:, 1 : lm_model.dep_q + 1, target_position], sampled_audio_tokens, state.cache[:, 1 : lm_model.dep_q + 1, target_position])
        report = {}
        if self.report_loss:
            report = create_loss_report(state.cache, lm_model, text_logits, audio_logits, target_, sampled_text_token, sampled_audio_tokens, target_position)
        if state.offset <= self.max_delay:
            state.offset += 1
            if self.report_loss:
                return None, report
            if self.return_logits:
                return None, None
            return None
        B = state.cache.shape[0]
        CT = state.cache.shape[2]
        gen_delays_cuda = self.delays_cuda[: lm_model.dep_q + 1]
        index = (((state.offset - self.max_delay + gen_delays_cuda) % CT).view(1, -1, 1).expand(B, -1, 1))
        out = state.cache.gather(dim=2, index=index)
        state.offset += 1
        if self.report_loss:
            return out, report
        elif self.return_logits and not self.report_loss:
            return out, (text_logits.clone(), audio_logits.clone())
        return out

    def load_voice_prompt(self, voice_prompt: str):
        self.voice_prompt = voice_prompt
        raw_audio = load_audio(voice_prompt, self._sample_rate)
        raw_audio = normalize_audio(raw_audio, self._sample_rate, -24.0)
        if raw_audio.ndim == 1:
            raw_audio = raw_audio[None, :]
        self.voice_prompt_audio = raw_audio
        self.voice_prompt_cache = None
        self.voice_prompt_embeddings = None

    def load_voice_prompt_embeddings(self, path: str):
        self.voice_prompt = path
        state = torch.load(path)
        self.voice_prompt_audio = None
        self.voice_prompt_embeddings = state["embeddings"].to(self.lm_model.device)
        self.voice_prompt_cache = state["cache"].to(self.lm_model.device)

    def _encode_zero_frame(self) -> torch.Tensor:
        return torch.as_tensor(SILENCE_TOKENS, dtype=torch.long, device=self.lm_model.device).view(1, 8, 1)

    def _encode_sine_frame(self) -> torch.Tensor:
        return torch.as_tensor(SINE_TOKENS, dtype=torch.long, device=self.lm_model.device).view(1, 8, 1)

    def _encode_voice_prompt_frames(self, mimi):
        return encode_from_sphn(mimi, _iterate_audio(self.voice_prompt_audio, sample_interval_size=self._frame_size, pad=True), max_batch=1)

    def _step_voice_prompt_frame(self, voice_prompt_frame_tokens, saved_embeddings=None):
        out = self.step(
            moshi_tokens=voice_prompt_frame_tokens,
            text_token=self.zero_text_code,
            input_tokens=self._encode_sine_frame(),
            return_embeddings=self.save_voice_prompt_embeddings,
        )
        if out is not None and self.save_voice_prompt_embeddings:
            _, embeddings = out
            saved_embeddings.append(embeddings)
        # ── NEW ──
        self.ctx_log.voice_prompt_frames += 1

    def _step_voice_prompt_core(self, mimi) -> Iterator[None]:
        if self.voice_prompt_embeddings is not None:
            for next_embed in self.voice_prompt_embeddings:
                yield
                self.step_embeddings(next_embed)
                # ── NEW ──
                self.ctx_log.voice_prompt_frames += 1
            state = self._streaming_state
            state.cache.copy_(self.voice_prompt_cache)
            return
        elif self.voice_prompt_audio is not None:
            saved_embeddings = []
            for voice_prompt_frame_tokens in self._encode_voice_prompt_frames(mimi):
                yield
                self._step_voice_prompt_frame(voice_prompt_frame_tokens, saved_embeddings)
            yield
            if self.save_voice_prompt_embeddings:
                torch.save(
                    {"embeddings": torch.stack(saved_embeddings, dim=0).detach().cpu(), "cache": self._streaming_state.cache},
                    splitext(self.voice_prompt)[0] + ".pt",
                )
        print('Done loading voice prompt.')

    def _step_voice_prompt(self, mimi):
        for _ in self._step_voice_prompt_core(mimi):
            pass

    async def _step_voice_prompt_async(self, mimi, is_alive=None):
        for _ in self._step_voice_prompt_core(mimi):
            if is_alive is not None and not await is_alive():
                break

    def _step_audio_silence_core(self) -> Iterator[None]:
        for _ in range(self.audio_silence_frame_cnt):
            yield
            self.step(
                moshi_tokens=self._encode_zero_frame(),
                text_token=self.zero_text_code,
                input_tokens=self._encode_sine_frame(),
            )
            # ── NEW ──
            self.ctx_log.silence_frames += 1
        print('Done loading audio silence.')

    def _step_audio_silence(self):
        for _ in self._step_audio_silence_core():
            pass

    async def _step_audio_silence_async(self, is_alive=None):
        for _ in self._step_audio_silence_core():
            if is_alive is not None and not await is_alive():
                break

    def _step_text_prompt_core(self) -> Iterator[None]:
        for text_prompt_token in self.text_prompt_tokens:
            yield
            self.step(
                moshi_tokens=self._encode_zero_frame(),
                text_token=text_prompt_token,
                input_tokens=self._encode_sine_frame(),
            )
        print('Done loading text prompt.')

    def _step_text_prompt(self):
        for _ in self._step_text_prompt_core():
            pass

    async def _step_text_prompt_async(self, is_alive=None):
        for _ in self._step_text_prompt_core():
            if is_alive is not None and not await is_alive():
                break

    async def step_system_prompts_async(self, mimi, is_alive=None):
        await self._step_voice_prompt_async(mimi, is_alive)
        await self._step_audio_silence_async(is_alive)
        await self._step_text_prompt_async(is_alive)
        await self._step_audio_silence_async(is_alive)

    def step_system_prompts(self, mimi):
        self._step_voice_prompt(mimi)
        self._step_audio_silence()
        self._step_text_prompt()
        self._step_audio_silence()

    def depformer_step(self, text_token, transformer_out, audio_tokens, audio_provided):
        (B,) = text_token.shape
        prev_token = text_token
        lm_model = self.lm_model
        depformer_tokens = []
        depformer_logits = []
        assert not lm_model.depformer.is_streaming
        with lm_model.depformer.streaming(B):
            for cb_index in range(lm_model.dep_q):
                input_ = prev_token[:, None, None]
                logits = lm_model.forward_depformer(cb_index, input_, transformer_out)
                if self.return_logits:
                    assert logits.shape == (B, 1, 1, lm_model.card)
                    depformer_logits.append(logits.squeeze(1).squeeze(1).float())
                next_token = sample_token(logits.float(), self.use_sampling, self.temp, self.top_k)
                assert next_token.shape == (B, 1, 1)
                next_token = next_token[:, 0, 0]
                prev_token = torch.where(audio_provided[:, cb_index], audio_tokens[:, cb_index], next_token)
                depformer_tokens.append(next_token)
        assert len(depformer_tokens) == lm_model.dep_q
        tokens = torch.stack(depformer_tokens, dim=1)
        assert tokens.shape == (B, lm_model.dep_q)
        if self.return_logits:
            all_logits = torch.stack(depformer_logits, dim=1)
            assert all_logits.shape == (B, lm_model.dep_q, lm_model.card)
            return tokens, all_logits
        return tokens
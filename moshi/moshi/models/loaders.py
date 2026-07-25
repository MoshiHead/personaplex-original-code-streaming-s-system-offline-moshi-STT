# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Retrieves the pretrained models for Moshi and Mimi."""
from dataclasses import dataclass
import json
from pathlib import Path
import logging
from typing import Optional

from huggingface_hub import hf_hub_download
import sentencepiece
from safetensors.torch import load_model, load_file
import torch

logger = logging.getLogger(__name__)

from .compression import MimiModel
from .lm import LMModel
from ..modules import SEANetEncoder, SEANetDecoder, transformer
from ..quantization import SplitResidualVectorQuantizer

SAMPLE_RATE = 24000
FRAME_RATE = 12.5

TEXT_TOKENIZER_NAME = 'tokenizer_spm_32k_3.model'
MOSHI_NAME = 'model.safetensors'
MIMI_NAME = 'tokenizer-e351c8d8-checkpoint125.safetensors'
DEFAULT_REPO = 'nvidia/personaplex-7b-v1'


_seanet_kwargs = {
    "channels": 1,
    "dimension": 512,
    "causal": True,
    "n_filters": 64,
    "n_residual_layers": 1,
    "activation": "ELU",
    "compress": 2,
    "dilation_base": 2,
    "disable_norm_outer_blocks": 0,
    "kernel_size": 7,
    "residual_kernel_size": 3,
    "last_kernel_size": 3,
    # We train using weight_norm but then the weights are pre-processed for inference so
    # that we can use a normal convolution.
    "norm": "none",
    "pad_mode": "constant",
    "ratios": [8, 6, 5, 4],
    "true_skip": True,
}
_quantizer_kwargs = {
    "dimension": 256,
    "n_q": 32,
    "bins": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimension": _seanet_kwargs["dimension"],
}
_transformer_kwargs = {
    "d_model": _seanet_kwargs["dimension"],
    "num_heads": 8,
    "num_layers": 8,
    "causal": True,
    "layer_scale": 0.01,
    "context": 250,
    "conv_layout": True,
    "max_period": 10000,
    "gating": "none",
    "norm": "layer_norm",
    "positional_embedding": "rope",
    "dim_feedforward": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimensions": [_seanet_kwargs["dimension"]],
}

_lm_kwargs = {
    "dim": 4096,
    "text_card": 32000,
    "existing_text_padding_id": 3,
    "n_q": 16,
    "dep_q": 8,
    "card": _quantizer_kwargs["bins"],
    "num_heads": 32,
    "num_layers": 32,
    "hidden_scale": 4.125,
    "causal": True,
    "layer_scale": None,
    "context": 3000,
    "max_period": 10000,
    "gating": "silu",
    "norm": "rms_norm_f32",
    "positional_embedding": "rope",
    "depformer_dim": 1024,
    "depformer_dim_feedforward": int(4.125 * 1024),
    "depformer_num_heads": 16,
    "depformer_num_layers": 6,
    "depformer_causal": True,
    "depformer_layer_scale": None,
    "depformer_multi_linear": True,
    "depformer_context": 8,
    "depformer_max_period": 10000,
    "depformer_gating": "silu",
    "depformer_pos_emb": "none",
    "depformer_weights_per_step": True,
    "delays": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
}


def _is_safetensors(path: Path | str) -> bool:
    return Path(path).suffix in (".safetensors", ".sft", ".sfts")


def get_mimi(filename: str | Path,
             device: torch.device | str = 'cpu') -> MimiModel:
    """Return a pretrained Mimi model."""
    encoder = SEANetEncoder(**_seanet_kwargs)
    decoder = SEANetDecoder(**_seanet_kwargs)
    encoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    decoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    quantizer = SplitResidualVectorQuantizer(
        **_quantizer_kwargs,
    )
    model = MimiModel(
        encoder,
        decoder,
        quantizer,
        channels=1,
        sample_rate=SAMPLE_RATE,
        frame_rate=FRAME_RATE,
        encoder_frame_rate=SAMPLE_RATE / encoder.hop_length,
        causal=True,
        resample_method="conv",
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
    ).to(device=device)
    model.eval()
    if _is_safetensors(filename):
        load_model(model, filename)
    else:
        pkg = torch.load(filename, "cpu")
        model.load_state_dict(pkg["model"])
    model.set_num_codebooks(8)
    return model


def get_moshi_lm(
    filename: str | Path | None,
    copy_missing_weights: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    delays=None,
    cpu_offload: bool = False,
    lm_kwargs: Optional[dict] = None,
) -> LMModel:
    """Return a pretrained Moshi LM model.

    Args:
        filename: Path to model weights.
        copy_missing_weights: Whether to copy missing weights from existing layers.
        device: Target device for the model.
        dtype: Data type for model weights.
        delays: Optional custom delays configuration.
        cpu_offload: If True, offload model layers to CPU when GPU memory is
                     insufficient. Uses accelerate's device_map="auto".
        lm_kwargs: If given, used verbatim as the LMModel constructor kwargs
                   instead of the fixed PersonaPlex `_lm_kwargs` below (plus
                   its `dep_q=16` override). Used by CheckpointInfo to load
                   auxiliary checkpoints (e.g. the STT model) that have a
                   different architecture from the main 7B PersonaPlex LM.
    """
    if lm_kwargs is None:
        # Copy to avoid mutating a shared/global dict
        lm_kwargs = dict(_lm_kwargs)
        lm_kwargs["dep_q"] = 16
    else:
        lm_kwargs = dict(lm_kwargs)
    if delays is not None:
        lm_kwargs["delays"] = delays

    if cpu_offload and filename is not None:
        return _get_moshi_lm_with_offload(
            filename, copy_missing_weights, device, dtype, lm_kwargs
        )

    # Init with meta device to avoid init dummy memory
    init_device = "meta" if filename is not None else device
    model = LMModel(device=init_device, dtype=dtype, **lm_kwargs)
    if filename is None:
        model.to(device=device, dtype=dtype)
        model.eval()
        return model

    filename = str(filename)

    # Load state_dict
    if filename.endswith(".safetensors"):
        # safetensors does not support mps directly
        dev = torch.device(device) if isinstance(device, str) else device
        if dev.type == "mps":
            state_dict = load_file(filename, device="cpu")
        else:
            state_dict = load_file(filename, device=dev.type)
    else:
        # torch checkpoint
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
    # Patch 1: expand depformer self_attn weights if needed
    model_sd = model.state_dict()
    for name, tensor in list(state_dict.items()):
        if "depformer" in name and "self_attn" in name and name in model_sd:
            if tensor.shape != model_sd[name].shape:
                print("Expanding %s", name)
                missing = (
                    tensor
                    if copy_missing_weights
                    else model_sd[name][tensor.shape[0] :]
                )
                state_dict[name] = torch.concat([tensor, missing], dim=0)

    # Patch 2: fill missing keys by copying 0..7 -> 8..15 for certain groups
    if copy_missing_weights:
        to_replace = ["gating", "linears", "depformer_in", "depformer_emb"]
        for name in model_sd.keys():
            if name in state_dict:
                continue
            replaced = False
            for old, new in zip(range(8), range(8, 16)):
                for rep in to_replace:
                    needle = f"{rep}.{new}."
                    if needle in name:
                        src = name.replace(needle, f"{rep}.{old}.")
                        if src in state_dict:
                            print("Replacing %s <- %s", name, src)
                            state_dict[name] = state_dict[src]
                            replaced = True
                        break
                if replaced:
                    break
            if not replaced:
                print("Missing %s", name)

    # Assign weights to target device
    dev = torch.device(device) if isinstance(device, str) else device
    for key in state_dict:
        state_dict[key] = state_dict[key].to(device=dev, dtype=dtype)
    
    model.load_state_dict(state_dict, strict=False, assign=True)

    # With assign=True, load_state_dict only replaces parameters/buffers whose name matched a
    # state_dict key -- anything unmatched is still the meta tensor it was constructed with above
    # (empty, no storage). Neither .to() nor a plain `.data = real_tensor` reassignment can move a
    # meta tensor (the latter raises "Attempted to call variable.set_data(tensor), but variable and
    # tensor have incompatible tensor type" -- meta participates in a different dispatch key set
    # that the .data setter's compatibility check rejects, same underlying reason .to() fails).
    # This matters for the (expected, already-logged-as-an-error) case of a checkpoint that doesn't
    # cover every parameter the constructed LMModel expects -- e.g. copy_missing_weights=False in
    # CheckpointInfo.get_moshi, used for auxiliary checkpoints like the STT model, which
    # intentionally skips the PersonaPlex-specific "fill missing keys" patch above. Replace any
    # leftover meta parameters/buffers in place on their owning module (not through the .data
    # setter) so loading degrades (uncovered weights start at zero) instead of crashing after
    # already having pulled the whole checkpoint onto the GPU.
    for submodule in model.modules():
        for name, param in list(submodule._parameters.items()):
            if param is not None and param.is_meta:
                submodule._parameters[name] = torch.nn.Parameter(
                    torch.zeros(param.shape, dtype=param.dtype, device=dev),
                    requires_grad=param.requires_grad,
                )
        for name, buf in list(submodule._buffers.items()):
            if buf is not None and buf.is_meta:
                submodule._buffers[name] = torch.zeros(buf.shape, dtype=buf.dtype, device=dev)

    model.eval()
    return model.to(device=device, dtype=dtype)


def _get_moshi_lm_with_offload(
    filename: str | Path,
    copy_missing_weights: bool,
    device: torch.device | str,
    dtype: torch.dtype,
    lm_kwargs: dict,
) -> LMModel:
    """Load Moshi LM with CPU offloading using accelerate.

    This function distributes model layers across GPU and CPU based on
    available GPU memory. Layers that don't fit on GPU are kept on CPU
    and moved to GPU only during forward pass.
    """
    try:
        from accelerate import infer_auto_device_map, dispatch_model
    except ImportError:
        raise ImportError(
            "CPU offloading requires the 'accelerate' package. "
            "Install it with: pip install accelerate"
        )

    filename = str(filename)
    logger.info("Loading model with CPU offloading enabled")

    # First, create model on CPU to get the architecture
    model = LMModel(device="cpu", dtype=dtype, **lm_kwargs)

    # Load state_dict to CPU
    if filename.endswith(".safetensors"):
        state_dict = load_file(filename, device="cpu")
    else:
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")

    # Apply weight patches (same as non-offload path)
    model_sd = model.state_dict()
    for name, tensor in list(state_dict.items()):
        if "depformer" in name and "self_attn" in name and name in model_sd:
            if tensor.shape != model_sd[name].shape:
                logger.info(f"Expanding {name}")
                missing = (
                    tensor
                    if copy_missing_weights
                    else model_sd[name][tensor.shape[0]:]
                )
                state_dict[name] = torch.concat([tensor, missing], dim=0)

    if copy_missing_weights:
        to_replace = ["gating", "linears", "depformer_in", "depformer_emb"]
        for name in model_sd.keys():
            if name in state_dict:
                continue
            replaced = False
            for old, new in zip(range(8), range(8, 16)):
                for rep in to_replace:
                    needle = f"{rep}.{new}."
                    if needle in name:
                        src = name.replace(needle, f"{rep}.{old}.")
                        if src in state_dict:
                            logger.info(f"Replacing {name} <- {src}")
                            state_dict[name] = state_dict[src]
                            replaced = True
                        break
                if replaced:
                    break
            if not replaced:
                logger.warning(f"Missing {name}")

    model.load_state_dict(state_dict, strict=False, assign=True)

    # Determine target device
    dev = torch.device(device) if isinstance(device, str) else device

    if dev.type != "cuda":
        # If not using CUDA, just move to the target device without offloading
        logger.info(f"CPU offload requested but device is {dev}, skipping offload")
        model.to(dev)
        model.eval()
        return model

    # Infer device map based on available GPU memory
    device_map = infer_auto_device_map(
        model,
        max_memory=None,  # Let accelerate auto-detect available memory
        no_split_module_classes=["StreamingTransformerLayer"],
        dtype=dtype,
    )

    # Log the device distribution
    gpu_layers = sum(1 for v in device_map.values() if v == 0 or v == "cuda:0")
    cpu_layers = sum(1 for v in device_map.values() if v == "cpu")
    logger.info(f"Device map: {gpu_layers} modules on GPU, {cpu_layers} modules on CPU")

    # Dispatch model across devices
    model = dispatch_model(
        model,
        device_map=device_map,
        offload_dir="offload_weights",  # Directory for disk offload if needed
    )

    model.eval()
    return model


# Keys `_lm_kwargs` knows how to fill from a repo's own config.json. Anything
# in raw_config outside this set (asset filenames, HF metadata, etc.) is
# ignored rather than forwarded to LMModel — LMModel's constructor swallows
# unrecognized kwargs into the transformer's **kwargs, so a stray config key
# could otherwise surface as a confusing TypeError deep inside
# StreamingTransformer instead of a clear "unknown config key" message here.
_KNOWN_LM_CONFIG_KEYS = set(_lm_kwargs.keys())


@dataclass
class CheckpointInfo:
    """Metadata + resolved weight paths for a Moshi-family checkpoint on the
    Hub, built from that repo's own config.json rather than the fixed
    PersonaPlex `_lm_kwargs` above.

    `get_moshi_lm`/`get_mimi` above assume every checkpoint is the one fixed
    7B PersonaPlex architecture. That's wrong for auxiliary checkpoints —
    e.g. the STT model (`--stt-hf-repo`, default `kyutai/stt-1b-en_fr-candle`)
    used by the RAG pipeline's turn-detection — which is a differently-sized
    model with its own config.json. This class loads that config and
    constructs matching kwargs instead of assuming the PersonaPlex shape.
    """

    hf_repo: str
    moshi_weights: Path
    mimi_weights: Path
    tokenizer: Path
    raw_config: dict

    @staticmethod
    def from_hf_repo(hf_repo: str) -> "CheckpointInfo":
        config_path = hf_hub_download(hf_repo, "config.json")
        with open(config_path, encoding="utf-8") as f:
            raw_config = json.load(f)

        moshi_name = raw_config.get("moshi_name", MOSHI_NAME)
        mimi_name = raw_config.get("mimi_name", MIMI_NAME)
        tokenizer_name = raw_config.get("tokenizer_name", TEXT_TOKENIZER_NAME)

        moshi_weights = hf_hub_download(hf_repo, moshi_name)
        mimi_weights = hf_hub_download(hf_repo, mimi_name)
        tokenizer = hf_hub_download(hf_repo, tokenizer_name)

        return CheckpointInfo(
            hf_repo=hf_repo,
            moshi_weights=Path(moshi_weights),
            mimi_weights=Path(mimi_weights),
            tokenizer=Path(tokenizer),
            raw_config=raw_config,
        )

    def _lm_kwargs(self) -> dict:
        kwargs = dict(_lm_kwargs)
        for key in _KNOWN_LM_CONFIG_KEYS:
            if key in self.raw_config:
                kwargs[key] = self.raw_config[key]
        return kwargs

    def get_mimi(self, device: torch.device | str = "cpu") -> MimiModel:
        return get_mimi(self.mimi_weights, device)

    def get_text_tokenizer(self) -> sentencepiece.SentencePieceProcessor:
        return sentencepiece.SentencePieceProcessor(str(self.tokenizer))

    def get_moshi(
        self,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> LMModel:
        return get_moshi_lm(
            self.moshi_weights,
            copy_missing_weights=False,  # this is a standalone checkpoint, not a PersonaPlex delta to patch
            device=device,
            dtype=dtype,
            lm_kwargs=self._lm_kwargs(),
        )

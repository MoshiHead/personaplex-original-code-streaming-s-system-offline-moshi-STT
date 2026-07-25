"""
dataset.py  (modified for RAG fine-tuning)
==========================================
Changes from original
---------------------
- `build_data_loader` yields `Batch` objects that now carry `segment_metas`.
  The Batch.collate change in interleaver.py handles this automatically.
- Everything else (load_file, maybe_load_local_dataset, parse_data_sources,
  build_dataset, get_dataset_iterator, interleave_iterators) is unchanged.

The training loop accesses RAG metadata via:
    batch.rag_segment_metas   → list[SegmentMeta]  (non-None only)
    batch.segment_metas       → list[SegmentMeta | None]  (all, indexed by batch slot)
    batch.condition_attributes[i]["streaming_sum_path"]  → path to .pt reference tensor
"""

import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import sphn
import torch.distributed as dist

from .distributed import get_rank
from .interleaver import InterleavedTokenizer, Sample, Batch

logger = logging.getLogger("dataset")


AudioChunkPath = tuple[str, float]
_LOADED_DATASETS: dict[Path, list[AudioChunkPath]] = {}


def main_logger_info(message: str) -> None:
    if dist.is_initialized() and get_rank() == 0:
        logger.info(message)


def load_file(path: Path, world_size: int, rank: int) -> list[str]:
    lines = []
    with path.open() as f:
        for idx, line in enumerate(f):
            if not idx % world_size == rank:
                continue
            lines.append(line)
    return lines


def maybe_load_local_dataset(
    path: Path, rank: int, world_size: int, instruct_tokenizer: InterleavedTokenizer
) -> list[AudioChunkPath]:
    if path in _LOADED_DATASETS:
        return _LOADED_DATASETS[path]

    duration = instruct_tokenizer.duration_sec
    main_logger_info(f"Loading {path} ...")
    lines: list[str] = load_file(path, rank=rank, world_size=world_size)

    chunks: list[AudioChunkPath] = []
    for line in lines:
        data = json.loads(line)
        start_sec = 0
        while start_sec < data["duration"]:
            chunks.append((data["path"], start_sec))
            start_sec += duration

    main_logger_info(f"{path} loaded and chunked.")
    _LOADED_DATASETS[path] = chunks
    return _LOADED_DATASETS[path]


@dataclass
class DataDir:
    path: Path

    @property
    def jsonl_files(self):
        assert self.path.exists(), f"Make sure that {self.path} exists"
        jsonl_files = list(self.path.rglob("*jsonl"))
        assert len(jsonl_files) > 0, (
            f"{self.path} does not seem to have any files ending with '.jsonl'"
        )
        return jsonl_files


@dataclass
class DataFile:
    path: Path

    @property
    def jsonl_files(self):
        assert self.path.exists(), f"Make sure that {self.path} exists"
        return [self.path]


def parse_data_sources(
    pretrain_data: str,
) -> tuple[list[DataDir | DataFile], list[float]]:
    seen: set[str] = set()
    sources: list[DataDir | DataFile] = []
    weights: list[float] = []

    for source in pretrain_data.strip().split(","):
        if not source:
            continue
        source_items = source.strip().split(":")
        if len(source_items) == 1:
            path_ = source_items[0]
            weight = 1.0
        elif len(source_items) == 2:
            path_, weight_ = source_items
            weight = float(weight_)
        else:
            raise ValueError(
                f"{source} is not correctly formatted. Format: <path>:<weight>"
            )

        assert path_ not in seen, f"{path_} duplicated."
        assert weight > 0

        if Path(path_).is_dir():
            data: DataDir | DataFile = DataDir(path=Path(path_))
        elif Path(path_).is_file():
            data = DataFile(path=Path(path_))
        else:
            raise FileNotFoundError(f"{path_} does not exist.")

        sources.append(data)
        weights.append(weight)
        seen.add(path_)

    sum_weights = sum(weights)
    n_weights = [w / sum_weights for w in weights]
    assert min(n_weights) > 0
    assert abs(1 - sum(n_weights)) < 1e-8
    return sources, n_weights


def build_dataset(
    pretrain_data: str,
    instruct_tokenizer: InterleavedTokenizer,
    seed: int | None,
    rank: int,
    world_size: int,
    is_eval: bool,
    shuffle_pretrain: bool = False,
) -> Iterator[Sample]:
    sources, probabilities = parse_data_sources(pretrain_data=pretrain_data)
    shuffle = not is_eval and shuffle_pretrain

    dataset_iterators = [
        get_dataset_iterator(
            source,
            instruct_tokenizer=instruct_tokenizer,
            rank=rank,
            world_size=world_size,
            is_finite=is_eval,
            seed=seed,
            shuffle_at_epoch=shuffle,
        )
        for source in sources
    ]

    if is_eval:
        combined_iterator = itertools.chain.from_iterable(dataset_iterators)
    else:
        random_seed = np.array((seed, rank))
        rng = np.random.RandomState(seed=random_seed)
        combined_iterator = interleave_iterators(
            dataset_iterators, probabilities=probabilities, rng=rng
        )

    return combined_iterator


def get_rng(seed: int, rank: int) -> np.random.RandomState:
    random_seed = np.array((seed, rank))
    return np.random.RandomState(seed=random_seed)


def get_dataset_iterator(
    source: DataDir | DataFile,
    instruct_tokenizer: InterleavedTokenizer,
    rank: int,
    world_size: int,
    is_finite: bool,
    seed: int | None,
    shuffle_at_epoch: bool,
) -> Iterator[Sample]:
    epoch = 1
    while True:
        for jsonl_file in source.jsonl_files:
            dataset = sphn.dataset_jsonl(
                str(jsonl_file),
                duration_sec=instruct_tokenizer.duration_sec,
                num_threads=4,
                sample_rate=instruct_tokenizer.mimi.sample_rate,
                pad_last_segment=True,
            )
            if shuffle_at_epoch:
                dataset = dataset.shuffle(
                    with_replacement=False, skip=rank, step_by=world_size, seed=seed
                )
                seed += 1
            else:
                dataset = dataset.seq(skip=rank, step_by=world_size)

            for sample in dataset:
                wav = sample["data"][..., : sample["unpadded_len"]]
                yield instruct_tokenizer(wav, sample["start_time_sec"], sample["path"])

        if is_finite:
            break
        print(f"Rank {rank} finished epoch {epoch}")
        epoch += 1


def interleave_iterators(iterators: list[Iterator], probabilities, rng):
    while True:
        it_id = rng.choice(range(len(iterators)), p=probabilities)
        yield next(iterators[it_id])


# ---------------------------------------------------------------------------
# Data loader  (minimal change: Batch now carries segment_metas automatically
#               via the updated Batch.collate in interleaver.py)
# ---------------------------------------------------------------------------

def build_data_loader(
    instruct_tokenizer: Any,
    args: Any,
    batch_size: int,
    seed: int | None,
    rank: int,
    world_size: int,
    is_eval: bool,
) -> Iterator[Batch]:
    if is_eval:
        assert args.eval_data != "", "No eval data provided."
    pretrain_data = args.train_data if not is_eval else args.eval_data

    dataset = build_dataset(
        pretrain_data=pretrain_data,
        instruct_tokenizer=instruct_tokenizer,
        seed=seed,
        rank=rank,
        world_size=world_size,
        is_eval=is_eval,
        shuffle_pretrain=args.shuffle,
    )

    pad_value = instruct_tokenizer.interleaver.zero_padding   # NEW

    sample_list: list[Sample] = []
    for sample in dataset:
        assert sample.codes.dim() == 3
        assert len(sample.codes) == 1
        sample_list.append(sample)
        if len(sample_list) == batch_size:
            yield Batch.collate(sample_list, pad_value=pad_value)   # CHANGED
            sample_list = []
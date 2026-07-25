# build_index.py
"""
Converts documents into a RAG index for use with run_inference.py.

Two indexing modes:
    --mode semantic   Encode chunks through Mimi cb0 for audio-based retrieval.
                      Retrieval matches user speech directly against stored cb0 histograms.
    --mode text       Embed chunks with a text embedding model (sentence-transformers).
                      Retrieval transcribes user audio (Whisper) then does vector search.

Supported document formats:
    .txt    Plain text
    .md     Markdown (strips formatting)
    .pdf    PDF (extracts text layer)
    .html   HTML (strips tags)
    .csv    CSV (each row becomes a chunk)
    .json   JSON (flattens string values)

Usage:
    # Semantic mode
    python -m moshi.build_index \
        --docs-dir ./documents \
        --output-dir ./rag_index \
        --mode semantic \
        --mimi-weight path/to/mimi.pt \
        --tokenizer path/to/tokenizer.model \
        --device cuda

    # Text mode
    python -m moshi.build_index \
        --docs-dir ./documents \
        --output-dir ./rag_index \
        --mode text \
        --tokenizer path/to/tokenizer.model \
        --embedding-model all-MiniLM-L6-v2
"""

import argparse
import csv
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import sentencepiece
import soundfile as sf
import sphn
from huggingface_hub import hf_hub_download

from .models import loaders

SILENCE_CB0 = 948


# ── Document reading ──────────────────────────────────────────────────────────

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def read_md(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    # Strip markdown formatting — links, images, headers, bold, italic, code
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)          # images
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text) # links → keep text
    text = re.sub(r'https?://\S+', '', text)              # bare URLs
    text = re.sub(r'#{1,6}\s*', '', text)                 # headers
    text = re.sub(r'[*_`~]{1,3}', '', text)               # bold/italic/code
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)  # bullets
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # numbered lists
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def read_pdf(path: Path) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n\n".join(pages).strip()
    except ImportError:
        raise ImportError("pdfplumber required for PDF support: pip install pdfplumber")


def read_html(path: Path) -> str:
    try:
        from bs4 import BeautifulSoup
        html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    except ImportError:
        raise ImportError("beautifulsoup4 required for HTML support: pip install beautifulsoup4")


def read_csv(path: Path) -> str:
    lines = []
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Join all column values as key: value pairs
            line = " | ".join(f"{k}: {v}" for k, v in row.items() if v and v.strip())
            if line.strip():
                lines.append(line)
    return "\n".join(lines)


def read_json(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))

    def extract_strings(obj, depth=0) -> list[str]:
        if depth > 5:
            return []
        if isinstance(obj, str):
            return [obj] if obj.strip() else []
        if isinstance(obj, dict):
            parts = []
            for k, v in obj.items():
                sub = extract_strings(v, depth + 1)
                if sub:
                    parts.append(f"{k}: {' '.join(sub)}")
            return parts
        if isinstance(obj, list):
            parts = []
            for item in obj:
                parts.extend(extract_strings(item, depth + 1))
            return parts
        return [str(obj)] if obj is not None else []

    return " ".join(extract_strings(data)).strip()


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    readers = {
        '.txt':  read_txt,
        '.md':   read_md,
        '.pdf':  read_pdf,
        '.html': read_html,
        '.htm':  read_html,
        '.csv':  read_csv,
        '.json': read_json,
    }
    if suffix not in readers:
        raise ValueError(f"Unsupported format: {suffix}")
    return readers[suffix](path)


SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf', '.html', '.htm', '.csv', '.json'}


# ── Text chunking ─────────────────────────────────────────────────────────────

def split_into_chunks(text: str, max_words: int = 30) -> list[str]:
    """
    Split text into sentence-bounded chunks capped at max_words.
    Handles multi-sentence paragraphs and preserves semantic units.
    """
    # Normalize whitespace
    text = re.sub(r'\n{2,}', ' <PARA> ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+|(?<=<PARA>)\s*', text)
    sentences = [s.replace('<PARA>', '').strip() for s in sentences if s.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        # If single sentence exceeds budget, split on commas
        if len(words) > max_words and not current:
            sub_parts = re.split(r'(?<=,)\s+', sentence)
            for part in sub_parts:
                part_words = part.split()
                if current_words + len(part_words) > max_words and current:
                    chunks.append(" ".join(current))
                    current = []
                    current_words = 0
                current.extend(part_words)
                current_words += len(part_words)
            continue
        if current_words + len(words) > max_words and current:
            chunks.append(" ".join(current))
            current = []
            current_words = 0
        current.extend(words)
        current_words += len(words)

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if c.strip() and len(c.split()) >= 3]


# ── TTS ───────────────────────────────────────────────────────────────────────

def text_to_audio(text: str, sample_rate: int) -> np.ndarray:
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp_path = f.name

    try:
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        time.sleep(0.1)

        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            raise RuntimeError(f"pyttsx3 failed to generate audio at {tmp_path}.")

        audio, sr = sf.read(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != sample_rate:
        audio = sphn.resample(
            audio[None, :],
            src_sample_rate=sr,
            dst_sample_rate=sample_rate,
        )[0]

    return audio[None, :]    # (1, T)


# ── Mimi encoding ─────────────────────────────────────────────────────────────

def filter_silence_from_cb0(cb0: np.ndarray) -> np.ndarray:
    return cb0[cb0 != SILENCE_CB0]


def encode_with_fresh_mimi(mimi_weight: str, device: str, audio_pcm: np.ndarray) -> np.ndarray:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    mimi = loaders.get_mimi(mimi_weight, device)
    mimi.eval()

    tensor = torch.tensor(audio_pcm, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        codes = mimi.encode(tensor)    # (1, 8, F)

    return codes[0].cpu().numpy().astype(np.int64)    # (8, F)


# ── Text embedding (for text mode) ────────────────────────────────────────────

def build_text_embeddings(
    chunks: list[str],
    embedding_model_name: str,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers required for text mode: "
            "pip install sentence-transformers"
        )
    print(f"[build_index] loading embedding model: {embedding_model_name}")
    model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
    return embeddings.astype(np.float32)


# ── Main indexing ─────────────────────────────────────────────────────────────

def build_index(
    docs_dir: str,
    output_dir: str,
    mode: str,                          # 'semantic' or 'text'
    tokenizer_path: str,
    device: str,
    mimi_weight: Optional[str] = None,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    max_words_per_chunk: int = 30,
):
    assert mode in ('semantic', 'text'), f"mode must be 'semantic' or 'text', got {mode}"
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

    sample_rate = None
    if mode == 'semantic':
        assert mimi_weight is not None, "--mimi-weight required for semantic mode"
        print("[build_index] probing mimi for sample_rate...")
        _probe = loaders.get_mimi(mimi_weight, device)
        sample_rate = _probe.sample_rate
        del _probe
        torch.cuda.empty_cache()
        print(f"[build_index] sample_rate={sample_rate}")

    manifest: dict = {
        "mode": mode,
        "chunks": [],
        "silence_cb0_token": SILENCE_CB0,
        "embedding_model": embedding_model_name if mode == 'text' else None,
    }
    npz_data: dict = {}
    chunk_id = 0
    all_chunk_texts: list[str] = []

    doc_paths = sorted(
        p for p in Path(docs_dir).iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not doc_paths:
        raise FileNotFoundError(
            f"No supported documents found in {docs_dir}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    print(f"[build_index] found {len(doc_paths)} document(s)")

    for doc_path in doc_paths:
        print(f"\n[build_index] reading {doc_path.name} ({doc_path.suffix})")
        try:
            text = read_document(doc_path)
        except Exception as e:
            print(f"  [SKIP] failed to read: {e}")
            continue

        if not text.strip():
            print(f"  [SKIP] empty after extraction")
            continue

        chunks = split_into_chunks(text, max_words=max_words_per_chunk)
        print(f"  {len(chunks)} chunks")

        for chunk_text in chunks:
            print(f"  [{chunk_id}] \"{chunk_text[:70]}\"")
            text_tokens = tokenizer.encode(chunk_text)

            entry = {
                "id": chunk_id,
                "text": chunk_text,
                "source": doc_path.name,
                "n_text_tokens": len(text_tokens),
                "text_tokens": text_tokens,
            }

            if mode == 'semantic':
                try:
                    audio_pcm = text_to_audio(chunk_text, sample_rate)
                    full_codes = encode_with_fresh_mimi(mimi_weight, device, audio_pcm)
                except Exception as e:
                    print(f"  [SKIP] encode failed: {e}")
                    continue

                cb0_raw      = full_codes[0]
                cb0_filtered = filter_silence_from_cb0(cb0_raw)
                npz_data[f"cb0_{chunk_id}"] = cb0_filtered
                entry["n_cb0_tokens"] = len(cb0_filtered)
                entry["n_frames"]     = int(full_codes.shape[1])
                print(
                    f"  → cb0: {len(cb0_raw)}→{len(cb0_filtered)} tokens | "
                    f"{len(text_tokens)} text tokens"
                )

            all_chunk_texts.append(chunk_text)
            manifest["chunks"].append(entry)
            chunk_id += 1

    if chunk_id == 0:
        raise RuntimeError("No chunks were successfully indexed.")

    # For text mode, build embeddings over all chunks at once
    if mode == 'text':
        print(f"\n[build_index] building text embeddings for {chunk_id} chunks...")
        embeddings = build_text_embeddings(all_chunk_texts, embedding_model_name)
        npz_data["text_embeddings"] = embeddings    # (N, D)
        print(f"  embeddings shape: {embeddings.shape}")

    manifest_path = os.path.join(output_dir, "manifest.json")
    chunks_path   = os.path.join(output_dir, "chunks.npz")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    np.savez_compressed(chunks_path, **npz_data)

    print(f"\n[build_index] complete: {chunk_id} chunks, mode={mode}")
    print(f"  manifest → {manifest_path}")
    print(f"  arrays   → {chunks_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir",    required=True,  type=str)
    parser.add_argument("--output-dir",  required=True,  type=str)
    parser.add_argument("--mode",        required=True,  type=str,
                        choices=["semantic", "text"],
                        help="semantic: Mimi cb0 matching. text: Whisper+embeddings.")
    parser.add_argument("--mimi-weight", type=str, default=None,
                        help="Required for semantic mode.")
    parser.add_argument("--tokenizer",   type=str, default=None)
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model for text mode.")
    parser.add_argument("--hf-repo",     type=str, default=loaders.DEFAULT_REPO)
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--max-words",   type=int, default=30)
    args = parser.parse_args()

    mimi_weight    = args.mimi_weight
    tokenizer_path = args.tokenizer or hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)

    if args.mode == 'semantic' and mimi_weight is None:
        mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)

    build_index(
        docs_dir=args.docs_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        tokenizer_path=tokenizer_path,
        device=args.device,
        mimi_weight=mimi_weight,
        embedding_model_name=args.embedding_model,
        max_words_per_chunk=args.max_words,
    )


if __name__ == "__main__":
    main()
"""Frozen sentence embeddings for Milestone 2 (no fine-tuning)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_encoder(model_name: str = DEFAULT_MODEL_NAME):
    """Load a pretrained SentenceTransformer. Requires `sentence-transformers`."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Install M2 deps first: pip install -r requirements-m2.txt"
        ) from exc
    return SentenceTransformer(model_name)


def encode_texts(
    texts: Sequence[str],
    encoder=None,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 64,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """Return float32 embeddings of shape (n_texts, dim). Encoder weights stay frozen."""
    if encoder is None:
        encoder = load_encoder(model_name)
    vectors = encoder.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def save_embeddings(path: Union[str, Path], embeddings: np.ndarray, texts_hash: Optional[str] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"embeddings": embeddings}
    if texts_hash is not None:
        payload["texts_hash"] = np.asarray([texts_hash])
    np.savez_compressed(path, **payload)
    return path


def load_embeddings(path: Union[str, Path]) -> np.ndarray:
    data = np.load(Path(path), allow_pickle=False)
    return data["embeddings"]

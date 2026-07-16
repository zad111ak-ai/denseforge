"""Adaptive Matryoshka Embeddings with quality-aware dimension selection."""
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    vectors: dict[int, np.ndarray]
    binary: np.ndarray
    selected_dim: int
    quality_score: float


class AdaptiveEmbedder:
    """Multi-resolution embeddings using Matryoshka representations."""

    MODEL_PROFILES = {
        "nomic-ai/nomic-embed-text-v1.5": {
            "full_dim": 768,
            "dims": [64, 128, 256, 512, 768],
            "quality": {64: 0.82, 128: 0.90, 256: 0.95, 512: 0.98, 768: 1.0},
        },
        "BAAI/bge-m3": {
            "full_dim": 1024,
            "dims": [128, 256, 512, 1024],
            "quality": {128: 0.85, 256: 0.92, 512: 0.97, 1024: 1.0},
        },
        "default": {
            "full_dim": 512,
            "dims": [64, 128, 256, 512],
            "quality": {64: 0.78, 128: 0.87, 256: 0.93, 512: 1.0},
        },
    }

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", device: str = "cpu"):
        self.model_name = model_name
        self.profile = self.MODEL_PROFILES.get(model_name, self.MODEL_PROFILES["default"])
        self.full_dim = self.profile["full_dim"]
        self.supported_dims = self.profile["dims"]
        self.quality_map = self.profile["quality"]
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.model_name, device=self.device, trust_remote_code=True,
            )
            _ = self._model.encode(["warmup"], normalize_embeddings=True)
        return self._model

    def encode(self, text: str, target_dim: Optional[int] = None, task: str = "retrieval") -> EmbeddingResult:
        full_vec = self.model.encode(text, normalize_embeddings=True).astype(np.float32)[: self.full_dim]
        full_vec = self._l2_normalize(full_vec)
        vectors = {dim: self._l2_normalize(full_vec[:dim]) for dim in self.supported_dims}
        selected_dim = target_dim if target_dim else self._auto_select_dim(text, task)
        binary = np.packbits((full_vec > 0).astype(np.uint8))
        return EmbeddingResult(
            vectors=vectors, binary=binary, selected_dim=selected_dim,
            quality_score=self.quality_map.get(selected_dim, 0.95),
        )

    def encode_batch(self, texts: list[str], batch_size: int = 64) -> list[EmbeddingResult]:
        full_vecs = self.model.encode(
            texts, normalize_embeddings=True, batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
        ).astype(np.float32)
        results = []
        for text, full_vec in zip(texts, full_vecs):
            full_vec = full_vec[: self.full_dim]
            full_vec = self._l2_normalize(full_vec)
            vectors = {dim: self._l2_normalize(full_vec[:dim]) for dim in self.supported_dims}
            results.append(EmbeddingResult(
                vectors=vectors, binary=np.packbits((full_vec > 0).astype(np.uint8)),
                selected_dim=self._auto_select_dim(text, "retrieval"),
                quality_score=self.quality_map.get(256, 0.95),
            ))
        return results

    def _auto_select_dim(self, text: str, task: str) -> int:
        words = text.split()
        complexity = min(len(set(words)) / max(len(words), 1), 1.0)
        if task == "retrieval":
            if complexity > 0.7 or len(words) > 200:
                return 512
            elif complexity > 0.4:
                return 256
            return 128
        return 256

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

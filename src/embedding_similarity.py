"""Embedding-based text similarity utilities for figure placement."""

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


project_root = Path(__file__).parent.parent
load_dotenv(project_root / '.env')


class EmbeddingSimilarityMatcher:
    """Scores paragraph-caption similarity with OpenAI-compatible embeddings."""

    def __init__(self):
        self.api_key = os.getenv('EMBEDDING_API_KEY') or os.getenv('API_KEY', '')
        self.base_url = self._resolve_base_url()
        self.model = os.getenv('EMBEDDING_MODEL', '').strip()
        self.timeout = float(os.getenv('EMBEDDING_TIMEOUT', '60'))
        self.batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE', '16'))
        self.similarity_threshold = float(os.getenv('EMBEDDING_SIMILARITY_THRESHOLD', '0.18'))
        self.cache_path = Path(
            os.getenv('EMBEDDING_CACHE_PATH', str(project_root / 'data' / 'embedding_cache.json'))
        )
        self.cache = self._load_cache()
        self.client = None

        if self.api_key and self.model:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )

    def _resolve_base_url(self) -> str:
        """Resolve a usable embeddings base URL from environment variables."""
        configured_base_url = os.getenv('EMBEDDING_BASE_URL', '').strip()
        if configured_base_url:
            return configured_base_url

        llm_base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1').strip()
        if '/api/coding/v3' in llm_base_url:
            return llm_base_url.replace('/api/coding/v3', '/api/v3')
        return llm_base_url

    def _load_cache(self) -> dict[str, list[float]]:
        """Load cached embeddings from disk."""
        if not self.cache_path.exists():
            return {}

        try:
            with open(self.cache_path, 'r', encoding='utf-8') as cache_file:
                cached = json.load(cache_file)
        except Exception:
            return {}

        if not isinstance(cached, dict):
            return {}
        return cached

    def _save_cache(self) -> None:
        """Persist cached embeddings to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, 'w', encoding='utf-8') as cache_file:
            json.dump(self.cache, cache_file, ensure_ascii=False)

    def _cache_key(self, text: str) -> str:
        """Build a stable cache key for one text under the current model."""
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        return f"{self.base_url}|{self.model}|{text_hash}"

    def is_configured(self) -> bool:
        """Return whether embeddings are configured and callable."""
        return self.client is not None

    def embed_texts(self, texts: list[str]) -> Optional[list[list[float]]]:
        """Embed texts with cache and OpenAI-compatible batching."""
        if not self.is_configured():
            return None

        normalized_texts = [text.strip() for text in texts]
        ordered_embeddings: list[Optional[list[float]]] = [None] * len(normalized_texts)
        uncached_indices: list[int] = []
        uncached_inputs: list[str] = []

        for index, text in enumerate(normalized_texts):
            if not text:
                ordered_embeddings[index] = []
                continue

            cache_key = self._cache_key(text)
            cached_embedding = self.cache.get(cache_key)
            if cached_embedding is not None:
                ordered_embeddings[index] = cached_embedding
                continue

            uncached_indices.append(index)
            uncached_inputs.append(text)

        try:
            for batch_start in range(0, len(uncached_inputs), self.batch_size):
                batch_inputs = uncached_inputs[batch_start:batch_start + self.batch_size]
                batch_indices = uncached_indices[batch_start:batch_start + self.batch_size]
                response = self.client.embeddings.create(model=self.model, input=batch_inputs)
                for batch_index, embedding_data in enumerate(response.data):
                    input_text = batch_inputs[batch_index]
                    cache_key = self._cache_key(input_text)
                    embedding = embedding_data.embedding
                    self.cache[cache_key] = embedding
                    ordered_embeddings[batch_indices[batch_index]] = embedding
        except Exception as exc:
            print(f"Embedding similarity request failed: {exc}")
            return None

        if uncached_inputs:
            self._save_cache()

        return [embedding or [] for embedding in ordered_embeddings]

    def similarity_matrix(self, left_texts: list[str], right_texts: list[str]) -> Optional[list[list[float]]]:
        """Compute a cosine-similarity matrix for two text lists."""
        if not left_texts or not right_texts:
            return []

        left_embeddings = self.embed_texts(left_texts)
        right_embeddings = self.embed_texts(right_texts)
        if left_embeddings is None or right_embeddings is None:
            return None

        normalized_left = [self._normalize_vector(vector) for vector in left_embeddings]
        normalized_right = [self._normalize_vector(vector) for vector in right_embeddings]

        matrix = []
        for left_vector in normalized_left:
            row = []
            for right_vector in normalized_right:
                if not left_vector or not right_vector:
                    row.append(0.0)
                    continue
                row.append(sum(left * right for left, right in zip(left_vector, right_vector)))
            matrix.append(row)

        return matrix

    def _normalize_vector(self, vector: list[float]) -> list[float]:
        """Normalize one embedding vector for cosine similarity."""
        if not vector:
            return []

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]
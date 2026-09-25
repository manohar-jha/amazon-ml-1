"""MinHash Locality Sensitive Hashing (LSH) for character 3-gram approximate matching.

Provides pure Python / NumPy MinHash signatures and band indexing
to rapidly find approximate name matches in sublinear time without full-table scans.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


def extract_character_ngrams(text: str, n: int = 3) -> List[str]:
    """Extract character n-grams from normalized text with boundary padding."""
    if not text:
        return []
    padded = f"^{text.strip()}$"
    if len(padded) < n:
        return [padded]
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]


class MinHashLSH:
    """Locality Sensitive Hashing using MinHash signatures for character 3-grams."""

    def __init__(
        self,
        num_permutations: int = 32,
        num_bands: int = 8,
        shingle_n: int = 3,
        seed: int = 42,
    ):
        """Initialize MinHash parameters and random hash coefficient tables.

        Args:
            num_permutations: Total number of hash functions (signatures length).
            num_bands: Number of bands (bands * rows_per_band = num_permutations).
            shingle_n: Character n-gram size.
            seed: Random seed for hash permutations.
        """
        if num_permutations % num_bands != 0:
            raise ValueError(f"num_permutations ({num_permutations}) must be divisible by num_bands ({num_bands})")

        self.num_permutations = num_permutations
        self.num_bands = num_bands
        self.rows_per_band = num_permutations // num_bands
        self.shingle_n = shingle_n

        # Large prime for 32-bit universal hashing
        self._prime = 4294967311

        rng = np.random.RandomState(seed)
        self._a = rng.randint(1, self._prime - 1, size=num_permutations, dtype=np.uint64)
        self._b = rng.randint(0, self._prime - 1, size=num_permutations, dtype=np.uint64)

        # Inverted band index: (country, band_idx, band_bucket_hash) -> list of entity IDs
        self.band_buckets: Dict[Tuple[str, int, int], List[str]] = defaultdict(list)
        # Global fallback index for unseen countries / soft country matching
        self.global_band_buckets: Dict[Tuple[int, int], List[str]] = defaultdict(list)

    def compute_signature(self, text: str) -> Optional[np.ndarray]:
        """Compute the MinHash signature array of length num_permutations for a text string."""
        ngrams = extract_character_ngrams(text, n=self.shingle_n)
        if not ngrams:
            return None

        # Hash shingles to 32-bit integers
        shingle_hashes = np.array([hash(g) & 0xFFFFFFFF for g in set(ngrams)], dtype=np.uint64)
        if len(shingle_hashes) == 0:
            return None

        # Vectorized minhash computation: (num_permutations, num_shingles)
        # h_i(x) = (a_i * x + b_i) % prime
        hashes_matrix = (
            self._a[:, None] * shingle_hashes[None, :] + self._b[:, None]
        ) % self._prime

        signature = np.min(hashes_matrix, axis=1).astype(np.uint32)
        return signature

    def index_entity(
        self,
        entity_id: str,
        text: str,
        country: str,
        max_bucket_size: int = 100,
    ) -> None:
        """Compute signature and insert into band buckets for both country and global lookup."""
        sig = self.compute_signature(text)
        if sig is None:
            return

        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_tuple = tuple(sig[start:end])
            band_hash = hash(band_tuple) & 0xFFFFFFFF

            country_key = (country, band_idx, band_hash)
            if len(self.band_buckets[country_key]) < max_bucket_size:
                self.band_buckets[country_key].append(entity_id)

            global_key = (band_idx, band_hash)
            if len(self.global_band_buckets[global_key]) < max_bucket_size:
                self.global_band_buckets[global_key].append(entity_id)

    def query_candidates(
        self,
        text: str,
        country: Optional[str] = None,
        max_candidates: int = 50,
        soft_country_fallback: bool = True,
    ) -> Set[str]:
        """Query matching candidate entity IDs sharing at least one band bucket."""
        sig = self.compute_signature(text)
        if sig is None:
            return set()

        candidates: Set[str] = set()

        # 1. Query within country
        if country:
            for band_idx in range(self.num_bands):
                start = band_idx * self.rows_per_band
                end = start + self.rows_per_band
                band_tuple = tuple(sig[start:end])
                band_hash = hash(band_tuple) & 0xFFFFFFFF
                key = (country, band_idx, band_hash)
                bucket = self.band_buckets.get(key, [])
                candidates.update(bucket)
                if len(candidates) >= max_candidates:
                    break

        # 2. Soft country fallback if candidates are insufficient or country is unseen
        if soft_country_fallback and len(candidates) < 5:
            for band_idx in range(self.num_bands):
                start = band_idx * self.rows_per_band
                end = start + self.rows_per_band
                band_tuple = tuple(sig[start:end])
                band_hash = hash(band_tuple) & 0xFFFFFFFF
                bucket = self.global_band_buckets.get((band_idx, band_hash), [])
                candidates.update(bucket)
                if len(candidates) >= max_candidates:
                    break

        if len(candidates) > max_candidates:
            return set(sorted(candidates)[:max_candidates])
        return candidates

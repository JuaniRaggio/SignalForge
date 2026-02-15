"""Document deduplication functionality."""

import hashlib
import re

import structlog

logger = structlog.get_logger(__name__)


class DocumentDeduplicator:
    """Detect and handle duplicate documents."""

    def __init__(self, similarity_threshold: float = 0.95) -> None:
        """
        Initialize deduplicator.

        Args:
            similarity_threshold: Minimum similarity score (0-1) for near-duplicates
        """
        self.similarity_threshold = similarity_threshold

    def compute_hash(self, content: str) -> str:
        """
        Compute content hash for exact duplicate detection.

        Args:
            content: Document content

        Returns:
            SHA-256 hash of normalized content
        """
        # Normalize content before hashing
        normalized = self._normalize_content(content)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def is_duplicate(self, content_hash: str, existing_hashes: set[str]) -> bool:
        """
        Check if document is exact duplicate.

        Args:
            content_hash: Hash of current document
            existing_hashes: Set of existing document hashes

        Returns:
            True if hash exists in set
        """
        return content_hash in existing_hashes

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two documents using Jaccard similarity on word n-grams.

        Args:
            text1: First document text
            text2: Second document text

        Returns:
            Similarity score between 0 and 1
        """
        # Generate word trigrams for both documents
        trigrams1 = self._generate_ngrams(text1, n=3)
        trigrams2 = self._generate_ngrams(text2, n=3)

        if not trigrams1 or not trigrams2:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(trigrams1 & trigrams2)
        union = len(trigrams1 | trigrams2)

        if union == 0:
            return 0.0

        similarity = intersection / union

        logger.debug(
            "computed_similarity",
            similarity=similarity,
            intersection=intersection,
            union=union,
        )

        return similarity

    def find_similar(
        self,
        content: str,
        candidates: list[tuple[str, str]],  # (doc_id, content)
    ) -> list[tuple[str, float]]:
        """
        Find similar documents above threshold.

        Args:
            content: Content to compare against
            candidates: List of (document_id, content) tuples to check

        Returns:
            List of (document_id, similarity_score) for matches above threshold
        """
        similar_docs: list[tuple[str, float]] = []

        for doc_id, candidate_content in candidates:
            similarity = self.compute_similarity(content, candidate_content)

            if similarity >= self.similarity_threshold:
                similar_docs.append((doc_id, similarity))
                logger.info(
                    "found_similar_document",
                    document_id=doc_id,
                    similarity=similarity,
                )

        # Sort by similarity descending
        similar_docs.sort(key=lambda x: -x[1])

        return similar_docs

    def _normalize_content(self, content: str) -> str:
        """
        Normalize content for consistent hashing.

        Args:
            content: Raw content

        Returns:
            Normalized content
        """
        # Convert to lowercase
        normalized = content.lower()

        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", normalized)

        # Remove common punctuation that doesn't affect meaning
        normalized = re.sub(r"[.,;:!?()[\]{}\"]", "", normalized)

        # Strip leading/trailing whitespace
        normalized = normalized.strip()

        return normalized

    def _generate_ngrams(self, text: str, n: int = 3) -> set[tuple[str, ...]]:
        """
        Generate word n-grams from text.

        Args:
            text: Input text
            n: N-gram size (default 3)

        Returns:
            Set of n-gram tuples
        """
        # Tokenize into words
        words = self._tokenize(text)

        if len(words) < n:
            # If text is too short, return word set
            return {tuple(words)}

        # Generate n-grams
        ngrams = set()
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i : i + n])
            ngrams.add(ngram)

        return ngrams

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Normalize
        text = text.lower()

        # Split on whitespace and punctuation
        tokens = re.findall(r"\b\w+\b", text)

        return tokens

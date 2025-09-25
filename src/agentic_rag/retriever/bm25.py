"""
BM25 keyword-based retrieval implementation for hybrid search.

This module provides BM25 scoring functionality to complement vector-based retrieval,
improving performance on queries with specific terms, names, or acronyms.
"""

import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


class BM25Retriever:
    """
    BM25 retriever for keyword-based document ranking.

    BM25 is particularly effective for:
    - Queries with specific terms that must match exactly
    - Named entities (people, places, organizations)
    - Acronyms and technical terms
    - Queries where term frequency matters
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """
        Initialize BM25 retriever.

        Args:
            k1: Term frequency saturation parameter (typically 1.2-2.0)
            b: Length normalization parameter (typically 0.75)
        """
        self.k1 = k1
        self.b = b
        # List of token lists; each entry is the tokenized document
        self.corpus: List[List[str]] = []
        self.doc_ids: List[str] = []
        self.doc_freqs: Dict[str, int] = {}
        self.idf_cache: Dict[str, float] = {}
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.corpus_size: int = 0
        self.stop_words = set(stopwords.words("english"))
        # Schema/version tracking for safe loading
        self.schema_version: int = 2  # current schema stores token lists in corpus

    def build_index(self, documents: List[Dict[str, str]]) -> None:
        """
        Build BM25 index from documents.

        Args:
            documents: List of dicts with 'id' and 'text' keys
        """
        self.corpus = []
        self.doc_ids = []
        self.doc_lengths = []

        # Process documents
        for doc in documents:
            doc_id = doc["id"]
            text = doc["text"]

            # Tokenize and filter
            tokens = self._tokenize(text)
            self.corpus.append(tokens) # Store tokens as a list
            self.doc_ids.append(doc_id)
            self.doc_lengths.append(len(tokens))

        self.corpus_size = len(self.corpus)
        self.avg_doc_length = sum(self.doc_lengths) / max(1, self.corpus_size)

        # Build document frequency counts
        self.doc_freqs = defaultdict(int)
        for doc_tokens_list in self.corpus: # Iterate over the list of tokens
            for token in set(doc_tokens_list): # Use the already tokenized list
                self.doc_freqs[token] += 1

        # Pre-compute IDF values for common terms
        self._compute_idf_cache()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text and remove stopwords."""
        try:
            raw_tokens = word_tokenize(text.lower())
            # Filter out stopwords and non-alphabetic tokens
            tokens = [
                token
                for token in raw_tokens
                if (token.isascii() and any(c.isalnum() for c in token)) and token not in self.stop_words and len(token) > 1
            ]
            return tokens
        except Exception:
            # Fallback tokenization
            tokens = [
                word.lower()
                for word in text.split()
                if (word.isascii() and any(c.isalnum() for c in word)) and len(word) > 1
                and word.lower() not in self.stop_words
            ]
            return tokens

    def _compute_idf_cache(self) -> None:
        """Pre-compute IDF values for frequent terms."""
        self.idf_cache = {}
        for term, doc_freq in self.doc_freqs.items():
            if doc_freq > 0:
                self.idf_cache[term] = math.log(
                    (self.corpus_size - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0
                )

    def _get_idf(self, term: str) -> float:
        """Get IDF score for a term."""
        if term in self.idf_cache:
            return self.idf_cache[term]

        doc_freq = self.doc_freqs.get(term, 0)
        if doc_freq == 0:
            return math.log(self.corpus_size + 1.0)  # Unseen term

        idf = math.log((self.corpus_size - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
        self.idf_cache[term] = idf
        return idf

    def _score_document(self, query_terms: List[str], doc_idx: int) -> float:
        """Calculate BM25 score for a document."""
        if doc_idx >= len(self.corpus):
            return 0.0

        doc_tokens = self.corpus[doc_idx]
        doc_length = self.doc_lengths[doc_idx]

        # Term frequency in document
        term_freqs = Counter(doc_tokens)

        score = 0.0
        for term in query_terms:
            if term in term_freqs:
                tf = term_freqs[term]
                idf = self._get_idf(term)

                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * (doc_length / self.avg_doc_length)
                )

                score += idf * (numerator / denominator)
        return score

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for top-k documents using BM25.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of (doc_id, score) tuples, sorted by score descending
        """
        if not self.corpus:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        # Score all documents
        scores = []
        for doc_idx in range(self.corpus_size):
            score = self._score_document(query_terms, doc_idx)
            scores.append((self.doc_ids[doc_idx], score))

        # Sort by score and return top-k (include zeros for diagnostic/recall)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def save_index(self, index_path: str) -> None:
        """Save BM25 index to disk."""
        index_data = {
            "k1": self.k1,
            "b": self.b,
            "corpus": self.corpus,
            "doc_ids": self.doc_ids,
            "doc_freqs": dict(self.doc_freqs),
            "idf_cache": self.idf_cache,
            "doc_lengths": self.doc_lengths,
            "avg_doc_length": self.avg_doc_length,
            "corpus_size": self.corpus_size,
            "schema_version": self.schema_version,
        }

        with open(index_path, "wb") as f:
            pickle.dump(index_data, f)

    def load_index(self, index_path: str) -> None:
        """Load BM25 index from disk."""
        with open(index_path, "rb") as f:
            index_data = pickle.load(f)

        self.k1 = index_data["k1"]
        self.b = index_data["b"]
        self.corpus = index_data["corpus"]
        self.doc_ids = index_data["doc_ids"]
        self.doc_freqs = defaultdict(int, index_data["doc_freqs"])
        self.idf_cache = index_data["idf_cache"]
        self.doc_lengths = index_data["doc_lengths"]
        self.avg_doc_length = index_data["avg_doc_length"]
        self.corpus_size = index_data["corpus_size"]
        self.schema_version = int(index_data.get("schema_version", 1))

    def is_schema_compatible(self) -> bool:
        """Return True if the loaded index matches current expectations.

        Current schema requires:
        - schema_version >= 2
        - corpus is a list of token lists (first element is a list)
        - lengths consistent
        """
        try:
            if int(self.schema_version) < 2:
                return False
            if not isinstance(self.corpus, list):
                return False
            if len(self.corpus) == 0:
                return True  # empty but OK
            if not isinstance(self.corpus[0], list):
                return False
            if len(self.doc_ids) != len(self.corpus) or len(self.doc_lengths) != len(self.corpus):
                return False
            return True
        except Exception:
            return False


class HybridRetriever:
    """
    Hybrid retriever combining vector search (FAISS) with keyword search (BM25).

    This retriever uses both dense (vector) and sparse (BM25) retrieval methods,
    combining their results to improve overall retrieval quality.
    """

    def __init__(self, vector_retriever, bm25_retriever, alpha: float = 0.7):
        """
        Initialize hybrid retriever.

        Args:
            vector_retriever: FAISS-based vector retriever
            bm25_retriever: BM25-based keyword retriever
            alpha: Weight for vector scores (1-alpha for BM25 scores)
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha  # Weight for vector search vs BM25

    def search_hybrid(
        self, query: str, k: int = 10, vector_k: int = None, bm25_k: int = None
    ) -> List[Tuple[str, float, str]]:
        """
        Perform hybrid search combining vector and BM25 results.

        Args:
            query: Search query
            k: Final number of results to return
            vector_k: Number of vector results to retrieve (default: 2*k)
            bm25_k: Number of BM25 results to retrieve (default: 2*k)

        Returns:
            List of (doc_id, combined_score, method) tuples
        """
        if vector_k is None:
            vector_k = min(k * 2, 50)
        if bm25_k is None:
            bm25_k = min(k * 2, 50)

        # Get results from both retrievers
        vector_results = self._get_vector_results(query, vector_k)
        bm25_results = self.bm25_retriever.search(query, bm25_k)

        # Normalize scores to [0,1] range
        vector_scores = self._normalize_scores([score for _, score in vector_results])
        bm25_scores = self._normalize_scores([score for _, score in bm25_results])

        # Combine results with score fusion
        combined_scores = {}

        # Add vector results
        for i, (doc_id, _) in enumerate(vector_results):
            combined_scores[doc_id] = {
                "vector_score": vector_scores[i],
                "bm25_score": 0.0,
                "method": "vector",
            }

        # Add/merge BM25 results
        for i, (doc_id, _) in enumerate(bm25_results):
            if doc_id in combined_scores:
                combined_scores[doc_id]["bm25_score"] = bm25_scores[i]
                combined_scores[doc_id]["method"] = "hybrid"
            else:
                combined_scores[doc_id] = {
                    "vector_score": 0.0,
                    "bm25_score": bm25_scores[i],
                    "method": "bm25",
                }

        # Calculate combined scores and sort
        results = []
        for doc_id, scores in combined_scores.items():
            combined_score = self.alpha * float(str(scores["vector_score"])) + (
                1 - self.alpha
            ) * float(str(scores["bm25_score"]))
            results.append((doc_id, combined_score, scores["method"]))

        results.sort(key=lambda x: x[1], reverse=True)
        return [(doc_id, score, str(method)) for doc_id, score, method in results[:k]]

    def _get_vector_results(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Get results from vector retriever."""
        try:
            # Use the existing vector retriever's search method
            if hasattr(self.vector_retriever, "retrieve"):
                chunks = self.vector_retriever.retrieve(query, k)
                return [
                    (chunk["id"].split("__")[0], chunk["score"]) for chunk in chunks
                ]
            else:
                return []
        except Exception:
            return []

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0,1] range using min-max normalization."""
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [(score - min_score) / (max_score - min_score) for score in scores]


def create_bm25_index(faiss_dir: str, output_path: str = None) -> BM25Retriever:
    """
    Create BM25 index from FAISS corpus data.

    Args:
        faiss_dir: Directory containing chunks.parquet
        output_path: Path to save BM25 index (optional)

    Returns:
        Initialized BM25Retriever
    """
    # Load chunks data
    chunks_path = Path(faiss_dir) / "chunks.parquet"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    chunks_df = pd.read_parquet(chunks_path)

    # Convert to documents format
    documents = []
    for _, row in chunks_df.iterrows():
        documents.append({"id": row["id"], "text": row["text"]})

    # Build BM25 index
    bm25 = BM25Retriever()
    bm25.build_index(documents)

    # Save index if path provided
    if output_path:
        bm25.save_index(output_path)

    return bm25


def load_bm25_index(index_path: str) -> BM25Retriever:
    """Load pre-built BM25 index from disk."""
    bm25 = BM25Retriever()
    bm25.load_index(index_path)
    return bm25

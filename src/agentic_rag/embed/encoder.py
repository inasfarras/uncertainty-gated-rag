"""Text embedding encoder using sentence-transformers or OpenAI."""

from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
from pydantic import BaseModel


class EmbeddingResult(BaseModel):
    """Result from embedding operation."""

    embeddings: List[List[float]]
    model: str
    dimensions: int
    usage: Optional[dict] = None


class BaseEmbeddingEncoder(ABC):
    """Abstract base class for embedding encoders."""

    def __init__(self, model_name: str, **kwargs) -> None:
        """
        Initialize the embedding encoder.

        Args:
            model_name: Name of the embedding model
            **kwargs: Additional configuration parameters
        """
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
        **kwargs,
    ) -> EmbeddingResult:
        """
        Encode texts into embeddings.

        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            **kwargs: Additional encoding parameters

        Returns:
            Embedding result with vectors and metadata
        """
        pass

    @abstractmethod
    async def encode_async(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
        **kwargs,
    ) -> EmbeddingResult:
        """
        Asynchronously encode texts into embeddings.

        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            **kwargs: Additional encoding parameters

        Returns:
            Embedding result with vectors and metadata
        """
        pass

    @abstractmethod
    def get_dimensions(self) -> int:
        """
        Get the dimensionality of the embeddings.

        Returns:
            Number of dimensions in the embedding vectors
        """
        pass


class SentenceTransformerEncoder(BaseEmbeddingEncoder):
    """Encoder using sentence-transformers models."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize sentence-transformers encoder.

        Args:
            model_name: HuggingFace model name
            device: Device to run model on
            **kwargs: Additional configuration
        """
        super().__init__(model_name, **kwargs)
        self.device = device
        self._model = None
        self._dimensions = None

    def _load_model(self):
        """Load the sentence-transformers model."""
        # TODO: Implement model loading
        pass

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
        **kwargs,
    ) -> EmbeddingResult:
        """Encode texts using sentence-transformers."""
        # TODO: Implement encoding
        raise NotImplementedError("SentenceTransformer encoding not yet implemented")

    async def encode_async(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
        **kwargs,
    ) -> EmbeddingResult:
        """Asynchronously encode texts."""
        # TODO: Implement async encoding
        raise NotImplementedError("Async encoding not yet implemented")

    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        # TODO: Implement dimension retrieval
        raise NotImplementedError("Dimension retrieval not yet implemented")


class OpenAIEmbeddingEncoder(BaseEmbeddingEncoder):
    """Encoder using OpenAI embedding models."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize OpenAI embedding encoder.

        Args:
            model_name: OpenAI embedding model name
            api_key: OpenAI API key
            **kwargs: Additional configuration
        """
        super().__init__(model_name, **kwargs)
        self.api_key = api_key

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
        **kwargs,
    ) -> EmbeddingResult:
        """Encode texts using OpenAI API."""
        # TODO: Implement OpenAI encoding
        raise NotImplementedError("OpenAI encoding not yet implemented")

    async def encode_async(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
        **kwargs,
    ) -> EmbeddingResult:
        """Asynchronously encode texts using OpenAI API."""
        # TODO: Implement async OpenAI encoding
        raise NotImplementedError("Async OpenAI encoding not yet implemented")

    def get_dimensions(self) -> int:
        """Get embedding dimensions for OpenAI models."""
        # TODO: Implement dimension mapping
        model_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return model_dims.get(self.model_name, 1536)


def create_encoder(
    model_name: str,
    provider: Optional[str] = None,
    **kwargs,
) -> BaseEmbeddingEncoder:
    """
    Factory function to create embedding encoders.

    Args:
        model_name: Name of the embedding model
        provider: Embedding provider ("sentence-transformers", "openai")
        **kwargs: Additional configuration parameters

    Returns:
        Configured embedding encoder instance

    Raises:
        ValueError: If provider cannot be determined or is unsupported
    """
    # Auto-detect provider if not specified
    if provider is None:
        if model_name.startswith("text-embedding"):
            provider = "openai"
        else:
            provider = "sentence-transformers"

    if provider == "sentence-transformers":
        return SentenceTransformerEncoder(model_name, **kwargs)
    elif provider == "openai":
        return OpenAIEmbeddingEncoder(model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def batch_cosine_similarity(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Calculate cosine similarity between query and multiple documents.

    Args:
        query_embedding: Query embedding vector
        doc_embeddings: Document embedding matrix (n_docs x embedding_dim)

    Returns:
        Array of similarity scores
    """
    # Normalize embeddings
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

    # Calculate similarities
    similarities = np.dot(doc_norms, query_norm)
    return similarities

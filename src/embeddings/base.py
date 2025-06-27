"""Base embedding interface."""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class BaseEmbedding(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def encode(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Encode text into embedding vector(s).
        
        Args:
            text: Single text string or list of text strings
            
        Returns:
            Single embedding vector or list of embedding vectors
        """
        pass
    
    @abstractmethod
    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Encode a batch of texts into embedding vectors.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        pass
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class EmbeddingFactory:
    """Factory for creating embedding instances."""
    
    _registry = {}
    
    @classmethod
    def register(cls, name: str, embedding_class):
        """Register an embedding class."""
        cls._registry[name] = embedding_class
    
    @classmethod
    def create(cls, embedding_type: str, **kwargs) -> BaseEmbedding:
        """
        Create an embedding instance.
        
        Args:
            embedding_type: Type of embedding to create
            **kwargs: Additional arguments for the embedding
            
        Returns:
            Embedding instance
            
        Raises:
            ValueError: If embedding type is not registered
        """
        if embedding_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown embedding type: {embedding_type}. "
                f"Available types: {available}"
            )
        
        return cls._registry[embedding_type](**kwargs)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List available embedding types."""
        return list(cls._registry.keys())

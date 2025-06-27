"""Base storage interface for document storage and retrieval."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class Document:
    """Document with content, metadata, and embedding."""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    id: Optional[str] = None


class BaseStorage(ABC):
    """Abstract base class for document storage."""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to storage.
        
        Args:
            documents: List of documents to add
        """
        pass
    
    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document if found, None otherwise
        """
        pass
    
    @abstractmethod
    def search_by_embedding(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Document]:
        """
        Search documents by embedding similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of most similar documents
        """
        pass
    
    @abstractmethod
    def get_all_documents(self) -> List[Document]:
        """
        Get all documents from storage.
        
        Returns:
            List of all documents
        """
        pass
    
    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    def clear_all(self) -> None:
        """Clear all documents from storage."""
        pass
    
    @abstractmethod
    def count_documents(self) -> int:
        """
        Get the total number of documents in storage.
        
        Returns:
            Number of documents
        """
        pass
    
    @abstractmethod
    def get_metadata_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored documents.
        
        Returns:
            Dictionary with storage statistics
        """
        pass


class StorageFactory:
    """Factory for creating storage instances."""
    
    _registry = {}
    
    @classmethod
    def register(cls, name: str, storage_class):
        """Register a storage class."""
        cls._registry[name] = storage_class
    
    @classmethod
    def create(cls, storage_type: str, **kwargs) -> BaseStorage:
        """
        Create a storage instance.
        
        Args:
            storage_type: Type of storage to create
            **kwargs: Additional arguments for the storage
            
        Returns:
            Storage instance
            
        Raises:
            ValueError: If storage type is not registered
        """
        if storage_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown storage type: {storage_type}. "
                f"Available types: {available}"
            )
        
        return cls._registry[storage_type](**kwargs)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List available storage types."""
        return list(cls._registry.keys())

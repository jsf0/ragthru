# src/__init__.py
"""RAG Proxy Server - Universal RAG middleware for chat applications."""

__version__ = "1.0.0"
__author__ = "jsf0"
__description__ = "Universal RAG middleware proxy server"

# src/embeddings/__init__.py
"""Embedding models for RAG."""

from .base import BaseEmbedding, EmbeddingFactory
from .simple import SimpleEmbedding

# Ensure the simple embedding is registered
import src.embeddings.simple

__all__ = ["BaseEmbedding", "EmbeddingFactory", "SimpleEmbedding"]

# src/storage/__init__.py
"""Storage backends for RAG."""

from .base import BaseStorage, Document, StorageFactory
from .sqlite import SQLiteStorage

# Ensure the SQLite storage is registered
import src.storage.sqlite

__all__ = ["BaseStorage", "Document", "StorageFactory", "SQLiteStorage"]

# src/rag/__init__.py
"""RAG (Retrieval-Augmented Generation) components."""

from .index import RAGIndex

__all__ = ["RAGIndex"]

# src/proxy/__init__.py
"""Proxy server components."""

from .server import RAGProxyServer
from .handlers import ProxyHandlers

__all__ = ["RAGProxyServer", "ProxyHandlers"]

# src/utils/__init__.py
"""Utility functions and classes."""

from .text_processing import TextProcessor, DocumentPreprocessor
from .logging import setup_logging, get_logger

__all__ = ["TextProcessor", "DocumentPreprocessor", "setup_logging", "get_logger"]

"""RAG indexing logic for document processing and storage."""

from typing import List, Dict, Any
from pathlib import Path
import logging

from ..embeddings.base import BaseEmbedding, EmbeddingFactory
from ..storage.base import BaseStorage, Document, StorageFactory
from ..utils.text_processing import TextProcessor
from ..config import config

logger = logging.getLogger(__name__)


class RAGIndex:
    """RAG index for managing documents and embeddings."""
    
    def __init__(
        self,
        storage: BaseStorage = None,
        embedding: BaseEmbedding = None,
        text_processor: TextProcessor = None
    ):
        """
        Initialize RAG index.
        
        Args:
            storage: Storage backend for documents
            embedding: Embedding model for vectorization
            text_processor: Text processing utilities
        """
        self.storage = storage or StorageFactory.create(
            "sqlite",
            db_path=config.rag_db_path
        )
        
        self.embedding = embedding or EmbeddingFactory.create(
            config.rag_embedding_type
        )
        
        self.text_processor = text_processor or TextProcessor(
            chunk_size=config.rag_chunk_size,
            chunk_overlap=config.rag_chunk_overlap
        )
        
        logger.info(f"RAG Index initialized with {type(self.storage).__name__} storage "
                   f"and {type(self.embedding).__name__} embeddings")
    
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> None:
        """
        Add documents to the index.
        
        Args:
            texts: List of document texts
            metadatas: Optional list of metadata for each document
        """
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        if len(texts) != len(metadatas):
            raise ValueError("Number of texts and metadatas must match")
        
        # Process all chunks first for embedding fitting
        all_chunks = []
        chunk_documents = []
        
        for text, metadata in zip(texts, metadatas):
            chunks = self.text_processor.chunk_text(text)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                
                # Create chunk metadata
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk)
                })
                
                chunk_documents.append(Document(
                    content=chunk,
                    metadata=chunk_metadata
                ))
        
        # Fit embedding model if it supports fitting
        if hasattr(self.embedding, 'fit') and hasattr(self.embedding, 'fitted'):
            if not self.embedding.fitted and all_chunks:
                logger.info(f"Fitting embedding model on {len(all_chunks)} chunks")
                self.embedding.fit(all_chunks)
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunk_documents)} chunks")
        embeddings = self.embedding.encode_batch([doc.content for doc in chunk_documents])
        
        # Add embeddings to documents
        for doc, embedding in zip(chunk_documents, embeddings):
            doc.embedding = embedding
        
        # Store documents
        logger.info(f"Storing {len(chunk_documents)} documents")
        self.storage.add_documents(chunk_documents)
        
        logger.info(f"Successfully added {len(texts)} documents "
                   f"({len(chunk_documents)} chunks) to index")
    
    def add_files(self, file_paths: List[str]) -> None:
        """
        Add text files to the index.
        
        Args:
            file_paths: List of file paths to add (supports glob patterns)
        """
        import glob
        
        # Expand glob patterns
        expanded_paths = []
        for pattern in file_paths:
            if '*' in pattern or '?' in pattern:
                # Use glob to expand pattern
                matches = glob.glob(pattern)
                if matches:
                    expanded_paths.extend(matches)
                else:
                    logger.warning(f"No files found matching pattern: {pattern}")
            else:
                expanded_paths.append(pattern)
        
        texts = []
        metadatas = []
        
        for file_path in expanded_paths:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
            
            try:
                content = self._read_file(path)
                texts.append(content)
                metadatas.append({
                    'source': str(path),
                    'filename': path.name,
                    'file_size': path.stat().st_size,
                    'file_type': path.suffix.lower()
                })
                logger.info(f"Read file: {path.name} ({len(content)} characters)")
                
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
        
        if texts:
            self.add_documents(texts, metadatas)
        else:
            logger.warning("No files were successfully loaded")
    
    def _read_file(self, path: Path) -> str:
        """
        Read content from a file.
        
        Args:
            path: Path to the file
            
        Returns:
            File content as string
        """
        # Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, read as binary and decode with errors='ignore'
        with open(path, 'rb') as f:
            content = f.read()
            return content.decode('utf-8', errors='ignore')
    
    def search(self, query: str, top_k: int = None, threshold: float = None) -> List[Document]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of relevant documents
        """
        if top_k is None:
            top_k = config.rag_top_k
        
        if threshold is None:
            threshold = config.rag_similarity_threshold
        
        # Generate query embedding
        query_embedding = self.embedding.encode(query)
        
        # Search storage
        results = self.storage.search_by_embedding(
            query_embedding,
            top_k=top_k,
            threshold=threshold
        )
        
        logger.info(f"Found {len(results)} documents for query: {query[:100]}...")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Dictionary with index statistics
        """
        storage_stats = self.storage.get_metadata_stats()
        
        return {
            'storage': storage_stats,
            'embedding': {
                'type': type(self.embedding).__name__,
                'dimension': self.embedding.dimension,
            },
            'config': {
                'chunk_size': config.rag_chunk_size,
                'chunk_overlap': config.rag_chunk_overlap,
                'top_k': config.rag_top_k,
                'similarity_threshold': config.rag_similarity_threshold,
            }
        }
    
    def clear(self) -> None:
        """Clear all documents from the index."""
        self.storage.clear_all()
        logger.info("Cleared all documents from index")
    
    def count_documents(self) -> int:
        """
        Get the number of documents in the index.
        
        Returns:
            Number of documents
        """
        return self.storage.count_documents()

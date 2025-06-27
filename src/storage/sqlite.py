"""SQLite storage implementation."""

import sqlite3
import json
import pickle
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from .base import BaseStorage, Document, StorageFactory


class SQLiteStorage(BaseStorage):
    """SQLite-based document storage."""
    
    def __init__(self, db_path: str = "rag_index.db"):
        """
        Initialize SQLite storage.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize SQLite database."""
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index for faster searches
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_created_at 
            ON documents(created_at)
        ''')
        
        conn.commit()
        conn.close()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to storage.
        
        Args:
            documents: List of documents to add
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            for doc in documents:
                # Generate ID if not provided
                if not doc.id:
                    doc.id = str(uuid.uuid4())
                
                # Serialize data
                metadata_json = json.dumps(doc.metadata)
                embedding_blob = pickle.dumps(doc.embedding)
                
                cursor.execute('''
                    INSERT OR REPLACE INTO documents (id, content, metadata, embedding)
                    VALUES (?, ?, ?, ?)
                ''', (doc.id, doc.content, metadata_json, embedding_blob))
            
            conn.commit()
        finally:
            conn.close()
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document if found, None otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                'SELECT id, content, metadata, embedding FROM documents WHERE id = ?',
                (doc_id,)
            )
            result = cursor.fetchone()
            
            if result:
                doc_id, content, metadata_json, embedding_blob = result
                metadata = json.loads(metadata_json)
                embedding = pickle.loads(embedding_blob)
                
                return Document(
                    id=doc_id,
                    content=content,
                    metadata=metadata,
                    embedding=embedding
                )
            
            return None
        finally:
            conn.close()
    
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
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT id, content, metadata, embedding FROM documents')
            results = cursor.fetchall()
            
            if not results:
                return []
            
            # Calculate similarities
            similarities = []
            documents = []
            
            for doc_id, content, metadata_json, embedding_blob in results:
                embedding = pickle.loads(embedding_blob)
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, embedding)
                
                if similarity >= threshold:
                    metadata = json.loads(metadata_json)
                    similarities.append(similarity)
                    documents.append(Document(
                        id=doc_id,
                        content=content,
                        metadata=metadata,
                        embedding=embedding
                    ))
            
            # Sort by similarity and return top_k
            if similarities:
                sorted_indices = np.argsort(similarities)[::-1]
                return [documents[i] for i in sorted_indices[:top_k]]
            
            return []
        finally:
            conn.close()
    
    def get_all_documents(self) -> List[Document]:
        """
        Get all documents from storage.
        
        Returns:
            List of all documents
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT id, content, metadata, embedding FROM documents')
            results = cursor.fetchall()
            
            documents = []
            for doc_id, content, metadata_json, embedding_blob in results:
                metadata = json.loads(metadata_json)
                embedding = pickle.loads(embedding_blob)
                
                documents.append(Document(
                    id=doc_id,
                    content=content,
                    metadata=metadata,
                    embedding=embedding
                ))
            
            return documents
        finally:
            conn.close()
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if deleted, False if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM documents WHERE id = ?', (doc_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def clear_all(self) -> None:
        """Clear all documents from storage."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM documents')
            conn.commit()
        finally:
            conn.close()
    
    def count_documents(self) -> int:
        """
        Get the total number of documents in storage.
        
        Returns:
            Number of documents
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT COUNT(*) FROM documents')
            return cursor.fetchone()[0]
        finally:
            conn.close()
    
    def get_metadata_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored documents.
        
        Returns:
            Dictionary with storage statistics
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Basic stats
            cursor.execute('SELECT COUNT(*) FROM documents')
            total_docs = cursor.fetchone()[0]
            
            cursor.execute('SELECT SUM(LENGTH(content)) FROM documents')
            total_content_length = cursor.fetchone()[0] or 0
            
            # Get unique sources
            cursor.execute('SELECT metadata FROM documents')
            metadatas = cursor.fetchall()
            
            sources = set()
            total_chunks = 0
            for (metadata_json,) in metadatas:
                metadata = json.loads(metadata_json)
                if 'filename' in metadata:
                    sources.add(metadata['filename'])
                if 'chunk_id' in metadata:
                    total_chunks += 1
            
            return {
                'total_documents': total_docs,
                'total_content_length': total_content_length,
                'unique_sources': len(sources),
                'total_chunks': total_chunks,
                'average_content_length': total_content_length / max(total_docs, 1),
                'sources': list(sources)
            }
        finally:
            conn.close()
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


# Register the storage type
StorageFactory.register("sqlite", SQLiteStorage)

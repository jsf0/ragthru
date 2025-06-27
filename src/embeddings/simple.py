"""Simple TF-IDF based embeddings."""

import re
import math
import numpy as np
from typing import List, Dict, Union
from collections import Counter

from .base import BaseEmbedding, EmbeddingFactory


class SimpleEmbedding(BaseEmbedding):
    """Simple TF-IDF based embedding."""
    
    def __init__(self, max_features: int = 5000):
        """
        Initialize the simple embedding model.
        
        Args:
            max_features: Maximum number of features to use in vocabulary
        """
        self.max_features = max_features
        self.vocabulary = {}
        self.idf_scores = {}
        self.fitted = False
        self._dimension = 0
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        return self._dimension
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Convert to lowercase and extract words
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency."""
        tf = Counter(tokens)
        total_tokens = len(tokens)
        if total_tokens == 0:
            return {}
        return {word: count / total_tokens for word, count in tf.items()}
    
    def fit(self, documents: List[str]) -> None:
        """
        Fit the embedding model on documents.
        
        Args:
            documents: List of documents to fit on
        """
        if not documents:
            raise ValueError("Cannot fit on empty document list")
        
        all_tokens = []
        doc_tokens = []
        
        # Tokenize all documents
        for doc in documents:
            tokens = self._tokenize(doc)
            doc_tokens.append(tokens)
            all_tokens.extend(tokens)
        
        # Build vocabulary with most frequent terms
        token_counts = Counter(all_tokens)
        most_common = token_counts.most_common(self.max_features)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(most_common)}
        self._dimension = len(self.vocabulary)
        
        # Compute IDF scores
        num_docs = len(documents)
        word_doc_count = Counter()
        
        for tokens in doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                if token in self.vocabulary:
                    word_doc_count[token] += 1
        
        self.idf_scores = {
            word: math.log(num_docs / (count + 1))
            for word, count in word_doc_count.items()
        }
        
        self.fitted = True
    
    def encode(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Encode text into embedding vector(s).
        
        Args:
            text: Single text string or list of text strings
            
        Returns:
            Single embedding vector or list of embedding vectors
        """
        if isinstance(text, str):
            return self._encode_single(text)
        else:
            return self.encode_batch(text)
    
    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Encode a batch of texts into embedding vectors.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        return [self._encode_single(text) for text in texts]
    
    def _encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into embedding vector."""
        if not self.fitted:
            # Simple fallback - use character-based features
            return self._simple_encode(text)
        
        tokens = self._tokenize(text)
        tf_scores = self._compute_tf(tokens)
        
        # Create TF-IDF vector
        vector = np.zeros(len(self.vocabulary))
        
        for word, tf in tf_scores.items():
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                idf = self.idf_scores.get(word, 0)
                vector[idx] = tf * idf
        
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def _simple_encode(self, text: str) -> np.ndarray:
        """Simple character-based encoding as fallback."""
        # Create a simple feature vector based on text characteristics
        text_lower = text.lower()
        
        features = [
            len(text),  # length
            len(text.split()),  # word count
            text.count('.'),  # sentence count
            text.count('?'),  # question marks
            text.count('!'),  # exclamations
            len(set(text_lower.split())) / max(len(text.split()), 1),  # unique word ratio
        ]
        
        # Add character frequency features for common letters
        common_chars = 'abcdefghijklmnopqrstuvwxyz'
        text_len = max(len(text), 1)
        for char in common_chars:
            features.append(text_lower.count(char) / text_len)
        
        vector = np.array(features, dtype=float)
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        # Update dimension if not set
        if self._dimension == 0:
            self._dimension = len(vector)
        
        return vector


# Register the embedding type
EmbeddingFactory.register("simple", SimpleEmbedding)


# Try to register sentence transformers if available
try:
    from sentence_transformers import SentenceTransformer
    
    class SentenceTransformerEmbedding(BaseEmbedding):
        """Wrapper for sentence transformers."""
        
        def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
            """
            Initialize sentence transformer embedding.
            
            Args:
                model_name: Name of the sentence transformer model
            """
            self.model = SentenceTransformer(model_name)
            self._dimension = self.model.get_sentence_embedding_dimension()
        
        @property
        def dimension(self) -> int:
            """Return the dimension of the embedding vectors."""
            return self._dimension
        
        def encode(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
            """Encode text into embedding vector(s)."""
            result = self.model.encode(text)
            if isinstance(text, str):
                return result
            else:
                return list(result)
        
        def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
            """Encode a batch of texts into embedding vectors."""
            return list(self.model.encode(texts))
    
    # Register sentence transformer embeddings
    EmbeddingFactory.register("sentence-transformer", SentenceTransformerEmbedding)
    
    # Register specific models
    for model_name in [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "multi-qa-MiniLM-L6-cos-v1"
    ]:
        EmbeddingFactory.register(
            model_name,
            lambda name=model_name: SentenceTransformerEmbedding(name)
        )

except ImportError:
    # sentence-transformers not available
    pass

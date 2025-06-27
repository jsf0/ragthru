"""Text processing utilities for chunking and preprocessing."""

import re
from typing import List, Optional


class TextProcessor:
    """Text processing utilities for RAG."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize text processor.
        
        Args:
            chunk_size: Maximum size of text chunks
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at natural boundaries
            if end < len(text):
                # Find the best break point
                break_point = self._find_break_point(chunk, start)
                if break_point > start + self.chunk_size // 2:
                    chunk = text[start:break_point]
                    end = break_point
            
            # Clean and add chunk
            cleaned_chunk = self._clean_chunk(chunk)
            if cleaned_chunk:
                chunks.append(cleaned_chunk)
            
            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def _find_break_point(self, chunk: str, start_pos: int) -> int:
        """
        Find the best break point for a chunk.
        
        Args:
            chunk: Text chunk to find break point in
            start_pos: Starting position in the original text
            
        Returns:
            Best break point position
        """
        # Priority order for break points
        break_patterns = [
            r'\n\n',      # Paragraph breaks
            r'\n',        # Line breaks
            r'\. ',       # Sentence ends
            r'[.!?] ',    # Punctuation with space
            r', ',        # Comma breaks
            r' ',         # Word boundaries
        ]
        
        best_break = len(chunk)
        
        for pattern in break_patterns:
            matches = list(re.finditer(pattern, chunk))
            if matches:
                # Find the match closest to the end but not too close to start
                min_pos = len(chunk) // 3
                for match in reversed(matches):
                    if match.end() >= min_pos:
                        best_break = match.end()
                        break
                break
        
        return start_pos + best_break
    
    def _clean_chunk(self, chunk: str) -> str:
        """
        Clean a text chunk.
        
        Args:
            chunk: Raw text chunk
            
        Returns:
            Cleaned text chunk
        """
        # Strip whitespace
        chunk = chunk.strip()
        
        # Remove excessive whitespace
        chunk = re.sub(r'\s+', ' ', chunk)
        
        # Remove very short chunks
        if len(chunk) < 10:
            return ""
        
        return chunk
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before chunking.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text
        """
        # Normalize whitespace
        text = re.sub(r'\r\n', '\n', text)  # Windows line endings
        text = re.sub(r'\r', '\n', text)    # Mac line endings
        
        # Remove excessive blank lines
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # Clean up common formatting issues
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs
        text = re.sub(r'\n[ \t]+', '\n', text)  # Leading whitespace on lines
        
        return text.strip()
    
    def extract_metadata(self, text: str) -> dict:
        """
        Extract metadata from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'line_count': text.count('\n') + 1,
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
        }
        
        # Count sentences (rough estimate)
        sentence_endings = len(re.findall(r'[.!?]+', text))
        metadata['estimated_sentences'] = sentence_endings
        
        # Detect language patterns (basic)
        if re.search(r'[a-zA-Z]', text):
            metadata['has_latin_chars'] = True
        
        return metadata


class DocumentPreprocessor:
    """Document-specific preprocessing utilities."""
    
    @staticmethod
    def clean_markdown(text: str) -> str:
        """
        Clean markdown formatting from text.
        
        Args:
            text: Markdown text
            
        Returns:
            Plain text
        """
        # Remove headers
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        
        # Remove bold/italic
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        
        # Remove links
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
        
        # Remove code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`(.+?)`', r'\1', text)
        
        # Remove horizontal rules
        text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
        
        return text
    
    @staticmethod
    def extract_code_blocks(text: str) -> List[dict]:
        """
        Extract code blocks from markdown text.
        
        Args:
            text: Markdown text
            
        Returns:
            List of code blocks with metadata
        """
        code_blocks = []
        
        # Find fenced code blocks
        pattern = r'```(\w+)?\n(.*?)\n```'
        matches = re.finditer(pattern, text, flags=re.DOTALL)
        
        for match in matches:
            language = match.group(1) or 'text'
            code = match.group(2).strip()
            
            code_blocks.append({
                'language': language,
                'code': code,
                'start_pos': match.start(),
                'end_pos': match.end()
            })
        
        return code_blocks
    
    @staticmethod
    def clean_html(text: str) -> str:
        """
        Clean HTML tags from text.
        
        Args:
            text: HTML text
            
        Returns:
            Plain text
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Decode HTML entities
        html_entities = {
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'",
            '&nbsp;': ' ',
        }
        
        for entity, char in html_entities.items():
            text = text.replace(entity, char)
        
        return text

"""Request handlers for the RAG proxy server."""

import logging
from typing import List, Dict, Any
from flask import request, jsonify, Response
import requests

from ..rag.index import RAGIndex
from ..config import config

logger = logging.getLogger(__name__)


class ProxyHandlers:
    """Request handlers for the RAG proxy."""
    
    def __init__(self, rag_index: RAGIndex, backend_url: str, rag_enabled: bool = True):
        """
        Initialize proxy handlers.
        
        Args:
            rag_index: RAG index instance
            backend_url: Backend server URL
            rag_enabled: Whether RAG is initially enabled
        """
        self.rag_index = rag_index
        self.backend_url = backend_url.rstrip('/')
        self.rag_enabled = rag_enabled
    
    def handle_chat_completion(self):
        """Handle chat completion requests with RAG enhancement."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            
            model = data.get('model', 'unknown')
            messages = data.get("messages", [])
            
            logger.info(f"Received chat completion request for model: {model}")
            logger.info(f"Original messages: {len(messages)} messages")
            
            # Extract and log user query
            user_query = self._extract_user_query(messages)
            logger.info(f"User query: {user_query[:200]}...")
            
            # Enrich with RAG if enabled
            if self.rag_enabled:
                logger.info("RAG is enabled, enriching messages...")
                enriched_messages = self._enrich_with_rag(messages)
                data["messages"] = enriched_messages
                logger.info(f"Enriched messages: {len(enriched_messages)} messages")
            else:
                logger.info("RAG is disabled, passing through original messages")
            
            # Forward to backend
            return self._forward_to_backend(data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Backend request error: {e}")
            return jsonify({"error": f"Backend request failed: {str(e)}"}), 502
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return jsonify({"error": str(e)}), 500
    
    def handle_list_models(self):
        """Handle model listing requests."""
        try:
            logger.info("Handling models list request")
            return self._proxy_request('GET', '/v1/models')
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return jsonify({"error": str(e)}), 500
    
    def handle_health(self):
        """Handle health check requests."""
        return jsonify({
            "status": "healthy",
            "rag_enabled": self.rag_enabled,
            "backend_url": self.backend_url,
            "documents_indexed": self.rag_index.count_documents()
        })
    
    def handle_rag_status(self):
        """Handle RAG status requests."""
        try:
            stats = self.rag_index.get_stats()
            return jsonify({
                "rag_enabled": self.rag_enabled,
                "backend_url": self.backend_url,
                "stats": stats
            })
        except Exception as e:
            logger.error(f"Error getting RAG status: {e}")
            return jsonify({"error": str(e)}), 500
    
    def handle_rag_toggle(self):
        """Handle RAG enable/disable toggle."""
        try:
            data = request.get_json() or {}
            
            if 'enabled' in data:
                self.rag_enabled = bool(data['enabled'])
            else:
                self.rag_enabled = not self.rag_enabled
            
            logger.info(f"RAG toggled to: {'enabled' if self.rag_enabled else 'disabled'}")
            
            return jsonify({
                "rag_enabled": self.rag_enabled,
                "message": f"RAG {'enabled' if self.rag_enabled else 'disabled'}"
            })
        except Exception as e:
            logger.error(f"Error toggling RAG: {e}")
            return jsonify({"error": str(e)}), 500
    
    def handle_rag_stats(self):
        """Handle RAG statistics requests."""
        try:
            stats = self.rag_index.get_stats()
            return jsonify(stats)
        except Exception as e:
            logger.error(f"Error getting RAG stats: {e}")
            return jsonify({"error": str(e)}), 500
    
    def handle_root(self):
        """Handle root endpoint requests."""
        return jsonify({
            "service": "RAG Proxy Server",
            "version": "1.0.0",
            "status": "running",
            "rag_enabled": self.rag_enabled,
            "backend_url": self.backend_url,
            "endpoints": {
                "chat": "/v1/chat/completions",
                "models": "/v1/models",
                "health": "/health",
                "rag_status": "/rag/status",
                "rag_toggle": "/rag/toggle",
                "rag_stats": "/rag/stats"
            },
            "documents_indexed": self.rag_index.count_documents()
        })
    
    def handle_proxy_request(self, subpath):
        """Handle general proxy requests to v1 endpoints."""
        try:
            return self._proxy_request(request.method, f'/v1/{subpath}')
        except Exception as e:
            logger.error(f"Error proxying request to /v1/{subpath}: {e}")
            return jsonify({"error": str(e)}), 500
    
    def handle_not_found(self, error):
        """Handle 404 errors."""
        return jsonify({
            "error": "Endpoint not found",
            "available_endpoints": [
                "/v1/chat/completions",
                "/v1/models",
                "/health",
                "/rag/status",
                "/rag/toggle",
                "/rag/stats"
            ]
        }), 404
    
    def handle_internal_error(self, error):
        """Handle 500 errors."""
        logger.error(f"Internal server error: {error}")
        return jsonify({
            "error": "Internal server error",
            "message": str(error)
        }), 500
    
    def _extract_user_query(self, messages: List[Dict]) -> str:
        """Extract the main user query from messages."""
        # Get the last user message
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                
                # Handle different content formats
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # Handle array of content objects (like from LibreChat)
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif isinstance(item, str):
                            text_parts.append(item)
                    return " ".join(text_parts)
                else:
                    # Convert other types to string
                    return str(content)
        
        return ""
    
    def _should_use_rag(self, query: str) -> bool:
        """Determine if RAG should be used for this query."""
        if not self.rag_enabled:
            return False
        
        # Skip RAG for very short queries
        if not query or len(query.strip()) < 10:
            return False
        
        return True
    
    def _enrich_with_rag(self, messages: List[Dict], top_k: int = None) -> List[Dict]:
        """Enrich messages with RAG context."""
        if top_k is None:
            top_k = config.rag_top_k
        
        # Extract user query
        user_query = self._extract_user_query(messages)
        
        if not self._should_use_rag(user_query):
            logger.info(f"Skipping RAG for query: {user_query[:50]}...")
            return messages
        
        # Search for relevant documents
        logger.info(f"Searching for documents related to: {user_query[:100]}...")
        relevant_docs = self.rag_index.search(user_query, top_k=top_k)
        
        if not relevant_docs:
            logger.info("No relevant documents found in knowledge base")
            return messages
        
        # Build context from retrieved documents
        context_parts = []
        sources = set()
        total_context_length = 0
        
        for i, doc in enumerate(relevant_docs, 1):
            source = doc.metadata.get('filename', f'Document {i}')
            sources.add(source)
            
            # Truncate very long documents to avoid context overflow
            max_doc_length = config.rag_max_context_length // top_k
            content = doc.content
            if len(content) > max_doc_length:
                content = content[:max_doc_length] + "..."
            
            context_parts.append(f"[Source: {source}]\n{content}")
            total_context_length += len(content)
        
        context = "\n\n".join(context_parts)
        
        # Create enriched messages - make a deep copy to avoid modifying original
        enriched_messages = []
        for msg in messages:
            enriched_messages.append({
                "role": msg.get("role"),
                "content": msg.get("content", "")
            })
        
        # Find the system message or create one
        system_message_idx = None
        for i, msg in enumerate(enriched_messages):
            if msg.get("role") == "system":
                system_message_idx = i
                break
        
        rag_instruction = f"""You have access to relevant information from the knowledge base. Use this information to provide accurate responses. If the context doesn't contain enough information to fully answer the question, say so clearly.

KNOWLEDGE BASE CONTEXT:
{context}

Sources: {', '.join(sources)}
---

"""
        
        if system_message_idx is not None:
            # Prepend to existing system message
            original_content = enriched_messages[system_message_idx]["content"]
            enriched_messages[system_message_idx]["content"] = rag_instruction + original_content
        else:
            # Insert new system message at the beginning
            enriched_messages.insert(0, {
                "role": "system",
                "content": rag_instruction
            })
        
        logger.info(f"RAG: Retrieved {len(relevant_docs)} documents from sources: {sources}")
        logger.info(f"RAG: Total context length: {total_context_length} characters")
        
        return enriched_messages
    
    def _forward_to_backend(self, data: Dict[str, Any]) -> Response:
        """Forward request to backend server."""
        logger.info(f"Forwarding request to backend: {self.backend_url}")
        
        response = requests.post(
            f"{self.backend_url}/v1/chat/completions",
            json=data,
            headers=self._get_proxy_headers(),
            stream=data.get("stream", False),
            timeout=config.backend_timeout
        )
        
        logger.info(f"Backend response status: {response.status_code}")
        
        # Handle streaming response
        if data.get("stream", False):
            logger.info("Returning streaming response")
            return Response(
                response.iter_content(chunk_size=1024),
                content_type=response.headers.get('content-type', 'text/plain'),
                status=response.status_code
            )
        else:
            logger.info("Returning non-streaming response")
            return Response(
                response.content,
                content_type=response.headers.get('content-type', 'application/json'),
                status=response.status_code
            )
    
    def _proxy_request(self, method: str, path: str) -> Response:
        """Proxy other requests to backend."""
        url = f"{self.backend_url}{path}"
        logger.info(f"Proxying {method} request to: {url}")
        
        if method == "GET":
            response = requests.get(url, headers=self._get_proxy_headers())
        elif method == "POST":
            response = requests.post(
                url, 
                json=request.get_json(), 
                headers=self._get_proxy_headers()
            )
        elif method == "PUT":
            response = requests.put(
                url, 
                json=request.get_json(), 
                headers=self._get_proxy_headers()
            )
        elif method == "DELETE":
            response = requests.delete(url, headers=self._get_proxy_headers())
        else:
            return jsonify({"error": f"Method {method} not supported"}), 405
        
        return Response(
            response.content,
            content_type=response.headers.get('content-type', 'application/json'),
            status=response.status_code
        )
    
    def _get_proxy_headers(self) -> Dict[str, str]:
        """Get headers for proxying requests."""
        headers = {}
        
        # Copy relevant headers from original request
        if request.headers:
            for key, value in request.headers:
                if key.lower() in ['authorization', 'content-type', 'user-agent']:
                    headers[key] = value
        
        # Ensure content-type is set for POST requests
        if request.method == "POST" and 'content-type' not in headers:
            headers['Content-Type'] = 'application/json'
        
        return headers

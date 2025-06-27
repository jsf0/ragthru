"""Flask proxy server for RAG middleware."""

import logging
from flask import Flask
from typing import Optional

from ..config import config
from ..rag.index import RAGIndex
from .handlers import ProxyHandlers
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)


class RAGProxyServer:
    """Main RAG proxy server application."""
    
    def __init__(self, rag_index: RAGIndex = None):
        """
        Initialize the RAG proxy server.
        
        Args:
            rag_index: RAG index instance
        """
        # Setup logging
        setup_logging(config)
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['JSON_SORT_KEYS'] = False
        
        # Initialize RAG index
        self.rag_index = rag_index or RAGIndex()
        
        # Initialize handlers
        self.handlers = ProxyHandlers(
            rag_index=self.rag_index,
            backend_url=config.backend_url,
            rag_enabled=config.rag_enabled
        )
        
        # Setup routes
        self._setup_routes()
        
        logger.info("RAG Proxy Server initialized")
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        # OpenAI API compatibility routes
        self.app.add_url_rule(
            '/v1/chat/completions',
            'chat_completions',
            self.handlers.handle_chat_completion,
            methods=['POST']
        )
        
        self.app.add_url_rule(
            '/v1/chat/completions/',
            'chat_completions_slash',
            self.handlers.handle_chat_completion,
            methods=['POST']
        )
        
        self.app.add_url_rule(
            '/v1/models',
            'list_models',
            self.handlers.handle_list_models,
            methods=['GET']
        )
        
        self.app.add_url_rule(
            '/v1/models/',
            'list_models_slash',
            self.handlers.handle_list_models,
            methods=['GET']
        )
        
        self.app.add_url_rule(
            '/models',
            'models_short',
            self.handlers.handle_list_models,
            methods=['GET']
        )
        
        self.app.add_url_rule(
            '/models/',
            'models_short_slash',
            self.handlers.handle_list_models,
            methods=['GET']
        )
        
        # Health and status endpoints
        self.app.add_url_rule(
            '/health',
            'health',
            self.handlers.handle_health,
            methods=['GET']
        )
        
        self.app.add_url_rule(
            '/health/',
            'health_slash',
            self.handlers.handle_health,
            methods=['GET']
        )
        
        # RAG management endpoints
        self.app.add_url_rule(
            '/rag/status',
            'rag_status',
            self.handlers.handle_rag_status,
            methods=['GET']
        )
        
        self.app.add_url_rule(
            '/rag/status/',
            'rag_status_slash',
            self.handlers.handle_rag_status,
            methods=['GET']
        )
        
        self.app.add_url_rule(
            '/rag/toggle',
            'rag_toggle',
            self.handlers.handle_rag_toggle,
            methods=['POST']
        )
        
        self.app.add_url_rule(
            '/rag/toggle/',
            'rag_toggle_slash',
            self.handlers.handle_rag_toggle,
            methods=['POST']
        )
        
        self.app.add_url_rule(
            '/rag/stats',
            'rag_stats',
            self.handlers.handle_rag_stats,
            methods=['GET']
        )
        
        # Root endpoint
        self.app.add_url_rule(
            '/',
            'root',
            self.handlers.handle_root,
            methods=['GET']
        )
        
        # Catch-all proxy for other v1 endpoints
        self.app.add_url_rule(
            '/v1/<path:subpath>',
            'proxy_v1',
            self.handlers.handle_proxy_request,
            methods=['GET', 'POST', 'PUT', 'DELETE']
        )
        
        # Error handlers
        self.app.register_error_handler(404, self.handlers.handle_not_found)
        self.app.register_error_handler(500, self.handlers.handle_internal_error)
    
    def run(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        debug: Optional[bool] = None
    ):
        """
        Run the proxy server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            debug: Enable debug mode
        """
        host = host or config.server_host
        port = port or config.server_port
        debug = debug if debug is not None else config.server_debug
        
        logger.info(f" RAG Proxy Server starting on {host}:{port}")
        logger.info(f" Backend: {config.backend_url}")
        logger.info(f" RAG: {'Enabled' if config.rag_enabled else 'Disabled'}")
        logger.info(f" Management UI: http://{host}:{port}/rag/status")
        logger.info(f" Database: {config.rag_db_path}")
        logger.info(f" Embedding: {config.rag_embedding_type}")
        
        try:
            self.app.run(
                host=host,
                port=port,
                debug=debug,
                threaded=True
            )
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise

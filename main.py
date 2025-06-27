#!/usr/bin/env python3
"""
ragthru

A universal RAG middleware proxy that sits between chat frontends (like LibreChat)
and inference backends (like llama.cpp, Ollama, etc.).
"""

import sys
import os
from pathlib import Path
import click
import logging

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import and ensure modules are loaded to register factories
from src.config import config
from src.proxy.server import RAGProxyServer
from src.rag.index import RAGIndex
from src.utils.logging import setup_logging

# Import storage and embedding modules to ensure registration
import src.storage.sqlite
import src.embeddings.simple

logger = logging.getLogger(__name__)


@click.command()
@click.option('--host', default=None, help='Host to bind to')
@click.option('--port', type=int, default=None, help='Port to bind to')
@click.option('--backend-url', default=None, help='Backend inference server URL')
@click.option('--db-path', default=None, help='RAG database path')
@click.option('--embedding-type', default=None, help='Embedding type to use')
@click.option('--chunk-size', type=int, default=None, help='Text chunk size')
@click.option('--chunk-overlap', type=int, default=None, help='Chunk overlap')
@click.option('--disable-rag', is_flag=True, help='Start with RAG disabled')
@click.option('--add-files', multiple=True, help='Files to add to RAG index')
@click.option('--clear-index', is_flag=True, help='Clear existing RAG index')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--log-level', default=None, help='Logging level')
@click.option('--log-file', default=None, help='Log file path')
def main(
    host, port, backend_url, db_path, embedding_type, chunk_size, chunk_overlap,
    disable_rag, add_files, clear_index, debug, log_level, log_file
):
    try:
        # Override config with command line arguments
        if host is not None:
            config.server_host = host
        if port is not None:
            config.server_port = port
        if backend_url is not None:
            config.backend_url = backend_url
        if db_path is not None:
            config.rag_db_path = db_path
        if embedding_type is not None:
            config.rag_embedding_type = embedding_type
        if chunk_size is not None:
            config.rag_chunk_size = chunk_size
        if chunk_overlap is not None:
            config.rag_chunk_overlap = chunk_overlap
        if disable_rag:
            config.rag_enabled = False
        if debug:
            config.server_debug = True
        if log_level is not None:
            config.log_level = log_level
        if log_file is not None:
            config.log_file = log_file
        
        # Validate configuration
        config.validate()
        
        # Setup logging
        setup_logging(config)
        
        logger.info("Starting RAG Proxy Server...")
        logger.info(f"Configuration loaded from environment")
        
        # Initialize RAG index
        rag_index = RAGIndex()
        
        # Clear index if requested
        if clear_index:
            logger.info("Clearing existing RAG index...")
            rag_index.clear()
        
        # Add files if specified
        if add_files:
            logger.info(f"Adding {len(add_files)} files to RAG index...")
            rag_index.add_files(list(add_files))
        
        # Create and run server
        server = RAGProxyServer(rag_index=rag_index)
        server.run()
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        if config.server_debug:
            raise
        sys.exit(1)


@click.group()
def cli():
    """ragthru CLI utilities."""
    pass


@cli.command()
@click.argument('files', nargs=-1, required=True)
@click.option('--db-path', default=None, help='RAG database path')
@click.option('--embedding-type', default=None, help='Embedding type to use')
@click.option('--chunk-size', type=int, default=None, help='Text chunk size')
@click.option('--chunk-overlap', type=int, default=None, help='Chunk overlap')
def add_documents(files, db_path, embedding_type, chunk_size, chunk_overlap):
    """Add documents to the RAG index."""
    # Override config if provided
    if db_path is not None:
        config.rag_db_path = db_path
    if embedding_type is not None:
        config.rag_embedding_type = embedding_type
    if chunk_size is not None:
        config.rag_chunk_size = chunk_size
    if chunk_overlap is not None:
        config.rag_chunk_overlap = chunk_overlap
    
    # Setup logging
    setup_logging(config)
    
    # Initialize RAG index and add files
    rag_index = RAGIndex()
    rag_index.add_files(list(files))
    
    stats = rag_index.get_stats()
    click.echo(f"Successfully added {len(files)} files")
    click.echo(f"Total documents in index: {stats['storage']['total_documents']}")


@cli.command()
@click.option('--db-path', default=None, help='RAG database path')
def stats(db_path):
    """Show RAG index statistics."""
    if db_path is not None:
        config.rag_db_path = db_path
    
    setup_logging(config)
    
    rag_index = RAGIndex()
    stats = rag_index.get_stats()
    
    click.echo("RAG Index Statistics:")
    click.echo("-" * 40)
    click.echo(f"Total documents: {stats['storage']['total_documents']}")
    click.echo(f"Total content length: {stats['storage']['total_content_length']:,} chars")
    click.echo(f"Unique sources: {stats['storage']['unique_sources']}")
    click.echo(f"Average content length: {stats['storage']['average_content_length']:.1f} chars")
    click.echo(f"Embedding type: {stats['embedding']['type']}")
    click.echo(f"Embedding dimension: {stats['embedding']['dimension']}")
    
    if stats['storage']['sources']:
        click.echo("\nSources:")
        for source in stats['storage']['sources']:
            click.echo(f"  - {source}")


@cli.command()
@click.option('--db-path', default=None, help='RAG database path')
@click.confirmation_option(prompt='Are you sure you want to clear all documents?')
def clear(db_path):
    """Clear all documents from the RAG index."""
    if db_path is not None:
        config.rag_db_path = db_path
    
    setup_logging(config)
    
    rag_index = RAGIndex()
    rag_index.clear()
    click.echo("RAG index cleared successfully")


@cli.command()
@click.argument('query')
@click.option('--db-path', default=None, help='RAG database path')
@click.option('--top-k', type=int, default=5, help='Number of results to return')
def search(query, db_path, top_k):
    """Search the RAG index."""
    if db_path is not None:
        config.rag_db_path = db_path
    
    setup_logging(config)
    
    rag_index = RAGIndex()
    results = rag_index.search(query, top_k=top_k)
    
    click.echo(f"Found {len(results)} results for: {query}")
    click.echo("-" * 60)
    
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get('filename', 'Unknown')
        click.echo(f"{i}. Source: {source}")
        click.echo(f"   Content: {doc.content[:200]}...")
        click.echo()


if __name__ == "__main__":
    # If run directly, use the main command
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ['--help', '-h']):
        main()
    else:
        # Check if first argument is a CLI command
        if len(sys.argv) > 1 and sys.argv[1] in ['add-documents', 'stats', 'clear', 'search']:
            cli()
        else:
            main()

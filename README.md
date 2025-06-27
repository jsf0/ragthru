# ragthru

A universal RAG (Retrieval-Augmented Generation) middleware proxy that sits between chat frontends (like LibreChat) and inference backends (like llama.cpp, Ollama, etc.), adding knowledge base capabilities to any OpenAI-compatible LLM service.

```
LibreChat → ragthru proxy server → llama.cpp/Ollama/vLLM/etc
     ^              ^                    ^
  (Frontend)   (Middleware)         (Backend)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/jsf0/ragthru
cd ragthru

# Install dependencies
pip install -r requirements.txt

```

### Basic Usage

```bash
# Start your inference backend (e.g., llama.cpp)
./llama-server -m model.gguf --port 8080

# Start RAG proxy with documents
python main.py \
    --backend-url http://localhost:8080 \
    --add-files "docs/*.txt" \
    --port 8081

# Configure LibreChat to use the proxy
# In librechat.yaml:
endpoints:
  custom:
    - name: "RAG Enhanced Chat"
      baseURL: "http://localhost:8081"
      apiKey: "dummy"
      models:
        default: ["your-model"]
```

Your model running in llama.cpp is now augmented with the knowledge base built by everything in `docs/*.txt` 

## Configuration

### Environment Variables

You can also create a `.env` file based on `.env.example`:

```bash
# Server Configuration
RAG_PROXY_HOST=0.0.0.0
RAG_PROXY_PORT=8081
BACKEND_URL=http://localhost:8080

# RAG Configuration
RAG_DB_PATH=rag_index.db
RAG_EMBEDDING_TYPE=simple
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=50
RAG_ENABLED=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=rag_proxy.log
```

### Command Line Options

```bash
python main.py --help

Options:
  --host                  Host to bind to
  --port                  Port to bind to  
  --backend-url           Backend inference server URL
  --db-path               RAG database path
  --embedding-type        Embedding type (simple, sentence-transformer, etc.)
  --chunk-size            Text chunk size
  --chunk-overlap         Chunk overlap
  --disable-rag           Start with RAG disabled
  --add-files             Files to add to RAG index (multiple)
  --clear-index           Clear existing RAG index
  --debug                 Enable debug mode
```


### Adding Documents

```bash
# Add documents during startup
python main.py --backend-url http://localhost:8080 --add-files "docs/*.txt"

# Add documents using CLI
python main.py add-documents docs/*.txt docs/*.md
```


## API Endpoints

### OpenAI Compatible

- `POST /v1/chat/completions` - Chat completions with RAG enhancement (proxied)
- `GET /v1/models` - List available models (proxied)

### RAG Management

- `GET /rag/status` - RAG system status and statistics
- `POST /rag/toggle` - Enable/disable RAG
- `GET /rag/stats` - Detailed statistics
- `GET /health` - Health check

### Caveats

- It doesn't work for code files
- I have no idea what will happen if you give it gigantic amounts of text files
- It's not multimodal, so it expects text only.

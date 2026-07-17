# DenseForge v13.2 "Crystalline"

Autonomous Cognitive Knowledge Platform — SOTA RAG + Agents + Causal Reasoning for Hermes Agent.

## Architecture

```
denseforge/
├── core/           # Forge engine (main orchestrator)
├── ingestion/      # Chunking, augmentation, streaming, semantic dedup, columnar metadata
├── retrieval/      # Triple hybrid (BM25 + FAISS + binary), CAG, HIPPO, RAPTOR, fusion
├── reasoning/      # Self-RAG, speculative decoding, conflict resolution, causal
├── synthesis/      # 6 modules: holistic optimizer, shared attention, UQR, session context, bidirectional feedback, adaptive controller
├── generation/     # Attribution, reranking, position tracking, LLM integration
├── embeddings/     # Adaptive + cached embedding pipelines
├── agents/         # Dynamic, memory, and orchestrator agents
├── optimization/   # Continual learning, carbon-aware compute
├── observability/  # Evaluation, provenance, metrics
├── protocols/      # MCP server, router
├── persistence.py  # Save/load to disk (JSON + numpy)
└── cli.py          # Command-line interface
```

## Features

- **Triple Hybrid Retrieval**: BM25 + FAISS dense + binary similarity (v12.0 Harmony)
- **Semantic Dedup**: SHA-256 + cosine similarity (threshold 0.92) — eliminates redundant chunks
- **Columnar Metadata**: NumPy-based storage, 5x faster filtering vs dict-based
- **6 Synthesis Modules**: Context-aware answer generation
- **Self-RAG**: LLM judges its own retrieval quality
- **Causal Reasoning**: Cause-effect chains across knowledge
- **Hermes Agent Integration**: `ingest_knowledge` + `search_knowledge` tools (knowledge toolset)
- **Daemon Mode**: HTTP API on port 9800, systemd service
- **Persistent Storage**: JSON + numpy save/load across restarts

## Quick Start

```bash
# Install
pip install -e .

# Run tests (32/32 passing)
pytest tests/ -v

# Start daemon
python -m denseforge.daemon  # or systemd: denseforge-daemon

# CLI
denseforge ingest "Your knowledge text" --title "Topic" --source "session"
denseforge search "query" --top-k 5
```

## Hermes Agent Integration

DenseForge tools are available via the **`knowledge`** toolset — NOT the core toolset. This saves ~235 tokens per API call by avoiding unnecessary tool schema injection.

### Enable for your platform

```bash
hermes tools enable knowledge
```

### Load as skill

```
/skill denseforge-memory
```

### Available tools

| Tool | Description |
|------|-------------|
| `ingest_knowledge` | Store knowledge in semantic memory |
| `search_knowledge` | Search knowledge base with semantic retrieval |

### Daemon API

```bash
# Ping
curl -X POST http://127.0.0.1:9800 -d '{"cmd":"ping"}'

# Ingest
curl -X POST http://127.0.0.1:9800 -d '{"cmd":"ingest","text":"...","title":"...","source":"session"}'

# Search
curl -X POST http://127.0.0.1:9800 -d '{"cmd":"search","query":"...","top_k":5}'
```

## Token Economics

By removing DenseForge from `_HERMES_CORE_TOOLS`:

| Metric | Before (core) | After (knowledge toolset) |
|--------|---------------|---------------------------|
| Tokens per API call | ~235 | 0 (on-demand only) |
| Daily savings (1000 msgs) | — | ~235,000 tokens |
| Activation | Always | `hermes tools enable knowledge` |

## Testing

```bash
cd /home/dima/denseforge
pytest tests/ -v  # 32/32 passing
```

## License

MIT

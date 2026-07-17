# 🧠 DenseForge — Semantic Memory for AI Agents

<p align="center">
  <em>The missing memory layer for AI agents.</em><br>
  <strong>Store → Search → Reason → Remember</strong>
</p>

<p align="center">
  <a href="https://github.com/zad111ak-ai/denseforge"><img src="https://img.shields.io/github/stars/zad111ak-ai/denseforge?style=social" alt="GitHub Stars"></a>
  <a href="https://pypi.org/project/denseforge/"><img src="https://img.shields.io/pypi/v/denseforge" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python" alt="Python 3.10+">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
</p>

---

DenseForge is a **local semantic memory platform** for AI agents. It stores text, finds relevant context, and returns it with source citations — like a search engine built specifically for LLM conversations.

**Part of [Harvest](https://github.com/zad111ak-ai/harvest)** — a web scraping toolkit for AI agents. DenseForge adds persistent memory to Harvest's scraping pipeline. Install separately if you need semantic memory.

## What It Actually Does

DenseForge gives your AI agent a **brain that persists across sessions**:

1. **Ingest** — Store any text with automatic chunking
2. **Search** — Find relevant context by meaning (not keywords)
3. **Reason** — Ask "why" questions and get source-backed answers
4. **Remember** — Everything stays searchable forever

### Before vs After

| Without DenseForge | With DenseForge |
|---|---|
| Agent forgets after each session | Agent remembers everything |
| Keyword search only | Semantic understanding |
| Manual context injection | Automatic retrieval |
| No source citations | Always shows where info came from |

## Quick Start

```bash
pip install denseforge
```

```python
from denseforge import DenseForge

# Initialize
memory = DenseForge()

# Store information
memory.ingest("Python 3.12 added pattern matching")
memory.ingest("Docker containers share the host kernel")

# Search by meaning
results = memory.search("How does Python handle conditional logic?")

# Ask why
answer = memory.ask_why("Why is Docker faster than VMs?")
```

## MCP Server

```bash
# Add to your MCP agent config
{
  "denseforge": {
    "command": "denseforge-mcp",
    "env": {
      "DENSEFORGE_PATH": "./memory.db"
    }
  }
}
```

## Features

### Semantic Search

Not keyword matching — **meaning-based retrieval**:

```python
# Store
memory.ingest("The quick brown fox jumps over the lazy dog")

# Search (finds related concepts)
results = memory.search("fast animal leaping")  # ✅ Found!
```

### Multi-Document Reasoning

Ask questions across multiple documents:

```python
# Store multiple sources
memory.ingest("Source A: Python is interpreted")
memory.ingest("Source B: Java is compiled")

# Ask cross-document question
answer = memory.ask_why(
    "Why might Python be slower than Java?"
)
# Returns: Explanation with citations to both sources
```

### Adaptive Chunking

Automatic intelligent chunking based on content:

- **Code** — Chunks by function/class boundaries
- **Markdown** — Chunks by headers
- **Plain text** — Chunks by paragraph + semantic breaks

## Architecture

```
denseforge/
├── core/
│   └── forge.py           # Main DenseForge class
├── embeddings/
│   ├── adaptive.py        # Adaptive embedding model
│   └── cache.py           # Embedding cache
├── retrieval/
│   ├── triple_hybrid.py   # BM25 + vector + ColBERT
│   ├── raptor.py          # Hierarchical summarization
│   └── cag.py             # Cache-augmented generation
├── protocols/
│   └── mcp_server.py      # MCP protocol server
└── integrations/
    └── harvest.py         # Harvest bridge
```

## Retrieval Methods

DenseForge uses **triple hybrid** retrieval:

1. **BM25** — Fast keyword matching
2. **Vector Search** — Semantic similarity
3. **ColBERT** — Token-level precision

Results are merged with **Reciprocal Rank Fusion (RRF)** for best accuracy.

## Performance

| Metric | Value |
|---|---|
| Ingest speed | ~100 docs/sec |
| Search latency | <50ms |
| Memory usage | ~100MB per 10k docs |
| Embedding model | all-MiniLM-L6-v2 (80MB) |

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_denseforge.py -v
```

**Test Coverage:** 175 tests across core, retrieval, and protocol modules.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

### Priority Areas
- 🔌 New MCP integrations
- 📊 Benchmark improvements
- 🧪 Edge case testing
- 📝 Documentation

---

## License

MIT License

---

## ☕ Support

If DenseForge saves you time/money, consider buying me a coffee:

- **BTC:** `bc1qd8sa7e4f696wmcyszuxh9snqt2n66zrhz9g80j`
- **ETH:** `0xD26f0efE6A8F11e127c3Af3D6163BD458a1693c3`
- **USDT (TON):** `UQAoI2i8P9-JeZhvGSUwKnymVyY5cb-1Rg7pdqoWMNena7DP`
- **SOL:** `99EtqBVTeF5UNp9a1oPi18iVXbXptTG7YQ6JeJvXMUJK`

---

<p align="center">
  <strong>🧠 DenseForge — Your AI agent remembers everything.</strong>
</p>

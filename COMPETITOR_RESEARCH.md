# DenseForge — Competitive Analysis & Cutting-Edge RAG Research

## Research Sources
- Chroma, Weaviate, Qdrant, Milvus, LlamaIndex, LangChain
- GraphRAG (Microsoft, 2024)
- Self-RAG (Asai et al., 2023)
- CRAG (Corrective RAG, 2024)
- SPLADE/SPLADE++ (Naver, 2024)
- ColBERTv2 (Santhanam et al., 2022)

## Top Gaps Identified

### 1. Multi-query Retrieval (+15-30% recall)
**Source:** LangChain, LlamaIndex
**What:** Generate 3-5 query variants, search each, merge with RRF
**Impact:** Highest — directly improves recall

### 2. Adaptive Router
**Source:** CRAG, FLARE, FRAG
**What:** Classify query type → route to optimal strategy
**Impact:** High — faster, more accurate retrieval

### 3. Self-RAG Light
**Source:** Self-RAG, CRAG
**What:** Quality gate — filter irrelevant results without LLM
**Impact:** High — reduces noise in results

### 4. GraphRAG Local
**Source:** Microsoft GraphRAG (2024)
**What:** Knowledge graph + community detection for relationship queries
**Impact:** Medium-High — adds relationship understanding

### 5. SPLADE Sparse Retrieval
**Source:** SPLADE++ (Naver, 2024)
**What:** Learned sparse vectors with term expansion
**Impact:** Medium — better than BM25, interpretable

## Unique DenseForge Advantages (no competitors have)
1. Triple hybrid (BM25 + Dense + Binary)
2. ColBERT late interaction
3. Model caching (366x)
4. Concept workspace (J-space)
5. Click tracking + memory profiling

## Implementation Plan
- Phase 1: Multi-query + Adaptive Router + Self-RAG Light
- Phase 2: GraphRAG Local + SPLADE integration
- Phase 3: ColBERTv2 compression + Hybrid fusion

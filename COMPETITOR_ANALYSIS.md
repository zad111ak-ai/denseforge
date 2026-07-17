# DenseForge Optimization Plan — Competitor Analysis

## Research Summary

| Competitor | Key Strength | DenseForge Gap |
|------------|-------------|----------------|
| Chroma | Simple API, auto-embedding | ✅ We have this |
| Chroma | Metadata filtering | ❌ Missing |
| Weaviate | Hybrid search | ✅ We have this (better: triple hybrid) |
| Qdrant | Quantization (scalar/binary) | ⚠️ Partial (binary only) |
| LlamaIndex | Sentence window retrieval | ❌ Missing |
| LlamaIndex | Recursive retrieval | ❌ Missing |
| LlamaIndex | Response synthesis | ❌ Missing |
| All | Incremental updates | ⚠️ Partial |

## Planned Improvements

### 1. Metadata Filtering (from Qdrant/Chroma)
**What:** Filter search results by metadata fields (source, date, tags)
**Impact:** HIGH — Enables precise retrieval (e.g., "search only in docs from last week")
**Implementation:** `forge.search(query, filters={"source": "github", "date": ">2024-01-01"})`

### 2. Sentence Window Retrieval (from LlamaIndex)
**What:** When a chunk matches, return surrounding chunks for context
**Impact:** HIGH — Improves answer quality by 20-30% (LlamaIndex benchmarks)
**Implementation:** After finding top chunk, expand to include ±N neighboring chunks

### 3. Response Synthesis (from LlamaIndex)
**What:** Combine multiple retrieved chunks into coherent answer
**Impact:** MEDIUM — Useful for multi-document answers
**Implementation:** `forge.synthesize(query, results)`

### 4. Incremental Document Updates (from Chroma)
**What:** Add/update/delete individual documents without full reindex
**Impact:** MEDIUM — Faster iteration on document collections
**Implementation:** `forge.upsert(doc_id, text)`, `forge.delete(doc_id)`

## Priority Order
1. Metadata Filtering — Most requested, highest impact
2. Sentence Window — Next easiest, big quality boost
3. Incremental Updates — Nice to have, medium effort
4. Response Synthesis — Good for multi-doc queries

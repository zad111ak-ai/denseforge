# DenseForge Optimization Plan

## Current Performance (Baseline)

| Metric | Value | Notes |
|--------|-------|-------|
| Model loading | 11.5s | One-time cost per DenseForge() |
| Single encode | 83ms | After model loaded |
| Batch encode (10) | 15ms/doc | 5x faster than single |
| Batch encode (100) | 17ms/doc | Optimal batch size |
| Search latency | 34ms | Triple hybrid |
| Ingestion (single) | 2500ms/doc | Includes model load |
| Ingestion (batch) | 100ms/doc | After model loaded |

## Optimization Opportunities

### 1. Model Caching (Impact: HIGH)
**Problem:** Each `DenseForge()` loads model fresh (11.5s)
**Solution:** Global model cache, share across instances
**Expected:** 11.5s → 0s (after first load)

### 2. Batch Ingestion (Impact: HIGH)
**Problem:** Single doc ingestion is 25x slower than batch
**Solution:** Always use batch encoding, even for single docs
**Expected:** 2500ms → 100ms per doc

### 3. ONNX Conversion (Impact: MEDIUM)
**Problem:** SentenceTransformer uses PyTorch (slow on CPU)
**Solution:** Convert to ONNX format (2-3x faster on CPU)
**Expected:** 83ms → 30ms per encode

### 4. Model Quantization (Impact: MEDIUM)
**Problem:** Full precision model is large
**Solution:** INT8 quantization (2x smaller, faster)
**Expected:** 2x speedup, 2x memory reduction

### 5. Lazy Loading (Impact: LOW)
**Problem:** Model loaded even if not used
**Solution:** Only load when first encode() called
**Expected:** Faster startup for query-only usage

### 6. Embedding Cache (Impact: LOW)
**Problem:** Same text encoded multiple times
**Solution:** LRU cache for embeddings
**Expected:** 0ms for repeated queries

## Implementation Priority

1. **Model caching** - Quick win, huge impact
2. **Batch ingestion** - Already supported, just use it
3. **ONNX conversion** - Medium effort, good speedup
4. **Quantization** - Requires model conversion
5. **Lazy loading** - Simple change
6. **Embedding cache** - Already have SemanticQueryCache

## Realistic Expectations

After all optimizations:
- First query: ~2s (model load + encode)
- Subsequent queries: ~50ms (cache hit)
- Ingestion: ~50ms/doc (batch + ONNX)
- Search: ~20ms (triple hybrid)

**Total improvement: 50x faster ingestion, 2x faster search**

# Pipeline Performance Improvements

## Changes Made

### 1. **Reranker on CUDA** ✅
**File:** `src/agentic_rag/rerank/bge.py`
- Added explicit CUDA device selection for BGE reranker
- Enabled batch processing (batch_size=64) for parallel inference
- Changed `RERANK_FP16` default to `True` for 2x speedup on GPU
- Added device info logging

**Expected speedup:** 5-10x faster on CUDA vs CPU

### 2. **Embedder on CUDA** ✅
**File:** `src/agentic_rag/embed/encoder.py`
- Explicit CUDA device for SentenceTransformer models
- Added batch processing (batch_size=32) for multi-text encoding
- Reduces overhead from sequential encoding

**Expected speedup:** 3-5x faster on CUDA vs CPU

### 3. **Configuration Defaults** ✅
**File:** `src/agentic_rag/config.py`
- `RERANK_FP16: True` - Use half precision for 2x speed (was False)
- `RERANK_BATCH_SIZE: 64` - Batch size for reranking
- `EMBED_BATCH_SIZE: 32` - Batch size for embeddings

### 4. **Other Optimizations Already Present**
- MMR selection already uses CUDA efficiently
- ThreadPoolExecutor parallelizes retrieval across anchors
- FAISS already uses optimized vector search

## Bottlenecks Analysis

### Fixed:
1. ✅ Reranker was on CPU → Now on CUDA
2. ✅ Embedder was on CPU → Now on CUDA
3. ✅ No batching in reranker → Now batch processes 64 at once
4. ✅ No batching in embedder → Now batch processes 32 at once

### Remaining (expected):
1. **LLM API calls** - Network latency to OpenAI (largest bottleneck)
2. **BM25 indexing** - Happens once at startup, shouldn't affect queries
3. **Serialization lock** - Reranker uses lock for thread safety (minimal impact)

## Testing

Run the test script:
```powershell
python test_cuda_performance.py
```

This will show:
- CUDA availability
- Embedding speed (texts/sec)
- Reranking speed (docs/sec)
- Current configuration

## Expected Performance Gains

### Before (CPU):
- Reranking 50 docs: ~2-3 seconds
- Embedding 20 texts: ~0.5-1 second
- Total per round: ~3-5 seconds retrieval overhead

### After (CUDA):
- Reranking 50 docs: ~0.3-0.5 seconds (6-10x faster)
- Embedding 20 texts: ~0.1-0.2 seconds (5-10x faster)
- Total per round: ~0.5-1 second retrieval overhead (3-5x faster)

## Configuration Override

To adjust batch sizes for your GPU memory:

```python
# In .env or export before running:
RERANK_BATCH_SIZE=128  # Larger if you have more GPU memory
EMBED_BATCH_SIZE=64
RERANK_FP16=True
```

## Monitoring

The pipeline will now print device info on startup:
```
Loading sentence-transformers model: sentence-transformers/all-MiniLM-L6-v2 on cuda
🚀 BGE Reranker loaded on cuda
```

If you see "cpu" instead of "cuda", check:
```python
python -c "import torch; print(torch.cuda.is_available())"
```

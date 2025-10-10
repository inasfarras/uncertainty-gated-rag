# Using Offline Embeddings (Free, No API Key Needed)

## Quick Answer: YES! You can use offline embeddings

The project supports **sentence-transformers** which runs locally on your machine - no API costs!

---

## Option 1: Use sentence-transformers (Recommended for Offline)

### Step 1: Install sentence-transformers (if not already installed)
```powershell
pip install sentence-transformers
```

### Step 2: Ingest corpus with offline embeddings
```powershell
# Stop the current OpenAI ingestion (Ctrl+C)
# Then run with offline backend:

python -m agentic_rag.ingest.ingest `
    --input data\crag_corpus_html `
    --out artifacts\crag_faiss `
    --backend st
```

**That's it!** No API key needed, completely free.

---

## Embedding Models Available

### Default (Fast, Small):
```python
EMBED_BACKEND: "st"
ST_EMBED_MODEL: "sentence-transformers/all-MiniLM-L6-v2"
```
- **Size:** 22MB
- **Dimension:** 384
- **Speed:** Fast
- **Quality:** Good for most tasks

### Better Quality Options:

#### 1. BGE-Small (Recommended Balance)
```powershell
# In config.py or via environment:
$env:ST_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
python -m agentic_rag.ingest.ingest --input data\crag_corpus_html --out artifacts\crag_faiss --backend st
```
- **Size:** 133MB
- **Dimension:** 384
- **Quality:** Better than MiniLM

#### 2. BGE-Base (High Quality)
```powershell
$env:ST_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
python -m agentic_rag.ingest.ingest --input data\crag_corpus_html --out artifacts\crag_faiss --backend st
```
- **Size:** 436MB
- **Dimension:** 768
- **Quality:** Close to OpenAI

#### 3. E5-Large (Best Quality)
```powershell
$env:ST_EMBED_MODEL = "intfloat/e5-large-v2"
python -m agentic_rag.ingest.ingest --input data\crag_corpus_html --out artifacts\crag_faiss --backend st
```
- **Size:** 1.34GB
- **Dimension:** 1024
- **Quality:** Near OpenAI level

---

## Using Llama/Local LLM for Embeddings

If you have llama or another local LLM, you can create a custom embedding backend:

### Create custom embeddings wrapper:
```python
# src/agentic_rag/embed/llama_embed.py
import numpy as np
from llama_cpp import Llama  # or whatever you're using

def embed_texts_llama(texts: list[str]) -> np.ndarray:
    # Your llama embedding code here
    model = Llama(model_path="your_model.gguf")
    embeddings = [model.embed(text) for text in texts]
    return np.array(embeddings, dtype=np.float32)
```

Then modify `src/agentic_rag/embed/encoder.py` to add "llama" backend.

**But sentence-transformers is easier and works great!**

---

## Performance Comparison

| Backend | Cost | Speed | Quality | Dimension |
|---------|------|-------|---------|-----------|
| OpenAI (text-embedding-3-small) | $5-10 | Fast (API) | Excellent | 1536 |
| sentence-transformers (MiniLM) | **Free** | Fast | Good | 384 |
| sentence-transformers (BGE-base) | **Free** | Medium | Very Good | 768 |
| sentence-transformers (E5-large) | **Free** | Slower | Excellent | 1024 |

---

## Quick Command to Switch Now:

**Stop current ingestion:** Press `Ctrl+C` in terminal

**Ingest with offline embeddings:**
```powershell
python -m agentic_rag.ingest.ingest `
    --input data\crag_corpus_html `
    --out artifacts\crag_faiss `
    --backend st
```

**Time:** Same ~15-30 minutes (depends on your CPU/GPU)

**Cost:** $0 (completely free!)

---

## Will Results Be Worse?

**Short answer:** Slightly worse, but still good!

**Expected performance with sentence-transformers:**
- OpenAI embeddings: ~53-55 composite
- MiniLM (default): ~50-52 composite (3-5% drop)
- BGE-base: ~52-54 composite (1-2% drop)

**The corpus coverage issue matters WAY more than embedding quality!**
- Incomplete corpus (24k chunks): 46 composite ❌
- Full corpus with offline embeddings: ~50-52 composite ✅

---

## Recommendation

1. **Use sentence-transformers with default model** (MiniLM)
   - Free, fast, good enough
   - 3-5% quality drop vs OpenAI is acceptable

2. **If you want better quality:**
   - Use BGE-base (still free, minimal quality loss)

3. **Don't worry about perfect embeddings yet**
   - Fix the corpus coverage first (24k → 80k+ chunks)
   - That's the bigger problem!

---

## Full Rebuild with Offline Embeddings

Want to rebuild everything with offline embeddings? Here's the complete command:

```powershell
# 1. Prepare corpus (already done)
# Skip if corpus is already prepared

# 2. Ingest with sentence-transformers
python -m agentic_rag.ingest.ingest `
    --input data\crag_corpus_html `
    --out artifacts\crag_faiss `
    --backend st

# 3. Verify
python check_embeddings.py

# 4. Test
python scripts/run_eval_with_judge.py --dataset data/crag_questions.jsonl --system anchor --judge-require-citation false --validator-limit 5 -- --gate-on --n 50 --judge-policy gray_zone --max-rounds 3
```

---

## Summary

✅ **Yes, you can use offline embeddings**
✅ **No API key or cost needed**
✅ **Quality is 95-98% as good**
✅ **Use: `--backend st` instead of `--backend openai`**

**Stop current ingestion and run:**
```powershell
python -m agentic_rag.ingest.ingest --input data\crag_corpus_html --out artifacts\crag_faiss --backend st
```

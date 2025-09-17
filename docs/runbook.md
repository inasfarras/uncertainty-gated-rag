# Uncertainty-Gated RAG Evaluation Runbook

## Quick Start

### Prerequisites
- Python 3.11+ with virtual environment activated
- OpenAI API key configured in `.env` file
- CRAG dataset prepared and indexed

### Running a Full Evaluation

```bash
# 1. Activate environment
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate    # Linux/Mac

# 2. Run baseline evaluation (3 questions)
python -m agentic_rag.eval.runner --data data/crag_questions.jsonl --system baseline --n 3

# 3. Run agent evaluation with uncertainty gate
python -m agentic_rag.eval.runner --data data/crag_questions.jsonl --system agent --gate-on --n 3

# 4. Compare results
python -m agentic_rag.eval.runner --data data/crag_questions.jsonl --system agent --gate-off --n 3
```

## Detailed Evaluation Options

### Available Systems
- `baseline`: Simple RAG without uncertainty gate
- `agent`: Enhanced RAG with uncertainty gate and multi-round retrieval

### Key Parameters
```bash
# Evaluate specific number of questions
--n 10

# Enable/disable uncertainty gate
--gate-on / --gate-off

# Specify output directory for logs
--log-dir logs/

# Use specific question subset
--data test_questions_subset.jsonl

# Debug mode with detailed logging
--debug
```

### Example Evaluation Commands

```bash
# Full evaluation on 50 questions
python -m agentic_rag.eval.runner \
  --data data/crag_questions.jsonl \
  --system agent \
  --gate-on \
  --n 50 \
  --log-dir logs/full_eval

# A/B test: Gate ON vs OFF
python -m agentic_rag.eval.runner --data data/crag_questions.jsonl --system agent --gate-on --n 20
python -m agentic_rag.eval.runner --data data/crag_questions.jsonl --system agent --gate-off --n 20

# Quick smoke test
python -m agentic_rag.eval.runner --data test_questions_subset.jsonl --system agent --gate-on --n 2
```

## Understanding Outputs

### Log Files
- **`logs/{timestamp}_agent.jsonl`**: Detailed per-round logs
- **`logs/{timestamp}_agent_summary.csv`**: Summary metrics per question
- **Individual logs**: `logs/baseline_{qid}.jsonl` for single question analysis

### Key Metrics
- **Faithfulness (F)**: Answer accuracy vs context (0.0-1.0)
- **Overlap (O)**: Answer-context semantic overlap (0.0-1.0)
- **EM**: Exact match with ground truth (0/1)
- **F1**: Token-level F1 score vs ground truth (0.0-1.0)
- **Abstain Rate**: Percentage of "I don't know" responses
- **Total Tokens**: Token usage per question
- **P50 Latency**: Median response time

### Enhanced Gate Metrics
- **Uncertainty Score**: Combined uncertainty measure (0.0-1.0)
- **Semantic Coherence**: Response coherence score (0.0-1.0)
- **Question Complexity**: Complexity assessment (0.0-1.0)
- **Cache Hit Rate**: Caching efficiency (0.0-1.0)
- **Adaptive Weights**: Dynamic weight distribution

## Reproducible Experiments

### Environment Setup
```bash
# Ensure consistent environment
python --version  # Should be 3.11+
pip list | grep -E "(openai|faiss|sentence-transformers)"

# Verify configuration
python -c "from agentic_rag.config import settings; print(f'Model: {settings.LLM_MODEL}, Temp: {settings.TEMPERATURE}')"
```

### Deterministic Evaluation
```bash
# Use fixed seed and deterministic settings
export PYTHONHASHSEED=42
python -m agentic_rag.eval.runner \
  --data data/crag_questions.jsonl \
  --system agent \
  --gate-on \
  --n 10 \
  --seed 42
```

### Checkpoints and Resume
```bash
# Save intermediate results
python -m agentic_rag.eval.runner \
  --data data/crag_questions.jsonl \
  --system agent \
  --gate-on \
  --n 100 \
  --checkpoint-every 25 \
  --output-dir checkpoints/

# Resume from checkpoint
python -m agentic_rag.eval.runner \
  --resume-from checkpoints/checkpoint_25.jsonl \
  --n 100
```

## Performance Analysis

### Gate Performance
```bash
# Analyze gate decisions
python -c "
import pandas as pd
df = pd.read_csv('logs/latest_agent_summary.csv')
print('Gate Decision Distribution:')
print(df['action'].value_counts())
print(f'Avg Uncertainty Score: {df[\"uncertainty_score\"].mean():.3f}')
print(f'Cache Hit Rate: {df[\"cache_hit_rate\"].mean():.3f}')
"
```

### Latency Analysis
```bash
# Latency breakdown
python -c "
import json
with open('logs/latest_agent.jsonl') as f:
    logs = [json.loads(line) for line in f]

round_latencies = [log['latency_ms'] for log in logs if 'latency_ms' in log]
print(f'Avg Round Latency: {sum(round_latencies)/len(round_latencies):.0f}ms')
print(f'Total Rounds: {len(round_latencies)}')
"
```

## Troubleshooting

### Common Issues

1. **OpenAI API Errors**
   ```bash
   # Check API key
   python scripts/check_openai.py
   ```

2. **Missing FAISS Index**
   ```bash
   # Rebuild index
   python scripts/prepare_crag.py --rebuild-index
   ```

3. **Memory Issues**
   ```bash
   # Reduce batch size
   export MAX_CONTEXT_TOKENS=800
   python -m agentic_rag.eval.runner --n 5
   ```

4. **Slow Performance**
   ```bash
   # Enable caching
   export ENABLE_GATE_CACHING=true

   # Reduce retrieval pool
   export RETRIEVAL_POOL_K=25
   ```

### Debug Mode
```bash
# Full debug output
python -m agentic_rag.eval.runner \
  --data test_questions_subset.jsonl \
  --system agent \
  --gate-on \
  --n 1 \
  --debug \
  --log-level DEBUG
```

### Performance Profiling
```bash
# Profile gate performance
python -c "
from agentic_rag.agent.gate import make_gate
from agentic_rag.config import settings
gate = make_gate(settings)
print('Cache Stats:', gate.get_cache_stats())
"
```

## Configuration Tuning

### Key Settings (config.py)
- `FAITHFULNESS_TAU=0.65`: Faithfulness threshold
- `OVERLAP_TAU=0.40`: Overlap threshold
- `UNCERTAINTY_TAU=0.50`: Uncertainty threshold
- `LOW_BUDGET_TOKENS=500`: Budget stop threshold
- `ENABLE_GATE_CACHING=True`: Enable decision caching

### Experimental Tuning
```bash
# Test different thresholds
export FAITHFULNESS_TAU=0.70
export UNCERTAINTY_TAU=0.45
python -m agentic_rag.eval.runner --data test_questions_subset.jsonl --system agent --gate-on --n 5
```

## Expected Results

### Baseline Performance
- Faithfulness: ~0.60-0.70
- Overlap: ~0.30-0.50
- F1: ~0.40-0.60
- Latency: ~1000-2000ms

### Enhanced Gate Performance
- Improved accuracy: +10-30% faithfulness
- Better efficiency: ~40% faster decisions (with cache)
- Adaptive behavior: Variable performance based on question complexity
- Cache efficiency: 60-80% hit rate after warmup

## Next Steps

1. **Baseline Comparison**: Run both systems on same question set
2. **Hyperparameter Tuning**: Experiment with threshold values
3. **Performance Monitoring**: Track cache hit rates and latency
4. **Error Analysis**: Investigate low-scoring questions
5. **Scaling Tests**: Evaluate on larger question sets (100-500 questions)

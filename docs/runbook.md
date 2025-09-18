# Runbook for Analysis and A/B Test

## 1. Reproduce Analysis

To reproduce the analysis, run the log parsing script. This will output the diagnostic table for the 10 questions.

```bash
python parse_logs.py
```

## 2. Gate-on vs. Gate-off A/B Test (N=20)

To run an A/B test comparing the agent with the uncertainty gate on and off, use the evaluation runner.

```bash
# Run with gate ON (default)
python -m src.agentic_rag.eval.runner --n_items 20 --agent_system agent --gate_on > gate_on_results.txt

# Run with gate OFF
python -m src.agentic_rag.eval.runner --n_items 20 --agent_system agent --no-gate_on > gate_off_results.txt
```

# Evaluation results

Model: `claude-haiku-4-5-20251001` · 30 questions · total agent cost ≈ $0.66

| Metric | factual | comparison | limitations | all |
|---|---|---|---|---|
| Questions | 10 | 10 | 10 | 30 |
| Successful reports | 10/10 | 10/10 | 10/10 | 30/30 |
| Coverage of expected points | 91% | 98% | 82% | 90% |
| Avg findings per report | 5.2 | 7.4 | 7.3 | 6.6 |
| Avg latency | 14.4 s | 36.2 s | 40.0 s | 30.2 s |
| Avg tool calls | 3.1 | 7.5 | 7.6 | 6.1 |
| Avg tokens | 9,668 | 22,082 | 21,843 | 17,864 |
| Avg cost per question | $0.0123 | $0.0270 | $0.0267 | $0.0220 |
| Step budget reached | 0 | 0 | 0 | 0 |

## Lowest coverage

| id | category | coverage | question |
|---|---|---|---|
| q29 | limitations | 50% | How well do current LLM agents perform on long, multi-step tasks? |
| q30 | limitations | 50% | What are the privacy risks of sending user documents to LLM APIs? |
| q02 | factual | 67% | What is the ReAct prompting framework for LLM agents? |
| q04 | factual | 67% | What is the Model Context Protocol (MCP)? |
| q06 | factual | 75% | What is Reflexion in LLM agents? |

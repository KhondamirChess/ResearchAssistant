# Evaluation set

`questions.csv` holds 30 research questions used to evaluate the agent in Live mode.

| Column | Meaning |
|---|---|
| `id` | `q01` … `q30` |
| `category` | `factual` (one fact or definition, easy to check) · `comparison` (needs several sources) · `limitations` (open problems, sources may disagree) |
| `difficulty` | `easy` · `medium` · `hard` |
| `question` | Text sent to the agent |
| `expected_points` | Key points a good report should contain, separated by `; ` |

10 questions per category. The topics are AI / LLM research, so answers can be checked
against well-known papers and documentation.

## How to score a run (planned for v1.0)

For every question record:

- **Success**: a report was returned (no error).
- **Coverage**: share of `expected_points` found in the findings (manual check or LLM judge).
- **Faithfulness**: findings are supported by the sources the agent read; no invented facts.
- **Cost**: steps, tool calls, input / output tokens, latency, USD (from the `metrics` event
  and `logs/app.log`).

`expected_points` are a checklist for grading, not the only correct answer. Points that
depend on dates (laws, products, lawsuits) should be re-checked before each evaluation.

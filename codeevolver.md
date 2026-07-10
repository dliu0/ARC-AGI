```
PARENT_MODULE_PATH: src.program.arc_pipeline.ARCPipeline
METRIC_MODULE_PATH: src.metric.metric.arc_grid_accuracy

## ARCHITECTURE TITLE: Single-shot LM ARC solver (DeepSeek-V4-Flash via litellm) with strict binary grid metric

## ARCHITECTURE SUMMARY:
The program is a thin, single-module pipeline that delegates all ARC-AGI reasoning to a single large language model call. `src/program/arc_pipeline.py` defines `ARCPipeline`, which builds a text prompt from a task's demonstration pairs (input/output grids serialized as JSON), asks the model to infer the transformation rule, and emits a JSON array-of-arrays prediction per test input. There is no separate inference engine, no DSL, no candidate generation, no voting, and no verifier — just one LM call per test case. The metric lives in `src/metric/metric.py` (`arc_grid_accuracy`) and scores predictions against the gold outputs with a strict binary 0.0/1.0 exact-match rule per test case, returning a rich feedback string for the optimizer.

## ARCHITECTURE DESCRIPTION:
`src/program/arc_pipeline.py::ARCPipeline` is the entry point invoked per task row. Its `__init__` configures a single LM backend: `openai/deepseek-ai/DeepSeek-V4-Flash` served through the GMI Cloud OpenAI-compatible endpoint (`https://api.gmi-serving.com/v1`) using the `litellm.completion` client with `reasoning_effort="high"` and a 2400s timeout. The module-level code also wires up OpenTelemetry tracing (OTLP exporter, defaulting to a local collector) so each `__call__` runs inside a `arc_predict` span tagged with `task_id` and `num_predictions`.

`__call__(self, train, test, task_id)` is the inference path. It constructs a fixed-system prompt instructing the model to act as an expert at visual/mathematical reasoning, then iterates over `train` cases appending each `Pair {i}` input/output grid serialized via `json.dumps`. For each test case in `test`, it appends `Test Case:\nInput: {...}` plus a strict instruction to output ONLY a valid JSON array-of-arrays. The raw model content is cleaned of any leading `<think>...</think>` reasoning block, stripped of any ```json fences, then `json.loads`-parsed. Validation requires a list-of-lists; anything malformed or any exception falls back to echoing the raw `test_input` grid (so errors do not abort the run but produce near-certainly-wrong answers). Predictions are returned as a list of grids (one per test case).

The metric, `src/metric/metric.py::arc_grid_accuracy`, accepts either DSPy-style positional args `(example, pred, trace)` or the evaluator's keyword form `output=..., example=Example(row)` via `_extract`. It normalizes cells so `"3"` equals `3` (type-normalized comparison), then for each test case: checks the prediction is a well-formed 2D list, compares predicted vs. target dimensions (mismatch → 0.0 with a dims-feedback note), and otherwise compares cells. A case scores 1.0 only when every cell matches exactly; otherwise 0.0 — there is no partial credit. The overall score is the fraction of cases solved exactly.

The metric's distinguishing feature is its feedback payload. Even though the score is binary (preventing partial-credit reward hacking), it returns rich textual feedback for the optimizer: the demonstration pairs rendered as readable grids (`_demos_str`), per-case notes including test input, predicted grid, correct output grid, an exact cell-by-cell diff for shape-correct misses (`_cell_diff`), and the set of colors used. This feedback intentionally reveals the gold output so the optimizer can reflect on the missed rule, with the understanding that real generalization is judged on a held-out test set the optimizer never sees during a run.

The data flow is therefore: task JSON → `ARCPipeline.__call__` builds prompt → single LM call per test input → JSON grid parsed and returned as predictions → `arc_grid_accuracy` consumes predictions vs. gold outputs → returns `{score, feedback}` for the optimizer to refine the program.
```

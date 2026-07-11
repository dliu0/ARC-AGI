import os
import re
import json
import traceback

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Set up tracing
provider = TracerProvider()
# Read endpoint from environment, defaulting to local Jaeger/Collector
otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318/v1/traces")
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)

import litellm


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert at solving ARC-AGI visual reasoning puzzles.\n"
    "Each puzzle gives 2-5 demonstration pairs of input/output grids. "
    "Every grid is a 2D array of single-digit integers 0-9. "
    "There is exactly ONE abstract transformation rule that maps every input "
    "grid to its output grid. Your job is to figure out that rule and express "
    "it as a Python function so it can be verified and applied.\n"
)


def _build_prompt(train_cases, test_cases):
    """Build the full user-message prompt for one ARC task.

    The model is asked to:
      1. Reason about the rule (it has reasoning_effort=high).
      2. Write a Python ``transform(grid)`` function implementing the rule.
      3. Provide the predicted output for the first test input as a JSON
         array-of-arrays (used as a fallback if the code can't be verified).
    """
    prompt = SYSTEM_PROMPT + "\n"

    # ---- Demonstrations ----
    prompt += "=== DEMONSTRATION PAIRS ===\n"
    for i, case in enumerate(train_cases):
        prompt += f"Pair {i + 1}:\n"
        prompt += f"Input:  {json.dumps(case.get('input'))}\n"
        prompt += f"Output: {json.dumps(case.get('output'))}\n\n"

    # ---- Test inputs ----
    prompt += "=== TEST INPUTS ===\n"
    for i, tc in enumerate(test_cases):
        prompt += f"Test {i + 1}: {json.dumps(tc.get('input', []))}\n"
    prompt += "\n"

    # ---- Instructions ----
    prompt += (
        "=== INSTRUCTIONS ===\n"
        "Step 1: Study the demonstration pairs and determine the single "
        "transformation rule that maps every input grid to its output grid.\n"
        "Step 2: Write a Python function named `transform` that takes one "
        "argument `grid` (a list of lists of ints, the input grid) and returns "
        "the output grid (also a list of lists of ints). The function must "
        "implement the rule EXACTLY — it will be tested against ALL "
        "demonstration pairs before being applied to the test inputs.\n"
        "Step 3: Also provide the predicted output for the FIRST test input "
        "as a JSON array-of-arrays, in case the code cannot be executed.\n\n"
        "=== OUTPUT FORMAT ===\n"
        "Respond with exactly two fenced blocks:\n"
        "```\n"
        "```python\n"
        "def transform(grid):\n"
        "    # your code here\n"
        "    return output_grid\n"
        "```\n"
        "\n"
        "```json\n"
        "[[...], ...]\n"
        "```\n"
        "\n"
        "The python block must define `transform(grid)`. The json block must "
        "contain the predicted output for the first test input only. Do not "
        "include any text outside the two fenced blocks."
    )
    return prompt


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _strip_reasoning(text):
    """Remove <think>...</think> / <reasoning>...</reasoning> blocks."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL)
    return text.strip()


def _extract_code_blocks(content):
    """Return list of (language, body) tuples from fenced code blocks."""
    blocks = []
    # Match ```lang\n ... ``` (greedy on inner, non-greedy across blocks)
    for m in re.finditer(
        r"```([a-zA-Z0-9_]*)\n?(.*?)```", content, flags=re.DOTALL
    ):
        lang = m.group(1).strip().lower()
        body = m.group(2).strip()
        blocks.append((lang, body))
    return blocks


def _parse_python_code(blocks):
    """Find the first python/plain block containing a def transform."""
    for lang, body in blocks:
        if lang in ("python", "py", ""):
            if re.search(r"def\s+transform\s*\(", body):
                return body
    # Fallback: any block with def transform
    for lang, body in blocks:
        if re.search(r"def\s+transform\s*\(", body):
            return body
    return None


def _parse_json_prediction(blocks, content):
    """Find the JSON array-of-arrays prediction from blocks or raw content."""
    # Prefer a json-labeled block
    for lang, body in blocks:
        if lang in ("json", "") and body.startswith("["):
            try:
                pred = json.loads(body)
                if isinstance(pred, list) and all(
                    isinstance(r, list) for r in pred
                ):
                    return pred
            except (json.JSONDecodeError, ValueError):
                pass
    # Fallback: scan raw content for the last JSON array
    for m in re.finditer(r"(\[\s*\[.*?\]\s*\])", content, flags=re.DOTALL):
        try:
            pred = json.loads(m.group(1))
            if isinstance(pred, list) and all(
                isinstance(r, list) for r in pred
            ):
                return pred
        except (json.JSONDecodeError, ValueError):
            continue
    return None


# ---------------------------------------------------------------------------
# Function verification & execution
# ---------------------------------------------------------------------------

# Restricted builtins for exec — enough for list/grid manipulation but no
# file/network/import access.  (The eval environment is already sandboxed;
# this is defense-in-depth.)
_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "divmod": divmod, "enumerate": enumerate, "filter": filter, "float": float,
    "int": int, "len": len, "list": list, "map": map, "max": max, "min": min,
    "range": range, "reversed": reversed, "round": round, "set": set,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "print": print, "isinstance": isinstance, "repr": repr, "chr": chr,
    "ord": ord, "frozenset": frozenset, "True": True, "False": False,
    "None": None, "complex": complex, "pow": pow, "hex": hex, "bin": bin,
    "oct": oct, "format": format, "hasattr": hasattr, "getattr": getattr,
    "setattr": setattr, "type": type, "slice": slice,
}


def _compile_transform(code_str):
    """Exec the model's code and return the `transform` callable, or None."""
    if not code_str:
        return None
    namespace = {"__builtins__": _SAFE_BUILTINS}
    try:
        exec(compile(code_str, "<transform>", "exec"), namespace)
        transform = namespace.get("transform")
        if callable(transform):
            return transform
    except Exception:
        return None
    return None


def _norm_cell(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return x


def _grids_equal(a, b):
    """Exact, type-normalized grid comparison."""
    if not isinstance(a, list) or not isinstance(b, list):
        return False
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if not isinstance(ra, list) or not isinstance(rb, list):
            return False
        if len(ra) != len(rb):
            return False
        for ca, cb in zip(ra, rb):
            if _norm_cell(ca) != _norm_cell(cb):
                return False
    return True


def _verify_transform(transform_fn, train_cases):
    """Return True iff transform reproduces EVERY train output exactly."""
    if transform_fn is None:
        return False
    for case in train_cases:
        inp = case.get("input")
        expected = case.get("output")
        try:
            got = transform_fn([row[:] for row in inp])
        except Exception:
            return False
        if not _grids_equal(got, expected):
            return False
    return True


def _apply_transform(transform_fn, test_cases):
    """Apply transform to each test input; return list of grids."""
    outputs = []
    for tc in test_cases:
        inp = tc.get("input", [])
        try:
            got = transform_fn([row[:] for row in inp])
            if isinstance(got, list) and all(
                isinstance(r, list) for r in got
            ):
                outputs.append(got)
            else:
                outputs.append(inp)
        except Exception:
            outputs.append(inp)
    return outputs


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ARCPipeline:
    def __init__(self):
        # Solver LM: DeepSeek-V4-Flash on GMI Cloud (reasoning=high set on the
        # call below). In this experiment the optimizer (GLM-5.2) and this
        # solver both run on GMI, so we pass the GMI endpoint + key explicitly
        # (the proven GMI-as-OpenAI pattern) rather than relying on OPENAI_*.
        self.model = "openai/deepseek-ai/DeepSeek-V4-Flash"
        self.api_base = "https://api.gmi-serving.com/v1"
        self.api_key = os.environ.get("GMI_CLOUD_API_KEY") or os.environ.get("GMI_API_KEY")

    def __call__(self, train: list = None, test: list = None, task_id: str = "unknown", **kwargs) -> list:
        with tracer.start_as_current_span("arc_predict") as span:
            span.set_attribute("task_id", task_id)

            train_cases = train or []
            test_cases = test or []

            # Default fallback: return each test input unchanged
            fallback = [tc.get("input", []) for tc in test_cases]

            if not train_cases or not test_cases:
                span.set_attribute("num_predictions", len(fallback))
                return fallback

            prompt = _build_prompt(train_cases, test_cases)

            try:
                response = litellm.completion(
                    model=self.model,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    messages=[{"role": "user", "content": prompt}],
                    reasoning_effort="high",
                    allowed_openai_params=["reasoning_effort"],
                    timeout=2400,
                    num_retries=1,
                )
                content = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error calling LLM: {e}")
                span.set_attribute("num_predictions", len(fallback))
                return fallback

            content = _strip_reasoning(content)
            blocks = _extract_code_blocks(content)

            # --- Path 1: Python transform verified against all train pairs ---
            code_str = _parse_python_code(blocks)
            transform_fn = _compile_transform(code_str)
            verified = _verify_transform(transform_fn, train_cases)

            if verified:
                outputs = _apply_transform(transform_fn, test_cases)
                span.set_attribute("num_predictions", len(outputs))
                span.set_attribute("verified", True)
                return outputs

            # --- Path 2: fallback to direct JSON prediction ---
            json_pred = _parse_json_prediction(blocks, content)

            outputs = []
            for i, tc in enumerate(test_cases):
                ti = tc.get("input", [])
                if i == 0 and json_pred is not None:
                    outputs.append(json_pred)
                else:
                    outputs.append(ti)

            span.set_attribute("num_predictions", len(outputs))
            span.set_attribute("verified", False)
            return outputs

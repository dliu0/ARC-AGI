import os
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

import json
import re
import signal
import copy
import litellm


class _TransformTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _TransformTimeout()


_SAFE_BUILTINS = {
    'range': range, 'len': len, 'int': int, 'str': str, 'list': list,
    'dict': dict, 'tuple': tuple, 'set': set, 'frozenset': frozenset,
    'enumerate': enumerate, 'zip': zip, 'map': map, 'filter': filter,
    'sorted': sorted, 'reversed': reversed, 'min': min, 'max': max,
    'sum': sum, 'abs': abs, 'any': any, 'all': all, 'bool': bool,
    'float': float, 'round': round, 'print': print, 'isinstance': isinstance,
    'type': type, 'copy': copy.copy, 'deepcopy': copy.deepcopy,
    'ord': ord, 'chr': chr, 'divmod': divmod, 'repr': repr, 'pow': pow,
    'bin': bin, 'oct': oct, 'hex': hex, 'hasattr': hasattr, 'getattr': getattr,
    'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
    'IndexError': IndexError, 'KeyError': KeyError, 'ZeroDivisionError': ZeroDivisionError,
    'StopIteration': StopIteration, 'ArithmeticError': ArithmeticError,
    'OverflowError': OverflowError, 'True': True, 'False': False, 'None': None,
}


class ARCPipeline:
    def __init__(self):
        # Solver LM: DeepSeek-V4-Flash on GMI Cloud (reasoning=high set on the
        # call below). In this experiment the optimizer (GLM-5.2) and this solver
        # both run on GMI, so we pass the GMI endpoint + key explicitly (the
        # proven GMI-as-OpenAI pattern) rather than relying on OPENAI_* env.
        self.model = "openai/deepseek-ai/DeepSeek-V4-Flash"
        self.api_base = "https://api.gmi-serving.com/v1"
        self.api_key = os.environ.get("GMI_CLOUD_API_KEY") or os.environ.get("GMI_API_KEY")

    def __call__(self, train: list = None, test: list = None, task_id: str = "unknown", **kwargs) -> list:
        with tracer.start_as_current_span("arc_predict") as span:
            span.set_attribute("task_id", task_id)

            train_cases = train or []
            test_cases = test or []

            if not test_cases:
                span.set_attribute("num_predictions", 0)
                return []

            prompt = self._build_prompt(train_cases, test_cases)

            try:
                response = litellm.completion(
                    model=self.model,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    messages=[{"role": "user", "content": prompt}],
                    reasoning_effort="high",
                    allowed_openai_params=["reasoning_effort"],
                    # Bound a hung/stalled request: without an explicit
                    # timeout litellm waits up to 6000s, and one stuck row
                    # holds the whole parallel eval hostage.
                    timeout=2400,
                    num_retries=1,
                )
                content = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error calling LLM: {e}")
                outputs = [tc.get("input", []) for tc in test_cases]
                span.set_attribute("num_predictions", len(outputs))
                return outputs

            # Strip any <think>...</think> reasoning block defensively.
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

            code = self._extract_code(content)
            predictions = self._extract_predictions(content, len(test_cases))

            outputs = []

            # If code found and train pairs exist, try verification
            if code and train_cases:
                transform_fn = self._safe_exec_code(code)
                if transform_fn is not None and self._verify_transform(transform_fn, train_cases):
                    for i, tc in enumerate(test_cases):
                        test_input = tc.get("input", [])
                        result = self._safe_call(transform_fn, test_input)
                        if result is not None and self._is_valid_grid(result):
                            outputs.append(result)
                        elif i < len(predictions) and self._is_valid_grid(predictions[i]):
                            outputs.append(predictions[i])
                        else:
                            outputs.append(test_input)
                    span.set_attribute("num_predictions", len(outputs))
                    span.set_attribute("used_verified_code", True)
                    return outputs

            # Fallback: direct predictions
            for i, tc in enumerate(test_cases):
                if i < len(predictions) and self._is_valid_grid(predictions[i]):
                    outputs.append(predictions[i])
                else:
                    outputs.append(tc.get("input", []))

            span.set_attribute("num_predictions", len(outputs))
            span.set_attribute("used_verified_code", False)
            return outputs

    def _build_prompt(self, train_cases, test_cases):
        prompt = (
            "You are an expert at solving ARC-AGI visual reasoning puzzles.\n"
            "Grid cells use digits 0-9 as colors (0 is usually background). "
            "Output dimensions may differ from input.\n"
            "Find the ONE rule mapping every demo input to its output.\n\n"
        )
        prompt += "Demonstrations:\n"
        for i, case in enumerate(train_cases):
            prompt += f"Pair {i+1}:\n"
            prompt += f"Input: {json.dumps(case.get('input'))}\n"
            prompt += f"Output: {json.dumps(case.get('output'))}\n\n"

        prompt += "Test inputs:\n"
        for i, case in enumerate(test_cases):
            prompt += f"Test {i+1}: {json.dumps(case.get('input', []))}\n"

        prompt += (
            "\nWrite a Python function `transform(grid)` where grid is a 2D list of ints "
            "and the function returns a 2D list of ints. It must reproduce every demo output exactly.\n"
            "Also output your predicted output grid(s).\n\n"
            "Format:\n"
            "```python\ndef transform(grid):\n    ...\n```\n"
            "```json\n[[...]]\n```\n"
            "The json block contains one predicted grid (2D int array) per test input.\n"
            "Mentally verify your function reproduces every demo output exactly."
        )
        return prompt

    def _extract_code(self, content):
        # Prefer python code blocks
        m = re.search(r'```python\s*(.*?)\s*```', content, flags=re.DOTALL)
        if m:
            return m.group(1)
        # Try any non-json code block
        for m in re.finditer(r'```(\w*)\s*(.*?)\s*```', content, flags=re.DOTALL):
            lang = m.group(1).lower()
            if lang and lang != 'json':
                return m.group(2)
        return None

    def _extract_predictions(self, content, num_expected):
        # Try json blocks first
        for m in re.finditer(r'```json\s*(.*?)\s*```', content, flags=re.DOTALL | re.IGNORECASE):
            try:
                preds = json.loads(m.group(1))
                normalized = self._normalize_predictions(preds, num_expected)
                if normalized:
                    return normalized
            except json.JSONDecodeError:
                continue

        # Remove python code blocks to avoid parsing code as JSON
        remaining = re.sub(r'```python\s*.*?\s*```', '', content, flags=re.DOTALL)
        # Remove other non-json code blocks
        remaining = re.sub(r'```[a-zA-Z]+\s*.*?\s*```', '', remaining, flags=re.DOTALL)

        # Try direct parse
        try:
            preds = json.loads(remaining.strip())
            normalized = self._normalize_predictions(preds, num_expected)
            if normalized:
                return normalized
        except json.JSONDecodeError:
            pass

        # Scan for 2D JSON arrays
        predictions = []
        for m in re.finditer(r'\[\s*\[.*?\]\s*\]', remaining, flags=re.DOTALL):
            try:
                pred = json.loads(m.group())
                if self._is_valid_grid(pred):
                    predictions.append(pred)
                    if len(predictions) >= num_expected:
                        break
            except json.JSONDecodeError:
                continue

        return predictions

    def _normalize_predictions(self, preds, num_expected):
        if not isinstance(preds, list):
            return []
        # 3D array (list of grids)?
        if len(preds) > 0 and isinstance(preds[0], list) and len(preds[0]) > 0 and isinstance(preds[0][0], list):
            return [p for p in preds if self._is_valid_grid(p)]
        # 2D array (single grid)?
        if self._is_valid_grid(preds):
            return [preds]
        # Flat list of grids?
        result = []
        for p in preds:
            if self._is_valid_grid(p):
                result.append(p)
        return result

    def _is_valid_grid(self, grid):
        if not isinstance(grid, list) or len(grid) == 0:
            return False
        if not all(isinstance(row, list) and len(row) > 0 for row in grid):
            return False
        return True

    def _safe_exec_code(self, code):
        if len(code) > 10000:
            return None
        try:
            namespace = {'__builtins__': _SAFE_BUILTINS}
            exec(code, namespace)
            return namespace.get('transform')
        except Exception as e:
            print(f"Code exec failed: {e}")
            return None

    def _verify_transform(self, transform_fn, train_cases):
        if not callable(transform_fn):
            return False
        for case in train_cases:
            result = self._safe_call(transform_fn, case.get('input', []))
            expected = case.get('output', [])
            if result is None or not self._grids_equal(result, expected):
                return False
        return True

    def _safe_call(self, func, *args, timeout=10):
        try:
            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(timeout)
            try:
                return func(*args)
            except _TransformTimeout:
                return None
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        except ValueError:
            # Not in main thread -- call directly
            try:
                return func(*args)
            except Exception:
                return None
        except Exception:
            return None

    def _grids_equal(self, a, b):
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
                if int(ca) != int(cb):
                    return False
        return True

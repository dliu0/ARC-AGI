import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Set up tracing
provider = TracerProvider()
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

            # --- Phase 1: Direct predictions (per-test-case, focused prompt) ---
            direct_predictions = self._get_direct_predictions(train_cases, test_cases)

            # --- Phase 2: Transform code (per-task, focused prompt) ---
            verified_transform = None
            if train_cases:
                verified_transform = self._get_verified_transform(train_cases)

            # --- Merge: use verified code output where available, else direct prediction ---
            outputs = []
            for i, tc in enumerate(test_cases):
                test_input = tc.get("input", [])
                if verified_transform is not None:
                    result = self._safe_call(verified_transform, test_input)
                    if result is not None and self._is_valid_grid(result):
                        outputs.append(result)
                        continue
                if i < len(direct_predictions) and self._is_valid_grid(direct_predictions[i]):
                    outputs.append(direct_predictions[i])
                else:
                    outputs.append(test_input)

            span.set_attribute("num_predictions", len(outputs))
            span.set_attribute("used_verified_code", verified_transform is not None)
            return outputs

    def _get_direct_predictions(self, train_cases, test_cases):
        """Baseline approach: per-test-case calls with focused 'output ONLY JSON' prompt."""
        prompt = (
            "You are an expert at solving ARC-AGI visual reasoning puzzles. "
            "Grid cells use digits 0-9 as colors (0 is usually background). "
            "The output grid may have different dimensions than the input. "
            "Study all demonstration pairs to find the ONE transformation rule "
            "that maps every input to its output, then apply it to the test input.\n\n"
        )
        prompt += "Demonstrations:\n"
        for i, case in enumerate(train_cases):
            prompt += f"Pair {i+1}:\n"
            prompt += f"Input: {json.dumps(case.get('input'))}\n"
            prompt += f"Output: {json.dumps(case.get('output'))}\n\n"

        predictions = []
        for test_case in test_cases:
            test_input = test_case.get("input", [])
            test_prompt = prompt + f"Test Case:\nInput: {json.dumps(test_input)}\n\n"
            test_prompt += (
                "Output ONLY a JSON array of arrays representing the output grid. "
                "No markdown, no explanation. "
                "Before responding, mentally verify your rule reproduces every demonstration output exactly."
            )
            try:
                response = litellm.completion(
                    model=self.model,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    messages=[{"role": "user", "content": test_prompt}],
                    reasoning_effort="high",
                    allowed_openai_params=["reasoning_effort"],
                    timeout=2400,
                    num_retries=1,
                )
                content = response.choices[0].message.content.strip()
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                content = re.sub(r'^```(?:json)?\s*', '', content)
                content = re.sub(r'\s*```$', '', content)
                content = content.strip()

                prediction = None
                try:
                    prediction = json.loads(content)
                except json.JSONDecodeError:
                    for m in re.finditer(r'\[\s*\[.*?\]\s*\]', content, flags=re.DOTALL):
                        try:
                            prediction = json.loads(m.group())
                            break
                        except json.JSONDecodeError:
                            continue

                if isinstance(prediction, list) and all(isinstance(row, list) for row in prediction):
                    predictions.append(prediction)
                else:
                    predictions.append(test_input)
            except Exception as e:
                print(f"Error calling LLM or parsing response: {e}")
                predictions.append(test_input)

        return predictions

    def _get_verified_transform(self, train_cases):
        """Separate per-task call asking ONLY for transform code. Returns verified transform_fn or None."""
        prompt = (
            "You are an expert at ARC-AGI visual reasoning puzzles. "
            "Grid cells use digits 0-9 as colors (0 is background). "
            "Find the ONE transformation rule mapping every demo input to its output.\n\n"
            "Demonstrations:\n"
        )
        for i, case in enumerate(train_cases):
            prompt += f"Pair {i+1}:\n"
            prompt += f"Input: {json.dumps(case.get('input'))}\n"
            prompt += f"Output: {json.dumps(case.get('output'))}\n\n"

        prompt += (
            "Write a Python function `transform(grid)` that takes a 2D list of ints and returns a 2D list of ints. "
            "It must reproduce EVERY demonstration output exactly when applied to the corresponding input.\n"
            "Output ONLY the function in a python code block:\n"
            "```python\ndef transform(grid):\n    ...\n```"
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            response = litellm.completion(
                model=self.model,
                api_base=self.api_base,
                api_key=self.api_key,
                messages=messages,
                reasoning_effort="high",
                allowed_openai_params=["reasoning_effort"],
                timeout=2400,
                num_retries=1,
            )
            content = response.choices[0].message.content.strip()
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        except Exception as e:
            print(f"Error calling LLM for transform code: {e}")
            return None

        # Try all code blocks through verification; keep first executable for repair feedback
        code_blocks = self._extract_all_code(content)
        if not code_blocks:
            return None
        transform_fn = None
        exec_err = None
        for code in code_blocks:
            fn, err = self._safe_exec_code(code)
            if fn is not None:
                if self._verify_transform(fn, train_cases):
                    return fn
                if transform_fn is None:
                    transform_fn = fn
                    exec_err = err
        if transform_fn is None:
            _, exec_err = self._safe_exec_code(code_blocks[0])

        # --- Single repair attempt with error feedback ---
        feedback = self._build_error_feedback(transform_fn, exec_err, train_cases)
        if feedback is None:
            return None
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": feedback})
        try:
            response = litellm.completion(
                model=self.model,
                api_base=self.api_base,
                api_key=self.api_key,
                messages=messages,
                reasoning_effort="high",
                allowed_openai_params=["reasoning_effort"],
                timeout=2400,
                num_retries=1,
            )
            content = response.choices[0].message.content.strip()
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        except Exception as e:
            print(f"Error calling LLM for repair: {e}")
            return None

        code_blocks = self._extract_all_code(content)
        if not code_blocks:
            return None
        for code in code_blocks:
            fn, _ = self._safe_exec_code(code)
            if fn is not None and self._verify_transform(fn, train_cases):
                return fn
        return None

    def _build_error_feedback(self, transform_fn, exec_err, train_cases):
        """Build diagnostic error feedback for the repair call."""
        if transform_fn is None:
            return (
                f"Your code failed to execute:\n{exec_err}\n\n"
                "Fix the error and output ONLY the corrected function in a python code block:\n"
                "```python\ndef transform(grid):\n    ...\n```"
            )
        failed = []
        for i, case in enumerate(train_cases):
            result = self._safe_call(transform_fn, case.get('input', []))
            expected = case.get('output', [])
            if result is None or not self._grids_equal(result, expected):
                failed.append((i, case, result, expected))
        if not failed:
            return None
        feedback = "Your transform function does not reproduce all demonstration outputs. It failed on:\n"
        for i, case, result, expected in failed[:3]:
            exp_h = len(expected)
            exp_w = len(expected[0]) if expected else 0
            got_h = len(result) if result else 0
            got_w = len(result[0]) if result and result[0] else 0
            feedback += f"Pair {i+1}: expected {exp_h}x{exp_w}, got {got_h}x{got_w}.\n"
            exp_colors = set()
            for row in expected:
                exp_colors.update(row)
            got_colors = set()
            if result:
                for row in result:
                    got_colors.update(row)
            if exp_colors != got_colors:
                feedback += f"  Expected colors {sorted(exp_colors)}, got colors {sorted(got_colors)}.\n"
            if result and exp_h == got_h and exp_w == got_w:
                diffs = []
                for r, (er, gr) in enumerate(zip(expected, result)):
                    for c, (ev, gv) in enumerate(zip(er, gr)):
                        if int(ev) != int(gv):
                            diffs.append((r, c, ev, gv))
                if diffs:
                    if len(diffs) <= 20:
                        feedback += f"  {len(diffs)} cells differ (row,col,expected,got):\n"
                        for r, c, ev, gv in diffs:
                            feedback += f"    [{r},{c}] expected {ev} got {gv}\n"
                    else:
                        feedback += f"  {len(diffs)} cells differ (too many to list).\n"
            elif result and exp_h * exp_w <= 225:
                feedback += f"  Expected: {json.dumps(expected)}\n  Got: {json.dumps(result)}\n"
        feedback += (
            "Fix the function so it reproduces EVERY demonstration output exactly. "
            "Output ONLY the corrected function in a python code block:\n"
            "```python\ndef transform(grid):\n    ...\n```"
        )
        return feedback

    def _extract_code(self, content):
        m = re.search(r'```python\s*(.*?)\s*```', content, flags=re.DOTALL)
        if m:
            return m.group(1)
        for m in re.finditer(r'```(\w*)\s*(.*?)\s*```', content, flags=re.DOTALL):
            lang = m.group(1).lower()
            if lang and lang != 'json':
                return m.group(2)
        return None

    def _extract_all_code(self, content):
        """Extract all python code blocks from content, in order of appearance."""
        blocks = []
        for m in re.finditer(r'```python\s*(.*?)\s*```', content, flags=re.DOTALL):
            blocks.append(m.group(1))
        if not blocks:
            for m in re.finditer(r'```(\w*)\s*(.*?)\s*```', content, flags=re.DOTALL):
                lang = m.group(1).lower()
                if lang and lang != 'json':
                    blocks.append(m.group(2))
        return blocks

    def _is_valid_grid(self, grid):
        if not isinstance(grid, list) or len(grid) == 0:
            return False
        if not all(isinstance(row, list) and len(row) > 0 for row in grid):
            return False
        return True

    def _safe_exec_code(self, code):
        if len(code) > 10000:
            return None, "Code too long"
        try:
            namespace = {'__builtins__': _SAFE_BUILTINS}
            exec(code, namespace)
            return namespace.get('transform'), None
        except Exception as e:
            print(f"Code exec failed: {e}")
            return None, str(e)

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

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
import litellm

class ARCPipeline:
    def __init__(self):
        # Solver LM: DeepSeek-V4-Flash on GMI Cloud (reasoning=high set on the
        # call below). In this experiment the optimizer (GLM-5.2) and this solver
        # both run on GMI, so we pass the GMI endpoint + key explicitly (the
        # proven GMI-as-OpenAI pattern) rather than relying on OPENAI_* env.
        self.model = "openai/deepseek-ai/DeepSeek-V4-Flash"
        self.api_base = "https://api.gmi-serving.com/v1"
        self.api_key = os.environ.get("GMI_CLOUD_API_KEY") or os.environ.get("GMI_API_KEY")

    # --- Hang guard -------------------------------------------------------
    # GMI occasionally "hangs" a request: zero bytes until its gateway kills
    # the connection at ~20 min. Measured on job_3362095d1463: ~4.7% of rows,
    # enough that nearly EVERY 50-row parallel eval walls at the straggler's
    # timeout (evals took 20-40 min with a 2-6 min median row). Streaming
    # makes hangs detectable: GMI streams reasoning deltas continuously
    # (measured max inter-chunk gap ~3s), so READ_GAP_TIMEOUT_S of total
    # silence is an unambiguous hang -> abort fast and retry, instead of
    # waiting for the gateway. httpx applies `timeout` per READ on a stream
    # (verified: a 31s stream survives timeout=15), so long generations are
    # unaffected; TOTAL_BUDGET_S caps the row across all attempts.
    READ_GAP_TIMEOUT_S = 240
    TOTAL_BUDGET_S = 2400
    MAX_ATTEMPTS = 2

    def _complete(self, messages):
        """Streaming completion with hang detection; returns content text."""
        import time as _time

        start = _time.monotonic()
        last_err = None
        for _attempt in range(self.MAX_ATTEMPTS):
            if _time.monotonic() - start > self.TOTAL_BUDGET_S - self.READ_GAP_TIMEOUT_S:
                break
            try:
                stream = litellm.completion(
                    model=self.model,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    messages=messages,
                    reasoning_effort="high",
                    allowed_openai_params=["reasoning_effort"],
                    stream=True,
                    # Per-read gap cap on a stream (NOT total duration): only
                    # trips when the connection goes fully silent (real hang).
                    timeout=self.READ_GAP_TIMEOUT_S,
                )
                parts = []
                for chunk in stream:
                    if _time.monotonic() - start > self.TOTAL_BUDGET_S:
                        raise TimeoutError(
                            f"row exceeded total budget {self.TOTAL_BUDGET_S}s"
                        )
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        if delta is not None and getattr(delta, "content", None):
                            parts.append(delta.content)
                return "".join(parts)
            except Exception as exc:  # noqa: BLE001 — hang/gap/transient
                last_err = exc
        raise last_err if last_err else RuntimeError("completion failed")

    def __call__(self, train: list = None, test: list = None, task_id: str = "unknown", **kwargs) -> list:
        with tracer.start_as_current_span("arc_predict") as span:
            span.set_attribute("task_id", task_id)
            
            train_cases = train or []
            test_cases = test or []
            
            # Format the demonstration pairs
            prompt = "You are an expert at visual and mathematical reasoning. You will be given a few demonstration pairs of input and output grids. You must deduce the abstract transformation rule and apply it to the final test input grid.\n\n"
            
            prompt += "Demonstrations:\n"
            for i, case in enumerate(train_cases):
                prompt += f"Pair {i+1}:\n"
                prompt += f"Input: {json.dumps(case.get('input'))}\n"
                prompt += f"Output: {json.dumps(case.get('output'))}\n\n"
            
            outputs = []
            for test_case in test_cases:
                test_input = test_case.get("input", [])
                
                test_prompt = prompt + f"Test Case:\nInput: {json.dumps(test_input)}\n\n"
                test_prompt += "Please output ONLY a valid JSON array of arrays (representing the output grid) and nothing else. No markdown formatting or explanation."
                
                try:
                    content = self._complete(
                        [{"role": "user", "content": test_prompt}]
                    ).strip()

                    # Try to parse the content as a JSON array. Strip any
                    # <think>...</think> reasoning block defensively.
                    import re
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

                    if content.startswith("```json"):
                        content = content.split("```json")[1]
                    if content.startswith("```"):
                        content = content.split("```")[1]
                    if content.endswith("```"):
                        content = content.rsplit("```", 1)[0]
                        
                    content = content.strip()
                    prediction = json.loads(content)
                    
                    if isinstance(prediction, list) and all(isinstance(row, list) for row in prediction):
                        outputs.append(prediction)
                    else:
                        outputs.append(test_input)  # Fallback
                except Exception as e:
                    print(f"Error calling LLM or parsing response: {e}")
                    outputs.append(test_input)  # Fallback on error
            
            span.set_attribute("num_predictions", len(outputs))
            return outputs

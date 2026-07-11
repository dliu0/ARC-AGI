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

    def __call__(self, train: list = None, test: list = None, task_id: str = "unknown", **kwargs) -> list:
        with tracer.start_as_current_span("arc_predict") as span:
            span.set_attribute("task_id", task_id)
            
            train_cases = train or []
            test_cases = test or []
            
            # Format the demonstration pairs
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
            
            outputs = []
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
                        # Bound a hung/stalled request: without an explicit
                        # timeout litellm waits up to 6000s, and one stuck row
                        # holds the whole parallel eval hostage.
                        timeout=2400,
                        num_retries=1,
                    )
                    content = response.choices[0].message.content.strip()
                    import re

                    # Strip any <think>...</think> reasoning block defensively.
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

                    # Strip code fences if present.
                    content = re.sub(r'^```(?:json)?\s*', '', content)
                    content = re.sub(r'\s*```$', '', content)
                    content = content.strip()

                    # Try direct parse first; if that fails, scan for the
                    # first valid JSON array-of-arrays in the text.
                    prediction = None
                    try:
                        prediction = json.loads(content)
                    except json.JSONDecodeError:
                        # Scan for JSON arrays: find all [...] substrings
                        for m in re.finditer(r'\[\s*\[.*?\]\s*\]', content, flags=re.DOTALL):
                            try:
                                prediction = json.loads(m.group())
                                break
                            except json.JSONDecodeError:
                                continue
                    
                    if isinstance(prediction, list) and all(isinstance(row, list) for row in prediction):
                        outputs.append(prediction)
                    else:
                        outputs.append(test_input)  # Fallback
                except Exception as e:
                    print(f"Error calling LLM or parsing response: {e}")
                    outputs.append(test_input)  # Fallback on error
            
            span.set_attribute("num_predictions", len(outputs))
            return outputs

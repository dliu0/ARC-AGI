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
                "You are an expert at solving Abstraction and Reasoning Corpus (ARC) tasks.\n\n"
                "Each task provides demonstration input->output grid pairs (each grid is a 2D array of integers 0-9 representing colors). "
                "Infer the single abstract transformation rule that maps EVERY train input to its train output, then apply that same rule to the test input.\n\n"
                "Before answering, reason through these steps:\n"
                "1. DIMENSIONS: For each train pair, compare input dimensions (rows x cols) to output dimensions. "
                "Find the consistent relationship (same size, scaled by a factor, cropped, padded, or derived from content). "
                "The test output MUST follow this same dimensional relationship.\n"
                "2. COLORS: Identify the background color (most frequent) and note any color mappings or substitutions between input and output.\n"
                "3. RULE: State the transformation in one sentence. It must explain how every train input becomes its train output.\n"
                "4. VERIFY: Mentally apply your rule to each train input and check it reproduces the train output exactly. If any pair fails, revise your rule.\n"
                "5. APPLY: Apply the verified rule to the test input to produce the output grid.\n\n"
                "Demonstrations:\n"
            )
            for i, case in enumerate(train_cases):
                prompt += f"Pair {i+1}:\n"
                prompt += f"Input: {json.dumps(case.get('input'))}\n"
                prompt += f"Output: {json.dumps(case.get('output'))}\n\n"
            
            outputs = []
            for test_case in test_cases:
                test_input = test_case.get("input", [])
                
                test_prompt = prompt + f"Test Case:\nInput: {json.dumps(test_input)}\n\n"
                test_prompt += "Reason step by step, then output the final answer as a JSON array of arrays (the output grid). Put the JSON array on its own line after your reasoning."
                
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

                    # Strip reasoning blocks and markdown code fences.
                    import re
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                    content = re.sub(r'```(?:json)?\s*', '', content).strip()

                    # Parse the output grid. The model may include reasoning
                    # text before the final JSON array, so first try a direct
                    # parse, then fall back to extracting the last balanced
                    # JSON array from the text.
                    prediction = None
                    try:
                        prediction = json.loads(content)
                    except json.JSONDecodeError:
                        depth = 0
                        start_idx = None
                        for i, ch in enumerate(content):
                            if ch == '[':
                                if depth == 0:
                                    start_idx = i
                                depth += 1
                            elif ch == ']':
                                depth -= 1
                                if depth == 0 and start_idx is not None:
                                    candidate = content[start_idx:i + 1]
                                    try:
                                        parsed = json.loads(candidate)
                                        if isinstance(parsed, list):
                                            prediction = parsed
                                    except json.JSONDecodeError:
                                        pass
                                    start_idx = None
                        if prediction is None:
                            raise ValueError("No valid JSON array found in response")

                    if isinstance(prediction, list) and all(isinstance(row, list) for row in prediction):
                        outputs.append(prediction)
                    else:
                        outputs.append(test_input)  # Fallback
                except Exception as e:
                    print(f"Error calling LLM or parsing response: {e}")
                    outputs.append(test_input)  # Fallback on error
            
            span.set_attribute("num_predictions", len(outputs))
            return outputs

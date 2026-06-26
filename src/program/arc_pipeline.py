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
        self.model = "openai/gpt-5.4-mini"

    def __call__(self, train: list = None, test: list = None, task_id: str = "unknown", **kwargs) -> list:
        with tracer.start_as_current_span("arc_predict") as span:
            span.set_attribute("task_id", task_id)
            
            train_cases = train or []
            test_cases = test or []
            prompt = (
                "Infer the ARC rule from all demonstrations, then solve every test input.\n"
                "First determine the exact output canvas size from the demonstrations.\n"
                "Verify the rule against every demo pair before answering.\n"
                "Do not copy the test input unless the demonstrations prove the output is identical.\n"
                "Return only valid JSON in the exact form {\"predictions\":[grid1, grid2,...]}.\n"
                "Each grid must be a 2D list of integers, one grid per test input.\n\n"
            )
            task_payload = {"train": train_cases, "test": test_cases}
            prompt += "Task:\n" + json.dumps(task_payload, separators=(",", ":"))

            fallback_outputs = [case.get("input", []) for case in test_cases]

            try:
                response = litellm.completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=384,
                )
                content = response.choices[0].message.content.strip()

                if content.startswith("```json"):
                    content = content.split("```json", 1)[1]
                if content.startswith("```"):
                    content = content.split("```", 1)[1]
                if content.endswith("```"):
                    content = content.rsplit("```", 1)[0]

                content = content.strip()
                prediction = json.loads(content)
                if isinstance(prediction, dict):
                    prediction = prediction.get("predictions", prediction.get("outputs"))

                if (
                    isinstance(prediction, list)
                    and len(prediction) == len(test_cases)
                    and all(isinstance(grid, list) and all(isinstance(row, list) for row in grid) for grid in prediction)
                ):
                    outputs = prediction
                else:
                    outputs = fallback_outputs
            except Exception as e:
                print(f"Error calling LLM or parsing response: {e}")
                outputs = fallback_outputs
            
            span.set_attribute("num_predictions", len(outputs))
            return outputs

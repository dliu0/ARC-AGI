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
                    response = litellm.completion(
                        model=self.model,
                        messages=[{"role": "user", "content": test_prompt}],
                        temperature=0.0
                    )
                    content = response.choices[0].message.content.strip()
                    
                    # Try to parse the content as a JSON array
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

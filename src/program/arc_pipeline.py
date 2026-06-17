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

class ARCPipeline:
    def __init__(self):
        pass

    def __call__(self, task_data: dict) -> list:
        with tracer.start_as_current_span("arc_predict") as span:
            span.set_attribute("task_id", task_data.get("task_id", "unknown"))
            # Dummy prediction: return the first test input as the output
            test_cases = task_data.get("test", [])
            outputs = []
            for test_case in test_cases:
                outputs.append(test_case.get("input", []))
            
            span.set_attribute("num_predictions", len(outputs))
            return outputs

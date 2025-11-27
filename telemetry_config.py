"""
OpenTelemetry configuration for tracing and logging
"""
import os
import logging
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTPSpanExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter as HTTPMetricExporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def setup_telemetry(
    service_name: str = "doctor-vinmec-agent",
    enable_console_export: bool = True,
    enable_otlp_export: bool = False,
    otlp_endpoint: str = None
):
    """
    Setup OpenTelemetry tracing and metrics
    
    Args:
        service_name: Name of the service
        enable_console_export: Export to console (for debugging)
        enable_otlp_export: Export to OTLP collector (for production)
        otlp_endpoint: OTLP collector endpoint (e.g., "localhost:4317")
    """
    
    # Create resource with service name
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0"
    })
    
    # ==================== TRACING ====================
    
    # Create tracer provider
    tracer_provider = TracerProvider(resource=resource)
    
    # Add console exporter for debugging
    if enable_console_export:
        console_exporter = ConsoleSpanExporter()
        tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))
    
    # Add OTLP exporter for production
    if enable_otlp_export and otlp_endpoint:
        # Ensure endpoint has the correct path for traces
        trace_endpoint = otlp_endpoint if otlp_endpoint.endswith('/v1/traces') else f"{otlp_endpoint}/v1/traces"
        otlp_exporter = HTTPSpanExporter(endpoint=trace_endpoint)
        tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    
    # Set global tracer provider
    trace.set_tracer_provider(tracer_provider)
    
    logger.info(f"✅ Tracing initialized for service: {service_name}")
    
    # ==================== METRICS ====================
    
    # Create metric readers
    metric_readers = []
    
    if enable_console_export:
        console_metric_reader = PeriodicExportingMetricReader(
            ConsoleMetricExporter(),
            export_interval_millis=60000  # Export every 60 seconds
        )
        metric_readers.append(console_metric_reader)
    
    if enable_otlp_export and otlp_endpoint:
        # Ensure endpoint has the correct path for metrics
        metric_endpoint = otlp_endpoint if otlp_endpoint.endswith('/v1/metrics') else f"{otlp_endpoint}/v1/metrics"
        otlp_metric_reader = PeriodicExportingMetricReader(
            HTTPMetricExporter(endpoint=metric_endpoint),
            export_interval_millis=60000
        )
        metric_readers.append(otlp_metric_reader)
    
    # Create meter provider
    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=metric_readers
    )
    
    # Set global meter provider
    metrics.set_meter_provider(meter_provider)
    
    logger.info(f"✅ Metrics initialized for service: {service_name}")
    
    # ==================== INSTRUMENTATION ====================
    
    # Instrument logging
    LoggingInstrumentor().instrument(set_logging_format=True)
    
    # Instrument HTTP requests
    RequestsInstrumentor().instrument()
    
    logger.info("✅ Auto-instrumentation enabled")
    
    return tracer_provider, meter_provider


def get_tracer(name: str = __name__):
    """Get a tracer instance"""
    return trace.get_tracer(name)


def get_meter(name: str = __name__):
    """Get a meter instance"""
    return metrics.get_meter(name)


# Initialize telemetry (can be configured via environment variables)
if os.getenv("ENABLE_TELEMETRY", "true").lower() == "true":
    setup_telemetry(
        service_name=os.getenv("service_name", "doctor-vinmec-agent"),
        enable_console_export=os.getenv("CONSOLE_EXPORT", "false").lower() == "true",
        enable_otlp_export=os.getenv("OTLP_EXPORT", "true").lower() == "true",
        otlp_endpoint=os.getenv("OTLP_ENDPOINT", "http://localhost:4318")
    )
    logger.info("🔍 OpenTelemetry enabled")
else:
    logger.info("⚠️ OpenTelemetry disabled")


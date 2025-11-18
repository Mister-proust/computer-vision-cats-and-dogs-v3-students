from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
import os

inference_time_histogram = Histogram(
    'cv_inference_time_seconds',
    'Temps d\'inférence en secondes'
)

def track_inference_time(inference_time_ms: float):
    """Enregistre le temps d'inférence"""
    inference_time_histogram.observe(inference_time_ms / 1000)

database_status = Gauge(
    'cv_database_connected',
    'Database connection status (1=connected, 0=disconnected)'
)
def setup_prometheus(app):
    if os.getenv('ENABLE_PROMETHEUS', 'false').lower() == 'true':
        Instrumentator().instrument(app).expose(app, endpoint="/metrics")
        print("✅ Prometheus metrics enabled at /metrics")
    else:
        print("ℹ️  Prometheus metrics disabled")
       

def update_db_status(is_connected: bool):
    database_status.set(1 if is_connected else 0)


feedback_counter = Counter(
    "cv_user_feedback_total",
    "Nombre total de feedbacks utilisateurs",
    ["feedback_type"]
)


def track_feedback(feedback_type: str):
    feedback_counter.labels(feedback_type=feedback_type).inc()



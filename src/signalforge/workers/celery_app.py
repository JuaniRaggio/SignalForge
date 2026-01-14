"""Celery application configuration."""

from celery import Celery

from signalforge.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "signalforge",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["signalforge.ingestion.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,
    task_soft_time_limit=270,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

celery_app.conf.beat_schedule = {
    "ingest-daily-prices": {
        "task": "signalforge.ingestion.tasks.ingest_daily_prices",
        "schedule": 3600.0,
        "options": {"queue": "prices"},
    },
    "scrape-news-rss": {
        "task": "signalforge.ingestion.tasks.scrape_news_rss",
        "schedule": 900.0,
        "options": {"queue": "news"},
    },
}

"""Celery application configuration."""

from celery import Celery

from signalforge.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "signalforge",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "signalforge.ingestion.tasks",
        "signalforge.workers.event_tasks",
        "signalforge.workers.orderflow_tasks",
        "signalforge.paper_trading.tasks",
    ],
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
    "sync-earnings-calendar": {
        "task": "signalforge.events.tasks.sync_earnings_calendar",
        "schedule": 21600.0,
        "options": {"queue": "events"},
    },
    "sync-fed-schedule": {
        "task": "signalforge.events.tasks.sync_fed_schedule",
        "schedule": 86400.0,
        "options": {"queue": "events"},
    },
    "sync-economic-calendar": {
        "task": "signalforge.events.tasks.sync_economic_calendar",
        "schedule": 21600.0,
        "options": {"queue": "events"},
    },
    "check-upcoming-events": {
        "task": "signalforge.events.tasks.check_upcoming_events",
        "schedule": 3600.0,
        "options": {"queue": "events"},
    },
    "detect-flow-anomalies": {
        "task": "signalforge.orderflow.tasks.detect_flow_anomalies",
        "schedule": 900.0,
        "options": {"queue": "orderflow"},
    },
    "aggregate-daily-flow": {
        "task": "signalforge.orderflow.tasks.aggregate_daily_flow",
        "schedule": 3600.0,
        "options": {"queue": "orderflow"},
    },
    "create-daily-snapshots": {
        "task": "paper_trading.create_daily_snapshots",
        "schedule": 86400.0,  # Daily at market close (configure time separately)
        "options": {"queue": "paper_trading"},
    },
    "update-position-prices": {
        "task": "paper_trading.update_position_prices",
        "schedule": 60.0,  # Every minute during market hours
        "options": {"queue": "paper_trading"},
    },
    "check-pending-orders": {
        "task": "paper_trading.check_pending_orders",
        "schedule": 60.0,  # Every minute during market hours
        "options": {"queue": "paper_trading"},
    },
    "update-competition-statuses": {
        "task": "paper_trading.update_competition_statuses",
        "schedule": 3600.0,  # Every hour
        "options": {"queue": "paper_trading"},
    },
    "finalize-completed-competitions": {
        "task": "paper_trading.finalize_completed_competitions",
        "schedule": 3600.0,  # Every hour
        "options": {"queue": "paper_trading"},
    },
    "calculate-competition-standings": {
        "task": "paper_trading.calculate_competition_standings",
        "schedule": 900.0,  # Every 15 minutes during market hours
        "options": {"queue": "paper_trading"},
    },
}

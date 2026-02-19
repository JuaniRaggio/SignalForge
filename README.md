# SignalForge

**AI-powered financial signal processing and market intelligence platform.**

SignalForge combines ensemble machine learning, NLP-driven document analysis, and real-time data pipelines to deliver actionable trading insights. Built for prosumer traders who need professional-grade analysis without the cost of institutional terminals.

---

## Core Capabilities

- **Ensemble ML Forecasting** -- Six model types (ARIMA, LSTM, Prophet, GARCH, Quantile Regression, Baseline) combined through a weighted ensemble for robust price predictions with confidence intervals.

- **Financial NLP Pipeline** -- PDF parsing with OCR fallback, sentiment analysis, price target extraction, contradiction detection, urgency scoring, and sector-specific intelligence across six industry verticals.

- **Personalized Signal Feed** -- Recommendation engine with collaborative, content-based, and hybrid algorithms. Adapts content complexity to four user tiers (casual, informed, active, quant).

- **Paper Trading & Competitions** -- Full portfolio simulation with realistic order execution, tournament-style competitions, live leaderboards, and detailed performance analytics (Sharpe, Sortino, max drawdown).

- **Risk Management** -- Kelly criterion position sizing, Value at Risk calculations, concentration monitoring, and correlation analysis.

- **Real-Time Alerts** -- Multi-channel delivery (WebSocket, email, SMS) with throttling and deduplication.

- **Explainability** -- SHAP-based model explanations translated into trader-friendly language, with per-feature impact breakdowns.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI, Uvicorn, Pydantic v2 |
| Database | PostgreSQL + TimescaleDB, pgvector, SQLAlchemy 2.0 (async) |
| Cache & Messaging | Redis, Celery |
| ML | PyTorch, scikit-learn, statsmodels, Prophet, ONNX Runtime |
| NLP | Hugging Face Transformers, spaCy, sentence-transformers |
| Data Processing | Polars, NumPy, SciPy |
| Experiment Tracking | MLflow |
| Observability | structlog (JSON), Prometheus |
| Auth | JWT (python-jose), bcrypt |

---

## Project Structure

```
src/signalforge/
  api/              REST API routes, middleware, dependencies
  ml/               Models, training, inference, backtesting, optimization
  nlp/              Document processing, sentiment, NER, sector analysis
  paper_trading/    Execution engine, portfolios, competitions, leaderboards
  recommendation/   Feed generation, ranking algorithms, user profiling
  risk/             Position sizing, VaR, correlation, concentration
  alerts/           Multi-channel alert delivery and throttling
  adaptation/       Content personalization by user experience level
  explainability/   SHAP explanations and human-readable translations
  orderflow/        Dark pools, options activity, short interest
  events/           Earnings, economic releases, Fed announcements
  execution/        Liquidity scoring, slippage, spread analysis
  billing/          Subscription management and usage tracking
  core/             Config, database, redis, logging, security, metrics
  models/           SQLAlchemy ORM models
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose

### Setup

```bash
# Clone and install
git clone <repo-url> && cd SignalForge
uv sync --all-extras

# Start infrastructure
docker compose up -d

# Run migrations
uv run alembic upgrade head

# Start the API
uv run uvicorn signalforge.api.main:app --reload
```

### Running Tests

```bash
uv run pytest -v
```

### Linting & Type Checking

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
```

---

## Architecture Notes

- Fully async from API to database (asyncpg + SQLAlchemy async sessions).
- TimescaleDB hypertables for time-series price data.
- pgvector for semantic document search via sentence embeddings.
- Celery workers handle background tasks: portfolio snapshots, competition updates, price syncs.
- ONNX export and INT8 quantization for inference optimization.
- Structured JSON logging with correlation IDs for request tracing.

---

## License

Licensed under the [Apache License 2.0](LICENSE).

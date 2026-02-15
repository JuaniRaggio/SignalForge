"""API routes module."""

from signalforge.api.routes.auth import router as auth_router
from signalforge.api.routes.competition import router as competition_router
from signalforge.api.routes.dashboard import router as dashboard_router
from signalforge.api.routes.events import router as events_router
from signalforge.api.routes.execution import router as execution_router
from signalforge.api.routes.explainability import router as explainability_router
from signalforge.api.routes.leaderboard import router as leaderboard_router
from signalforge.api.routes.market import router as market_router
from signalforge.api.routes.ml import router as ml_router
from signalforge.api.routes.nlp import router as nlp_router
from signalforge.api.routes.orderflow import router as orderflow_router
from signalforge.api.routes.paper_trading import router as paper_trading_router
from signalforge.api.routes.recommendations import router as recommendations_router
from signalforge.api.routes.websocket import router as websocket_router

__all__ = [
    "auth_router",
    "competition_router",
    "dashboard_router",
    "events_router",
    "execution_router",
    "explainability_router",
    "leaderboard_router",
    "market_router",
    "ml_router",
    "nlp_router",
    "orderflow_router",
    "paper_trading_router",
    "recommendations_router",
    "websocket_router",
]

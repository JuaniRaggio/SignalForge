"""Events module for financial calendar management.

This module provides comprehensive financial event tracking and analysis including:
- Earnings reports
- Federal Reserve (FOMC) meetings
- Economic releases (CPI, NFP, GDP)
- Event impact analysis
"""

from signalforge.events.calendar import EventCalendar
from signalforge.events.earnings import EarningsTracker
from signalforge.events.economic_releases import EconomicCalendar
from signalforge.events.fed import FedTracker
from signalforge.events.schemas import (
    EarningsEvent,
    EconomicEvent,
    EventCalendarResponse,
    EventCreate,
    EventQuery,
    EventResponse,
    EventUpdate,
    FedEvent,
)

__all__ = [
    # Calendar
    "EventCalendar",
    # Earnings
    "EarningsTracker",
    "EarningsEvent",
    # Fed
    "FedTracker",
    "FedEvent",
    # Economic
    "EconomicCalendar",
    "EconomicEvent",
    # Schemas
    "EventCreate",
    "EventUpdate",
    "EventResponse",
    "EventQuery",
    "EventCalendarResponse",
]

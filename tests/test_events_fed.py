"""Tests for Federal Reserve events tracking service."""

from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.events.fed import FedTracker
from signalforge.events.schemas import FedEvent
from signalforge.models.event import Event, EventImportance, EventType


async def create_fed_event(
    db_session: AsyncSession,
    event_date: datetime,
    rate_decision: float | None = None,
    statement_summary: str | None = None,
) -> Event:
    """Helper to create a test Fed event."""
    metadata = {
        "rate_decision": rate_decision,
        "statement_summary": statement_summary,
    }

    event = Event(
        symbol=None,
        event_type=EventType.FOMC,
        event_date=event_date,
        importance=EventImportance.CRITICAL,
        title=f"FOMC Meeting - {event_date.strftime('%B %Y')}",
        description="Federal Open Market Committee meeting",
        expected_value=None,
        actual_value=rate_decision,
        metadata_json=metadata,
        source="federal_reserve",
    )
    db_session.add(event)
    await db_session.commit()
    await db_session.refresh(event)
    return event


@pytest.mark.asyncio
async def test_get_fomc_schedule_2024(db_session: AsyncSession) -> None:
    """Test retrieving FOMC schedule for 2024."""
    tracker = FedTracker(db_session)

    events = await tracker.get_fomc_schedule(year=2024)

    assert len(events) == 8  # 8 FOMC meetings in 2024
    assert all(isinstance(event, FedEvent) for event in events)
    assert all(event.importance == EventImportance.CRITICAL for event in events)
    assert all(event.event_date.year == 2024 for event in events)


@pytest.mark.asyncio
async def test_get_fomc_schedule_2025(db_session: AsyncSession) -> None:
    """Test retrieving FOMC schedule for 2025."""
    tracker = FedTracker(db_session)

    events = await tracker.get_fomc_schedule(year=2025)

    assert len(events) == 8
    assert all(event.event_date.year == 2025 for event in events)


@pytest.mark.asyncio
async def test_get_fomc_schedule_2026(db_session: AsyncSession) -> None:
    """Test retrieving FOMC schedule for 2026."""
    tracker = FedTracker(db_session)

    events = await tracker.get_fomc_schedule(year=2026)

    assert len(events) == 8
    assert all(event.event_date.year == 2026 for event in events)


@pytest.mark.asyncio
async def test_get_fomc_schedule_current_year(db_session: AsyncSession) -> None:
    """Test retrieving FOMC schedule for current year."""
    tracker = FedTracker(db_session)

    events = await tracker.get_fomc_schedule()  # No year specified

    current_year = datetime.now(UTC).year
    if current_year in [2024, 2025, 2026]:
        assert len(events) > 0
        assert all(event.event_date.year == current_year for event in events)


@pytest.mark.asyncio
async def test_get_fomc_schedule_unavailable_year(db_session: AsyncSession) -> None:
    """Test retrieving FOMC schedule for unavailable year."""
    tracker = FedTracker(db_session)

    events = await tracker.get_fomc_schedule(year=2030)

    assert len(events) == 0


@pytest.mark.asyncio
async def test_get_fomc_schedule_with_stored_events(db_session: AsyncSession) -> None:
    """Test retrieving FOMC schedule when events are stored in database."""
    tracker = FedTracker(db_session)

    # Create a stored event for 2024
    test_date = datetime(2024, 1, 31, tzinfo=UTC)
    await create_fed_event(
        db_session,
        event_date=test_date,
        rate_decision=5.25,
        statement_summary="Rate maintained at target range",
    )

    events = await tracker.get_fomc_schedule(year=2024)

    assert len(events) > 0
    # Should contain the stored event with rate decision
    stored_event = next((e for e in events if e.event_date == test_date), None)
    assert stored_event is not None
    assert stored_event.rate_decision == 5.25


@pytest.mark.asyncio
async def test_get_next_fomc_meeting(db_session: AsyncSession) -> None:
    """Test retrieving next FOMC meeting."""
    tracker = FedTracker(db_session)

    next_meeting = await tracker.get_next_fomc_meeting()

    if next_meeting:
        assert isinstance(next_meeting, FedEvent)
        assert next_meeting.event_date > datetime.now(UTC)
        assert next_meeting.importance == EventImportance.CRITICAL


@pytest.mark.asyncio
async def test_parse_fed_statement_hawkish(db_session: AsyncSession) -> None:
    """Test parsing a hawkish Fed statement."""
    tracker = FedTracker(db_session)

    statement = """
    The Federal Reserve is committed to fighting inflation. Recent data shows
    persistent elevated inflation, requiring us to raise interest rates to
    tighten monetary policy and maintain price stability.
    """

    parsed = tracker.parse_fed_statement(statement)

    assert parsed["sentiment"] == "hawkish"
    assert parsed["rate_action"] in ["increase", None]
    assert "inflation concerns" in parsed["key_phrases"]
    assert parsed["hawkish_count"] > parsed["dovish_count"]


@pytest.mark.asyncio
async def test_parse_fed_statement_dovish(db_session: AsyncSession) -> None:
    """Test parsing a dovish Fed statement."""
    tracker = FedTracker(db_session)

    statement = """
    The Committee will remain patient and continue to support the economy
    with accommodative policy. We will be gradual in our approach and
    carefully monitor economic conditions to lower policy rates if needed.
    """

    parsed = tracker.parse_fed_statement(statement)

    assert parsed["sentiment"] == "dovish"
    assert parsed["dovish_count"] > parsed["hawkish_count"]


@pytest.mark.asyncio
async def test_parse_fed_statement_neutral(db_session: AsyncSession) -> None:
    """Test parsing a neutral Fed statement."""
    tracker = FedTracker(db_session)

    statement = """
    The Committee is monitoring economic indicators and labor market conditions.
    We will continue to assess the appropriate stance of monetary policy.
    """

    parsed = tracker.parse_fed_statement(statement)

    assert parsed["sentiment"] == "neutral"


@pytest.mark.asyncio
async def test_parse_fed_statement_rate_increase(db_session: AsyncSession) -> None:
    """Test detecting rate increase in Fed statement."""
    tracker = FedTracker(db_session)

    statement = """
    In light of persistent inflation, the Committee has decided to raise
    the federal funds rate by 25 basis points.
    """

    parsed = tracker.parse_fed_statement(statement)

    assert parsed["rate_action"] == "increase"


@pytest.mark.asyncio
async def test_parse_fed_statement_rate_decrease(db_session: AsyncSession) -> None:
    """Test detecting rate decrease in Fed statement."""
    tracker = FedTracker(db_session)

    statement = """
    Given current economic conditions, the Committee has decided to lower
    the federal funds rate to support economic growth.
    """

    parsed = tracker.parse_fed_statement(statement)

    assert parsed["rate_action"] == "decrease"


@pytest.mark.asyncio
async def test_parse_fed_statement_rate_hold(db_session: AsyncSession) -> None:
    """Test detecting rate hold in Fed statement."""
    tracker = FedTracker(db_session)

    statement = """
    The Committee decided to maintain the federal funds rate at its current
    level, keeping interest rates unchanged at this time.
    """

    parsed = tracker.parse_fed_statement(statement)

    assert parsed["rate_action"] == "hold"


@pytest.mark.asyncio
async def test_parse_fed_statement_key_phrases(db_session: AsyncSession) -> None:
    """Test extracting key phrases from Fed statement."""
    tracker = FedTracker(db_session)

    statement = """
    Inflation remains elevated. The labor market continues to be strong.
    Economic growth has moderated. The Committee is monitoring interest rate
    policy carefully.
    """

    parsed = tracker.parse_fed_statement(statement)

    assert "inflation concerns" in parsed["key_phrases"]
    assert "employment/labor market" in parsed["key_phrases"]
    assert "economic growth" in parsed["key_phrases"]
    assert "interest rate policy" in parsed["key_phrases"]


@pytest.mark.asyncio
async def test_get_fed_rate_history(db_session: AsyncSession) -> None:
    """Test retrieving Fed rate history."""
    tracker = FedTracker(db_session)

    # Create historical rate decisions
    await create_fed_event(
        db_session,
        event_date=datetime(2024, 1, 31, tzinfo=UTC),
        rate_decision=5.25,
    )
    await create_fed_event(
        db_session,
        event_date=datetime(2024, 3, 20, tzinfo=UTC),
        rate_decision=5.50,
    )
    await create_fed_event(
        db_session,
        event_date=datetime(2024, 5, 1, tzinfo=UTC),
        rate_decision=5.50,
    )

    history = await tracker.get_fed_rate_history()

    assert len(history) == 3
    assert all("date" in entry for entry in history)
    assert all("rate" in entry for entry in history)
    # Should be in descending order
    assert history[0]["date"] > history[1]["date"]


@pytest.mark.asyncio
async def test_get_fed_rate_history_empty(db_session: AsyncSession) -> None:
    """Test Fed rate history when no events exist."""
    tracker = FedTracker(db_session)

    history = await tracker.get_fed_rate_history()

    assert len(history) == 0


@pytest.mark.asyncio
async def test_get_fed_rate_history_excludes_future_events(db_session: AsyncSession) -> None:
    """Test that rate history excludes future events."""
    tracker = FedTracker(db_session)

    # Create past event
    await create_fed_event(
        db_session,
        event_date=datetime(2024, 1, 31, tzinfo=UTC),
        rate_decision=5.25,
    )
    # Create future event
    await create_fed_event(
        db_session,
        event_date=datetime.now(UTC).replace(year=datetime.now(UTC).year + 1),
        rate_decision=6.00,
    )

    history = await tracker.get_fed_rate_history()

    # Should only include past events
    assert all(
        datetime.fromisoformat(entry["date"].replace("Z", "+00:00")) < datetime.now(UTC)
        for entry in history
    )


@pytest.mark.asyncio
async def test_calculate_fed_sentiment_delta_more_hawkish(db_session: AsyncSession) -> None:
    """Test sentiment delta when current is more hawkish."""
    tracker = FedTracker(db_session)

    previous = "The Committee will remain patient and support the economy."
    current = "Inflation is elevated and persistent, requiring tighter policy."

    delta = tracker.calculate_fed_sentiment_delta(current, previous)

    assert delta > 0  # Positive delta means more hawkish


@pytest.mark.asyncio
async def test_calculate_fed_sentiment_delta_more_dovish(db_session: AsyncSession) -> None:
    """Test sentiment delta when current is more dovish."""
    tracker = FedTracker(db_session)

    previous = "Inflation requires raising interest rates to tighten policy."
    current = "The Committee will be patient and support economic growth."

    delta = tracker.calculate_fed_sentiment_delta(current, previous)

    assert delta < 0  # Negative delta means more dovish


@pytest.mark.asyncio
async def test_calculate_fed_sentiment_delta_no_change(db_session: AsyncSession) -> None:
    """Test sentiment delta when sentiment is unchanged."""
    tracker = FedTracker(db_session)

    previous = "The Committee is monitoring economic conditions."
    current = "The Committee continues to assess economic indicators."

    delta = tracker.calculate_fed_sentiment_delta(current, previous)

    assert delta == 0.0  # No change in sentiment


@pytest.mark.asyncio
async def test_store_fed_event(db_session: AsyncSession) -> None:
    """Test storing a Fed event."""
    tracker = FedTracker(db_session)

    fed_event = FedEvent(
        event_date=datetime(2024, 6, 12, tzinfo=UTC),
        importance=EventImportance.CRITICAL,
        rate_decision=5.25,
        statement_summary="Rate maintained at current level",
        title="FOMC Meeting - June 2024",
        description="Federal Open Market Committee meeting",
        source="federal_reserve",
    )

    event = await tracker.store_fed_event(fed_event)

    assert event.id is not None
    assert event.event_type == EventType.FOMC
    assert event.importance == EventImportance.CRITICAL
    assert event.actual_value == 5.25
    assert event.metadata_json is not None
    assert event.metadata_json["rate_decision"] == 5.25


@pytest.mark.asyncio
async def test_fomc_schedule_chronological_order(db_session: AsyncSession) -> None:
    """Test that FOMC schedule is in chronological order."""
    tracker = FedTracker(db_session)

    events = await tracker.get_fomc_schedule(year=2024)

    # Verify ascending order
    for i in range(len(events) - 1):
        assert events[i].event_date <= events[i + 1].event_date


@pytest.mark.asyncio
async def test_fed_event_always_critical_importance(db_session: AsyncSession) -> None:
    """Test that Fed events are always marked as critical importance."""
    tracker = FedTracker(db_session)

    events = await tracker.get_fomc_schedule(year=2024)

    assert all(event.importance == EventImportance.CRITICAL for event in events)


@pytest.mark.asyncio
async def test_parse_fed_statement_empty(db_session: AsyncSession) -> None:
    """Test parsing empty Fed statement."""
    tracker = FedTracker(db_session)

    parsed = tracker.parse_fed_statement("")

    assert parsed["sentiment"] == "neutral"
    assert parsed["key_phrases"] == []
    assert parsed["rate_action"] is None


@pytest.mark.asyncio
async def test_next_fomc_meeting_is_future(db_session: AsyncSession) -> None:
    """Test that next FOMC meeting is in the future."""
    tracker = FedTracker(db_session)

    next_meeting = await tracker.get_next_fomc_meeting()

    if next_meeting:
        assert next_meeting.event_date > datetime.now(UTC)


@pytest.mark.asyncio
async def test_fed_event_no_symbol(db_session: AsyncSession) -> None:
    """Test that Fed events have no symbol."""
    tracker = FedTracker(db_session)

    fed_event = FedEvent(
        event_date=datetime(2024, 6, 12, tzinfo=UTC),
        importance=EventImportance.CRITICAL,
        title="FOMC Meeting",
        source="federal_reserve",
    )

    event = await tracker.store_fed_event(fed_event)

    assert event.symbol is None


@pytest.mark.asyncio
async def test_parse_fed_statement_case_insensitive(db_session: AsyncSession) -> None:
    """Test that Fed statement parsing is case insensitive."""
    tracker = FedTracker(db_session)

    statement_lower = "the committee will raise interest rates due to inflation"
    statement_upper = "THE COMMITTEE WILL RAISE INTEREST RATES DUE TO INFLATION"

    parsed_lower = tracker.parse_fed_statement(statement_lower)
    parsed_upper = tracker.parse_fed_statement(statement_upper)

    assert parsed_lower["sentiment"] == parsed_upper["sentiment"]
    assert parsed_lower["rate_action"] == parsed_upper["rate_action"]


@pytest.mark.asyncio
async def test_fed_rate_history_limit(db_session: AsyncSession) -> None:
    """Test that Fed rate history is limited to 20 entries."""
    from dateutil.relativedelta import relativedelta

    tracker = FedTracker(db_session)

    # Create 25 historical events
    base_date = datetime(2020, 1, 1, tzinfo=UTC)
    for i in range(25):
        event_date = base_date + relativedelta(months=i)
        await create_fed_event(
            db_session,
            event_date=event_date,
            rate_decision=5.0 + (i * 0.1),
        )

    history = await tracker.get_fed_rate_history()

    # Should be limited to 20 most recent
    assert len(history) <= 20


@pytest.mark.asyncio
async def test_get_fomc_schedule_with_partial_data(db_session: AsyncSession) -> None:
    """Test FOMC schedule when some meetings have data stored."""
    tracker = FedTracker(db_session)

    # Store data for only one meeting
    await create_fed_event(
        db_session,
        event_date=datetime(2024, 1, 31, tzinfo=UTC),
        rate_decision=5.25,
    )

    events = await tracker.get_fomc_schedule(year=2024)

    # Current implementation returns only stored events if any exist
    assert len(events) >= 1

    # The stored event should have rate decision
    events_with_rate = [e for e in events if e.rate_decision is not None]
    assert len(events_with_rate) >= 1
    assert events_with_rate[0].rate_decision == 5.25

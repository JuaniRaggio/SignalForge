"""Tests for competition functionality."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.models.competition import Competition, CompetitionParticipant, CompetitionStatus, CompetitionType
from signalforge.models.paper_trading import PaperPortfolio, PortfolioStatus
from signalforge.models.user import User
from signalforge.paper_trading.competition_service import CompetitionService, CompetitionValidationError


@pytest.fixture
def future_dates():
    """Fixture providing valid future dates for competition."""
    now = datetime.now(UTC)
    return {
        "registration_start": now + timedelta(days=1),
        "registration_end": now + timedelta(days=7),
        "competition_start": now + timedelta(days=8),
        "competition_end": now + timedelta(days=30),
    }


@pytest.fixture
def active_registration_dates():
    """Fixture providing dates where registration is currently open."""
    now = datetime.now(UTC)
    return {
        "registration_start": now - timedelta(days=1),
        "registration_end": now + timedelta(days=7),
        "competition_start": now + timedelta(days=8),
        "competition_end": now + timedelta(days=30),
    }


@pytest.fixture
async def test_user(db_session: AsyncSession) -> User:
    """Create a test user."""
    user = User(
        email="test@example.com",
        username="testuser",
        hashed_password="hashed_password",
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def test_competition(db_session: AsyncSession, test_user: User, future_dates: dict) -> Competition:
    """Create a test competition."""
    competition = Competition(
        name="Test Competition",
        description="A test competition",
        competition_type=CompetitionType.PUBLIC,
        status=CompetitionStatus.DRAFT,
        registration_start=future_dates["registration_start"],
        registration_end=future_dates["registration_end"],
        competition_start=future_dates["competition_start"],
        competition_end=future_dates["competition_end"],
        initial_capital=Decimal("100000"),
        max_participants=100,
        created_by=test_user.id,
    )
    db_session.add(competition)
    await db_session.commit()
    await db_session.refresh(competition)
    return competition


@pytest.fixture
async def open_registration_competition(db_session: AsyncSession, test_user: User, active_registration_dates: dict) -> Competition:
    """Create a competition with registration currently open."""
    competition = Competition(
        name="Open Registration Competition",
        description="Competition with active registration period",
        competition_type=CompetitionType.PUBLIC,
        status=CompetitionStatus.REGISTRATION_OPEN,
        registration_start=active_registration_dates["registration_start"],
        registration_end=active_registration_dates["registration_end"],
        competition_start=active_registration_dates["competition_start"],
        competition_end=active_registration_dates["competition_end"],
        initial_capital=Decimal("100000"),
        max_participants=100,
        created_by=test_user.id,
    )
    db_session.add(competition)
    await db_session.commit()
    await db_session.refresh(competition)
    return competition


class TestCompetitionCreation:
    """Tests for competition creation."""

    async def test_create_competition_success(self, db_session: AsyncSession, test_user: User, future_dates: dict):
        """Test successful competition creation."""
        service = CompetitionService()

        competition = await service.create_competition(
            session=db_session,
            name="Test Competition",
            description="Test description",
            registration_start=future_dates["registration_start"],
            registration_end=future_dates["registration_end"],
            competition_start=future_dates["competition_start"],
            competition_end=future_dates["competition_end"],
            initial_capital=Decimal("100000"),
            created_by=test_user.id,
            max_participants=50,
            rules={"max_position_pct": 20},
            prizes={"1": "First prize", "2": "Second prize"},
            competition_type="public",
        )

        await db_session.commit()

        assert competition.id is not None
        assert competition.name == "Test Competition"
        assert competition.status == CompetitionStatus.DRAFT
        assert competition.initial_capital == Decimal("100000")
        assert competition.max_participants == 50
        assert competition.rules == {"max_position_pct": 20}

    async def test_create_competition_invalid_dates(self, db_session: AsyncSession, test_user: User):
        """Test competition creation with invalid dates."""
        service = CompetitionService()
        now = datetime.now(UTC)

        # Registration end before start
        with pytest.raises(CompetitionValidationError, match="Registration end must be after"):
            await service.create_competition(
                session=db_session,
                name="Invalid Competition",
                description="Invalid dates",
                registration_start=now + timedelta(days=7),
                registration_end=now + timedelta(days=1),
                competition_start=now + timedelta(days=8),
                competition_end=now + timedelta(days=30),
                initial_capital=Decimal("100000"),
                created_by=test_user.id,
            )

    async def test_create_competition_past_registration_start(self, db_session: AsyncSession, test_user: User):
        """Test competition creation with past registration start."""
        service = CompetitionService()
        now = datetime.now(UTC)

        with pytest.raises(CompetitionValidationError, match="must be in the future"):
            await service.create_competition(
                session=db_session,
                name="Invalid Competition",
                description="Past start",
                registration_start=now - timedelta(days=1),
                registration_end=now + timedelta(days=7),
                competition_start=now + timedelta(days=8),
                competition_end=now + timedelta(days=30),
                initial_capital=Decimal("100000"),
                created_by=test_user.id,
            )


class TestCompetitionRegistration:
    """Tests for competition registration."""

    async def test_register_participant_success(
        self, db_session: AsyncSession, open_registration_competition: Competition, test_user: User
    ):
        """Test successful participant registration."""
        service = CompetitionService()

        participant = await service.register_participant(
            session=db_session,
            competition_id=open_registration_competition.id,
            user_id=test_user.id,
        )

        await db_session.commit()

        assert participant.id is not None
        assert participant.user_id == test_user.id
        assert participant.competition_id == open_registration_competition.id
        assert participant.is_active is True
        assert participant.disqualified is False

        # Verify portfolio was created
        result = await db_session.execute(
            select(PaperPortfolio).where(PaperPortfolio.id == participant.portfolio_id)
        )
        portfolio = result.scalar_one_or_none()

        assert portfolio is not None
        assert portfolio.is_competition_portfolio is True
        assert portfolio.competition_id == open_registration_competition.id
        assert portfolio.initial_capital == open_registration_competition.initial_capital

    async def test_register_duplicate_user(
        self, db_session: AsyncSession, open_registration_competition: Competition, test_user: User
    ):
        """Test registering same user twice."""
        service = CompetitionService()

        # First registration
        await service.register_participant(
            session=db_session,
            competition_id=open_registration_competition.id,
            user_id=test_user.id,
        )
        await db_session.commit()

        # Second registration should fail
        with pytest.raises(CompetitionValidationError, match="already registered"):
            await service.register_participant(
                session=db_session,
                competition_id=open_registration_competition.id,
                user_id=test_user.id,
            )

    async def test_register_wrong_status(
        self, db_session: AsyncSession, test_competition: Competition, test_user: User
    ):
        """Test registration when competition is not open."""
        # Competition is in DRAFT status
        service = CompetitionService()

        with pytest.raises(CompetitionValidationError, match="not open for registration"):
            await service.register_participant(
                session=db_session,
                competition_id=test_competition.id,
                user_id=test_user.id,
            )

    async def test_register_max_participants_reached(self, db_session: AsyncSession, test_user: User, active_registration_dates: dict):
        """Test registration when max participants reached."""
        # Create competition with max 1 participant
        competition = Competition(
            name="Limited Competition",
            description="Max 1 participant",
            competition_type=CompetitionType.PUBLIC,
            status=CompetitionStatus.REGISTRATION_OPEN,
            registration_start=active_registration_dates["registration_start"],
            registration_end=active_registration_dates["registration_end"],
            competition_start=active_registration_dates["competition_start"],
            competition_end=active_registration_dates["competition_end"],
            initial_capital=Decimal("100000"),
            max_participants=1,
            created_by=test_user.id,
        )
        db_session.add(competition)
        await db_session.commit()

        service = CompetitionService()

        # First user registers
        await service.register_participant(
            session=db_session,
            competition_id=competition.id,
            user_id=test_user.id,
        )
        await db_session.commit()

        # Second user tries to register
        second_user = User(
            email="second@example.com",
            username="seconduser",
            hashed_password="hashed_password",
        )
        db_session.add(second_user)
        await db_session.commit()

        with pytest.raises(CompetitionValidationError, match="full"):
            await service.register_participant(
                session=db_session,
                competition_id=competition.id,
                user_id=second_user.id,
            )


class TestCompetitionWithdrawal:
    """Tests for competition withdrawal."""

    async def test_withdraw_participant_success(
        self, db_session: AsyncSession, open_registration_competition: Competition, test_user: User
    ):
        """Test successful participant withdrawal."""
        service = CompetitionService()

        # Register first
        participant = await service.register_participant(
            session=db_session,
            competition_id=open_registration_competition.id,
            user_id=test_user.id,
        )
        await db_session.commit()

        # Withdraw
        await service.withdraw_participant(
            session=db_session,
            competition_id=open_registration_competition.id,
            user_id=test_user.id,
        )
        await db_session.commit()

        # Verify participant is inactive
        await db_session.refresh(participant)
        assert participant.is_active is False

    async def test_withdraw_from_active_competition(
        self, db_session: AsyncSession, open_registration_competition: Competition, test_user: User
    ):
        """Test withdrawal from active competition should fail."""
        service = CompetitionService()

        # Register
        await service.register_participant(
            session=db_session,
            competition_id=open_registration_competition.id,
            user_id=test_user.id,
        )
        await db_session.commit()

        # Change status to active
        open_registration_competition.status = CompetitionStatus.ACTIVE
        await db_session.commit()

        # Try to withdraw
        with pytest.raises(CompetitionValidationError, match="Cannot withdraw from active"):
            await service.withdraw_participant(
                session=db_session,
                competition_id=open_registration_competition.id,
                user_id=test_user.id,
            )


class TestCompetitionDisqualification:
    """Tests for participant disqualification."""

    async def test_disqualify_participant_success(
        self, db_session: AsyncSession, open_registration_competition: Competition, test_user: User
    ):
        """Test successful participant disqualification."""
        service = CompetitionService()

        # Register first
        participant = await service.register_participant(
            session=db_session,
            competition_id=open_registration_competition.id,
            user_id=test_user.id,
        )
        await db_session.commit()

        # Disqualify
        reason = "Violated trading rules"
        await service.disqualify_participant(
            session=db_session,
            competition_id=open_registration_competition.id,
            user_id=test_user.id,
            reason=reason,
        )
        await db_session.commit()

        # Verify participant is disqualified
        await db_session.refresh(participant)
        assert participant.disqualified is True
        assert participant.disqualification_reason == reason
        assert participant.is_active is False


class TestCompetitionStatusUpdates:
    """Tests for competition status transitions."""

    async def test_update_status_draft_to_registration_open(
        self, db_session: AsyncSession, test_user: User
    ):
        """Test status transition from DRAFT to REGISTRATION_OPEN."""
        now = datetime.now(UTC)
        competition = Competition(
            name="Status Test",
            description="Test status transitions",
            competition_type=CompetitionType.PUBLIC,
            status=CompetitionStatus.DRAFT,
            registration_start=now - timedelta(hours=1),  # Past start
            registration_end=now + timedelta(days=7),
            competition_start=now + timedelta(days=8),
            competition_end=now + timedelta(days=30),
            initial_capital=Decimal("100000"),
            created_by=test_user.id,
        )
        db_session.add(competition)
        await db_session.commit()

        service = CompetitionService()

        updated = await service.update_competition_status(
            session=db_session,
            competition_id=competition.id,
        )
        await db_session.commit()

        assert updated.status == CompetitionStatus.REGISTRATION_OPEN

    async def test_update_status_registration_to_active(
        self, db_session: AsyncSession, test_user: User
    ):
        """Test status transition from REGISTRATION_OPEN to ACTIVE."""
        now = datetime.now(UTC)
        competition = Competition(
            name="Status Test",
            description="Test status transitions",
            competition_type=CompetitionType.PUBLIC,
            status=CompetitionStatus.REGISTRATION_OPEN,
            registration_start=now - timedelta(days=10),
            registration_end=now - timedelta(days=2),
            competition_start=now - timedelta(hours=1),  # Past start
            competition_end=now + timedelta(days=20),
            initial_capital=Decimal("100000"),
            created_by=test_user.id,
        )
        db_session.add(competition)
        await db_session.commit()

        service = CompetitionService()

        updated = await service.update_competition_status(
            session=db_session,
            competition_id=competition.id,
        )
        await db_session.commit()

        assert updated.status == CompetitionStatus.ACTIVE

    async def test_update_status_active_to_completed(
        self, db_session: AsyncSession, test_user: User
    ):
        """Test status transition from ACTIVE to COMPLETED."""
        now = datetime.now(UTC)
        competition = Competition(
            name="Status Test",
            description="Test status transitions",
            competition_type=CompetitionType.PUBLIC,
            status=CompetitionStatus.ACTIVE,
            registration_start=now - timedelta(days=30),
            registration_end=now - timedelta(days=25),
            competition_start=now - timedelta(days=20),
            competition_end=now - timedelta(hours=1),  # Past end
            initial_capital=Decimal("100000"),
            created_by=test_user.id,
        )
        db_session.add(competition)
        await db_session.commit()

        service = CompetitionService()

        updated = await service.update_competition_status(
            session=db_session,
            competition_id=competition.id,
        )
        await db_session.commit()

        assert updated.status == CompetitionStatus.COMPLETED


class TestCompetitionStandings:
    """Tests for competition standings calculation."""

    async def test_get_standings_empty_competition(
        self, db_session: AsyncSession, test_competition: Competition
    ):
        """Test getting standings for competition with no participants."""
        service = CompetitionService()

        standings = await service.get_standings(
            session=db_session,
            competition_id=test_competition.id,
        )

        assert standings == []

    async def test_get_standings_with_disqualified(
        self, db_session: AsyncSession, open_registration_competition: Competition, test_user: User
    ):
        """Test standings with disqualified participants at the end."""
        service = CompetitionService()

        # Register first during open registration
        participant = await service.register_participant(
            session=db_session,
            competition_id=open_registration_competition.id,
            user_id=test_user.id,
        )
        await db_session.commit()

        # Change to active competition
        open_registration_competition.status = CompetitionStatus.ACTIVE
        await db_session.commit()

        # Disqualify
        await service.disqualify_participant(
            session=db_session,
            competition_id=open_registration_competition.id,
            user_id=test_user.id,
            reason="Rule violation",
        )
        await db_session.commit()

        standings = await service.get_standings(
            session=db_session,
            competition_id=open_registration_competition.id,
        )

        # Should have one standing, and it should be marked as disqualified
        assert len(standings) == 1
        assert standings[0].is_disqualified is True


class TestCompetitionFinalization:
    """Tests for competition finalization."""

    async def test_finalize_competition_success(
        self, db_session: AsyncSession, open_registration_competition: Competition, test_user: User
    ):
        """Test successful competition finalization."""
        service = CompetitionService()

        # Register a participant while registration is open
        participant = await service.register_participant(
            session=db_session,
            competition_id=open_registration_competition.id,
            user_id=test_user.id,
        )
        await db_session.commit()

        # Move to completed status
        open_registration_competition.status = CompetitionStatus.COMPLETED
        await db_session.commit()

        # Finalize
        finalized = await service.finalize_competition(
            session=db_session,
            competition_id=open_registration_competition.id,
        )
        await db_session.commit()

        # Check participant has final rank
        await db_session.refresh(participant)
        assert participant.final_rank is not None
        assert participant.final_return_pct is not None

    async def test_finalize_non_completed_competition(
        self, db_session: AsyncSession, test_competition: Competition
    ):
        """Test finalization of non-completed competition should fail."""
        # Competition is in DRAFT status
        service = CompetitionService()

        with pytest.raises(CompetitionValidationError, match="must be COMPLETED"):
            await service.finalize_competition(
                session=db_session,
                competition_id=test_competition.id,
            )


class TestCompetitionQueries:
    """Tests for competition query methods."""

    async def test_get_competition(self, db_session: AsyncSession, test_competition: Competition):
        """Test getting competition by ID."""
        service = CompetitionService()

        competition = await service.get_competition(
            session=db_session,
            competition_id=test_competition.id,
        )

        assert competition is not None
        assert competition.id == test_competition.id

    async def test_get_nonexistent_competition(self, db_session: AsyncSession):
        """Test getting non-existent competition."""
        service = CompetitionService()

        competition = await service.get_competition(
            session=db_session,
            competition_id=uuid4(),
        )

        assert competition is None

    async def test_list_competitions(self, db_session: AsyncSession, test_competition: Competition):
        """Test listing competitions."""
        service = CompetitionService()

        competitions = await service.list_competitions(
            session=db_session,
            limit=10,
            offset=0,
        )

        assert len(competitions) >= 1
        assert any(c.id == test_competition.id for c in competitions)

    async def test_list_competitions_with_status_filter(
        self, db_session: AsyncSession, test_competition: Competition
    ):
        """Test listing competitions with status filter."""
        service = CompetitionService()

        competitions = await service.list_competitions(
            session=db_session,
            status=CompetitionStatus.DRAFT,
            limit=10,
            offset=0,
        )

        assert all(c.status == CompetitionStatus.DRAFT for c in competitions)

    async def test_get_user_competitions(
        self, db_session: AsyncSession, open_registration_competition: Competition, test_user: User
    ):
        """Test getting user's competitions."""
        service = CompetitionService()

        # Register user
        await service.register_participant(
            session=db_session,
            competition_id=open_registration_competition.id,
            user_id=test_user.id,
        )
        await db_session.commit()

        # Get user competitions
        user_comps = await service.get_user_competitions(
            session=db_session,
            user_id=test_user.id,
        )

        assert len(user_comps) == 1
        assert user_comps[0]["competition"].id == open_registration_competition.id
        assert user_comps[0]["participant"].user_id == test_user.id

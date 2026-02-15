"""Competition management service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.core.logging import get_logger
from signalforge.models.competition import (
    Competition,
    CompetitionParticipant,
    CompetitionStatus,
    CompetitionType,
)
from signalforge.models.paper_trading import PaperPortfolio, PortfolioStatus
from signalforge.models.user import User
from signalforge.paper_trading.performance_service import PerformanceService

logger = get_logger(__name__)


@dataclass
class CompetitionStanding:
    """Competition participant standing."""

    rank: int
    user_id: UUID
    username: str
    portfolio_id: UUID
    total_return_pct: Decimal
    sharpe_ratio: Decimal | None
    total_trades: int
    is_disqualified: bool


class CompetitionValidationError(Exception):
    """Raised when competition validation fails."""


class CompetitionService:
    """Manage trading competitions."""

    def __init__(self, performance_service: PerformanceService | None = None) -> None:
        """Initialize competition service.

        Args:
            performance_service: Optional performance service for metrics calculation
        """
        self.performance_service = performance_service or PerformanceService()

    def _validate_competition_dates(
        self,
        registration_start: datetime,
        registration_end: datetime,
        competition_start: datetime,
        competition_end: datetime,
    ) -> None:
        """Validate competition date constraints.

        Args:
            registration_start: Registration start datetime
            registration_end: Registration end datetime
            competition_start: Competition start datetime
            competition_end: Competition end datetime

        Raises:
            CompetitionValidationError: If dates are invalid
        """
        now = datetime.now(UTC)

        if registration_start < now:
            raise CompetitionValidationError("Registration start must be in the future")

        if registration_end <= registration_start:
            raise CompetitionValidationError("Registration end must be after registration start")

        if competition_start <= registration_end:
            raise CompetitionValidationError("Competition start must be after registration end")

        if competition_end <= competition_start:
            raise CompetitionValidationError("Competition end must be after competition start")

    async def create_competition(
        self,
        session: AsyncSession,
        name: str,
        description: str | None,
        registration_start: datetime,
        registration_end: datetime,
        competition_start: datetime,
        competition_end: datetime,
        initial_capital: Decimal,
        created_by: UUID,
        max_participants: int | None = None,
        rules: dict | None = None,
        prizes: dict | None = None,
        competition_type: str = "public",
    ) -> Competition:
        """Create a new competition.

        Args:
            session: Database session
            name: Competition name
            description: Competition description
            registration_start: When registration opens
            registration_end: When registration closes
            competition_start: When competition starts
            competition_end: When competition ends
            initial_capital: Starting capital for participants
            created_by: User ID of creator
            max_participants: Maximum number of participants
            rules: Competition rules as JSON
            prizes: Prize structure as JSON
            competition_type: Type of competition (public, private, sponsored)

        Returns:
            Created competition

        Raises:
            CompetitionValidationError: If validation fails
        """
        logger.info(
            "creating_competition",
            name=name,
            competition_type=competition_type,
            created_by=str(created_by),
        )

        # Validate dates
        self._validate_competition_dates(
            registration_start,
            registration_end,
            competition_start,
            competition_end,
        )

        # Validate participants
        if max_participants is not None and max_participants < 2:
            raise CompetitionValidationError("Competition must allow at least 2 participants")

        # Create competition
        competition = Competition(
            name=name,
            description=description,
            competition_type=CompetitionType(competition_type),
            status=CompetitionStatus.DRAFT,
            registration_start=registration_start,
            registration_end=registration_end,
            competition_start=competition_start,
            competition_end=competition_end,
            initial_capital=initial_capital,
            max_participants=max_participants,
            rules=rules,
            prizes=prizes,
            created_by=created_by,
        )

        session.add(competition)
        await session.flush()

        logger.info("competition_created", competition_id=str(competition.id))
        return competition

    async def register_participant(
        self,
        session: AsyncSession,
        competition_id: UUID,
        user_id: UUID,
    ) -> CompetitionParticipant:
        """Register user for competition.

        Args:
            session: Database session
            competition_id: Competition UUID
            user_id: User UUID

        Returns:
            Created participant record

        Raises:
            CompetitionValidationError: If registration validation fails
        """
        logger.info(
            "registering_participant",
            competition_id=str(competition_id),
            user_id=str(user_id),
        )

        # Get competition
        competition = await self.get_competition(session, competition_id)
        if not competition:
            raise CompetitionValidationError("Competition not found")

        # Validate competition is in registration phase
        now = datetime.now(UTC)
        if competition.status != CompetitionStatus.REGISTRATION_OPEN:
            raise CompetitionValidationError(
                f"Competition is not open for registration (status: {competition.status})"
            )

        if now < competition.registration_start or now > competition.registration_end:
            raise CompetitionValidationError("Registration is not currently open")

        # Check if user already registered
        existing = await session.execute(
            select(CompetitionParticipant).where(
                CompetitionParticipant.competition_id == competition_id,
                CompetitionParticipant.user_id == user_id,
            )
        )
        if existing.scalar_one_or_none():
            raise CompetitionValidationError("User already registered for this competition")

        # Check max participants
        if competition.max_participants:
            participant_count = await session.scalar(
                select(func.count(CompetitionParticipant.id)).where(
                    CompetitionParticipant.competition_id == competition_id,
                    CompetitionParticipant.is_active.is_(True),
                )
            )
            if participant_count is not None and participant_count >= competition.max_participants:
                raise CompetitionValidationError("Competition is full")

        # Get user for username
        user_result = await session.execute(select(User).where(User.id == user_id))
        user = user_result.scalar_one_or_none()
        if not user:
            raise CompetitionValidationError("User not found")

        # Create dedicated portfolio for competition
        portfolio = PaperPortfolio(
            user_id=user_id,
            name=f"{competition.name} - {user.username}",
            initial_capital=competition.initial_capital,
            current_cash=competition.initial_capital,
            status=PortfolioStatus.ACTIVE,
            is_competition_portfolio=True,
            competition_id=competition_id,
        )
        session.add(portfolio)
        await session.flush()

        # Create participant record
        participant = CompetitionParticipant(
            competition_id=competition_id,
            user_id=user_id,
            portfolio_id=portfolio.id,
            registered_at=now,
            is_active=True,
        )
        session.add(participant)
        await session.flush()

        logger.info(
            "participant_registered",
            competition_id=str(competition_id),
            user_id=str(user_id),
            portfolio_id=str(portfolio.id),
        )

        return participant

    async def withdraw_participant(
        self,
        session: AsyncSession,
        competition_id: UUID,
        user_id: UUID,
    ) -> None:
        """Withdraw from competition (before it starts).

        Args:
            session: Database session
            competition_id: Competition UUID
            user_id: User UUID

        Raises:
            CompetitionValidationError: If withdrawal validation fails
        """
        logger.info(
            "withdrawing_participant",
            competition_id=str(competition_id),
            user_id=str(user_id),
        )

        # Get competition
        competition = await self.get_competition(session, competition_id)
        if not competition:
            raise CompetitionValidationError("Competition not found")

        # Can only withdraw before competition starts
        if competition.status == CompetitionStatus.ACTIVE:
            raise CompetitionValidationError("Cannot withdraw from active competition")

        if competition.status == CompetitionStatus.COMPLETED:
            raise CompetitionValidationError("Cannot withdraw from completed competition")

        # Get participant
        result = await session.execute(
            select(CompetitionParticipant).where(
                CompetitionParticipant.competition_id == competition_id,
                CompetitionParticipant.user_id == user_id,
            )
        )
        participant = result.scalar_one_or_none()
        if not participant:
            raise CompetitionValidationError("User not registered for this competition")

        # Mark as inactive
        participant.is_active = False
        await session.flush()

        logger.info(
            "participant_withdrawn",
            competition_id=str(competition_id),
            user_id=str(user_id),
        )

    async def disqualify_participant(
        self,
        session: AsyncSession,
        competition_id: UUID,
        user_id: UUID,
        reason: str,
    ) -> None:
        """Disqualify a participant for rule violation.

        Args:
            session: Database session
            competition_id: Competition UUID
            user_id: User UUID
            reason: Reason for disqualification

        Raises:
            CompetitionValidationError: If disqualification validation fails
        """
        logger.info(
            "disqualifying_participant",
            competition_id=str(competition_id),
            user_id=str(user_id),
            reason=reason,
        )

        # Get participant
        result = await session.execute(
            select(CompetitionParticipant).where(
                CompetitionParticipant.competition_id == competition_id,
                CompetitionParticipant.user_id == user_id,
            )
        )
        participant = result.scalar_one_or_none()
        if not participant:
            raise CompetitionValidationError("Participant not found")

        # Mark as disqualified
        participant.disqualified = True
        participant.disqualification_reason = reason
        participant.is_active = False
        await session.flush()

        logger.info(
            "participant_disqualified",
            competition_id=str(competition_id),
            user_id=str(user_id),
        )

    async def get_competition(
        self,
        session: AsyncSession,
        competition_id: UUID,
    ) -> Competition | None:
        """Get competition by ID.

        Args:
            session: Database session
            competition_id: Competition UUID

        Returns:
            Competition or None if not found
        """
        result = await session.execute(
            select(Competition).where(Competition.id == competition_id)
        )
        return result.scalar_one_or_none()

    async def list_competitions(
        self,
        session: AsyncSession,
        status: CompetitionStatus | None = None,
        include_private: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Competition]:
        """List competitions with filters.

        Args:
            session: Database session
            status: Filter by status
            include_private: Include private competitions
            limit: Maximum results to return
            offset: Number of results to skip

        Returns:
            List of competitions
        """
        query = select(Competition).order_by(Competition.created_at.desc())

        if status:
            query = query.where(Competition.status == status)

        if not include_private:
            query = query.where(Competition.competition_type != CompetitionType.PRIVATE)

        query = query.limit(limit).offset(offset)

        result = await session.execute(query)
        return list(result.scalars().all())

    async def get_standings(
        self,
        session: AsyncSession,
        competition_id: UUID,
        limit: int = 100,
    ) -> list[CompetitionStanding]:
        """Get current competition standings.

        Args:
            session: Database session
            competition_id: Competition UUID
            limit: Maximum standings to return

        Returns:
            List of standings ordered by rank
        """
        logger.info("calculating_standings", competition_id=str(competition_id))

        # Get competition
        competition = await self.get_competition(session, competition_id)
        if not competition:
            raise CompetitionValidationError("Competition not found")

        # Get all participants (active and disqualified, excluding withdrawn)
        result = await session.execute(
            select(CompetitionParticipant, User)
            .join(User, CompetitionParticipant.user_id == User.id)
            .where(
                CompetitionParticipant.competition_id == competition_id,
                or_(
                    CompetitionParticipant.is_active.is_(True),
                    CompetitionParticipant.disqualified.is_(True),
                ),
            )
        )
        participants_with_users = result.all()

        standings: list[CompetitionStanding] = []

        # Calculate metrics for each participant
        for participant, user in participants_with_users:
            try:
                metrics = await self.performance_service.calculate_metrics(
                    portfolio_id=participant.portfolio_id,
                    session=session,
                    start_date=competition.competition_start,
                    end_date=competition.competition_end if competition.status == CompetitionStatus.COMPLETED else None,
                )

                standings.append(
                    CompetitionStanding(
                        rank=0,  # Will be assigned after sorting
                        user_id=participant.user_id,
                        username=user.username,
                        portfolio_id=participant.portfolio_id,
                        total_return_pct=metrics.total_return_pct,
                        sharpe_ratio=metrics.sharpe_ratio,
                        total_trades=metrics.total_trades,
                        is_disqualified=participant.disqualified,
                    )
                )
            except Exception as e:
                logger.error(
                    "error_calculating_participant_metrics",
                    participant_id=str(participant.id),
                    error=str(e),
                )
                # Include participant with zero metrics if calculation fails
                standings.append(
                    CompetitionStanding(
                        rank=0,
                        user_id=participant.user_id,
                        username=user.username,
                        portfolio_id=participant.portfolio_id,
                        total_return_pct=Decimal("0"),
                        sharpe_ratio=None,
                        total_trades=0,
                        is_disqualified=participant.disqualified,
                    )
                )

        # Sort by total return (descending), with disqualified at the end
        standings.sort(
            key=lambda x: (x.is_disqualified, -x.total_return_pct)
        )

        # Assign ranks
        for rank, standing in enumerate(standings[:limit], start=1):
            standing.rank = rank

        return standings[:limit]

    async def update_competition_status(
        self,
        session: AsyncSession,
        competition_id: UUID,
    ) -> Competition:
        """Update competition status based on dates.

        Handles status transitions:
        DRAFT -> REGISTRATION_OPEN -> ACTIVE -> COMPLETED

        Args:
            session: Database session
            competition_id: Competition UUID

        Returns:
            Updated competition
        """
        competition = await self.get_competition(session, competition_id)
        if not competition:
            raise CompetitionValidationError("Competition not found")

        now = datetime.now(UTC)
        old_status = competition.status

        # State machine for status transitions
        if competition.status == CompetitionStatus.DRAFT and now >= competition.registration_start:
            competition.status = CompetitionStatus.REGISTRATION_OPEN
        elif (
            competition.status == CompetitionStatus.REGISTRATION_OPEN
            and now >= competition.competition_start
        ):
            competition.status = CompetitionStatus.ACTIVE
        elif (
            competition.status == CompetitionStatus.ACTIVE
            and now >= competition.competition_end
        ):
            competition.status = CompetitionStatus.COMPLETED

        if competition.status != old_status:
            logger.info(
                "competition_status_changed",
                competition_id=str(competition_id),
                old_status=old_status,
                new_status=competition.status,
            )
            await session.flush()

        return competition

    async def finalize_competition(
        self,
        session: AsyncSession,
        competition_id: UUID,
    ) -> Competition:
        """Finalize competition and award prizes.

        Args:
            session: Database session
            competition_id: Competition UUID

        Returns:
            Finalized competition

        Raises:
            CompetitionValidationError: If competition is not ready to finalize
        """
        logger.info("finalizing_competition", competition_id=str(competition_id))

        competition = await self.get_competition(session, competition_id)
        if not competition:
            raise CompetitionValidationError("Competition not found")

        if competition.status != CompetitionStatus.COMPLETED:
            raise CompetitionValidationError(
                f"Competition must be COMPLETED to finalize (current: {competition.status})"
            )

        # Get final standings
        standings = await self.get_standings(session, competition_id, limit=1000)

        # Update participant records with final standings
        for standing in standings:
            result = await session.execute(
                select(CompetitionParticipant).where(
                    CompetitionParticipant.competition_id == competition_id,
                    CompetitionParticipant.user_id == standing.user_id,
                )
            )
            participant = result.scalar_one_or_none()
            if participant:
                participant.final_rank = standing.rank
                participant.final_return_pct = standing.total_return_pct
                participant.final_sharpe = standing.sharpe_ratio

                # Award prizes based on rank
                if competition.prizes and str(standing.rank) in competition.prizes:
                    participant.prize_awarded = competition.prizes[str(standing.rank)]

        await session.flush()

        logger.info(
            "competition_finalized",
            competition_id=str(competition_id),
            total_participants=len(standings),
        )

        return competition

    async def get_user_competitions(
        self,
        session: AsyncSession,
        user_id: UUID,
        status: CompetitionStatus | None = None,
    ) -> list[dict]:
        """Get competitions user is participating in.

        Args:
            session: Database session
            user_id: User UUID
            status: Optional filter by competition status

        Returns:
            List of competition info with participant data
        """
        query = (
            select(Competition, CompetitionParticipant)
            .join(CompetitionParticipant, Competition.id == CompetitionParticipant.competition_id)
            .where(CompetitionParticipant.user_id == user_id)
            .order_by(Competition.competition_start.desc())
        )

        if status:
            query = query.where(Competition.status == status)

        result = await session.execute(query)
        competitions_with_participants = result.all()

        return [
            {
                "competition": comp,
                "participant": participant,
            }
            for comp, participant in competitions_with_participants
        ]

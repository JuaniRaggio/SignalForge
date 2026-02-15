"""Competition API routes."""

from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.api.dependencies.auth import get_current_user
from signalforge.api.dependencies.database import get_db
from signalforge.core.logging import get_logger
from signalforge.models.competition import Competition, CompetitionParticipant, CompetitionStatus
from signalforge.models.user import User
from signalforge.paper_trading.competition_schemas import (
    CompetitionCreate,
    CompetitionDisqualificationRequest,
    CompetitionListResponse,
    CompetitionResponse,
    CompetitionStandingResponse,
    CompetitionStandingsResponse,
    ParticipantResponse,
    UserCompetitionResponse,
)
from signalforge.paper_trading.competition_service import (
    CompetitionService,
    CompetitionValidationError,
)

router = APIRouter(prefix="/api/v1/competitions", tags=["competitions"])
logger = get_logger(__name__)


def _competition_to_response(competition: Competition, participant_count: int = 0) -> CompetitionResponse:
    """Convert Competition model to CompetitionResponse schema.

    Args:
        competition: Competition model instance
        participant_count: Number of active participants

    Returns:
        CompetitionResponse schema
    """
    return CompetitionResponse(
        id=competition.id,
        name=competition.name,
        description=competition.description,
        competition_type=competition.competition_type.value,
        status=competition.status.value,
        registration_start=competition.registration_start,
        registration_end=competition.registration_end,
        competition_start=competition.competition_start,
        competition_end=competition.competition_end,
        initial_capital=competition.initial_capital,
        max_participants=competition.max_participants,
        min_participants=competition.min_participants,
        current_participants=participant_count,
        rules=competition.rules,
        prizes=competition.prizes,
        created_by=competition.created_by,
        created_at=competition.created_at,
        updated_at=competition.updated_at,
    )


@router.post("", response_model=CompetitionResponse, status_code=status.HTTP_201_CREATED)
async def create_competition(
    data: CompetitionCreate,
    session: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> CompetitionResponse:
    """Create a new competition.

    Args:
        data: Competition creation data
        session: Database session
        current_user: Authenticated user

    Returns:
        Created competition
    """
    logger.info("create_competition_request", user_id=str(current_user.id), name=data.name)

    service = CompetitionService()

    try:
        competition = await service.create_competition(
            session=session,
            name=data.name,
            description=data.description,
            registration_start=data.registration_start,
            registration_end=data.registration_end,
            competition_start=data.competition_start,
            competition_end=data.competition_end,
            initial_capital=data.initial_capital,
            created_by=current_user.id,
            max_participants=data.max_participants,
            rules=data.rules,
            prizes=data.prizes,
            competition_type=data.competition_type,
        )

        await session.commit()

        return _competition_to_response(competition, 0)

    except CompetitionValidationError as e:
        logger.error("competition_creation_failed", error=str(e), user_id=str(current_user.id))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("unexpected_error_creating_competition", error=str(e), user_id=str(current_user.id))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create competition")


@router.get("", response_model=CompetitionListResponse)
async def list_competitions(
    *,
    session: Annotated[AsyncSession, Depends(get_db)],
    status_filter: str | None = Query(None, alias="status", description="Filter by competition status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
) -> CompetitionListResponse:
    """List available competitions.

    Args:
        status_filter: Optional status filter
        limit: Maximum results to return
        offset: Number of results to skip
        session: Database session

    Returns:
        List of competitions with pagination info
    """
    logger.info("list_competitions_request", status=status_filter, limit=limit, offset=offset)

    service = CompetitionService()

    # Parse status filter
    comp_status = None
    if status_filter:
        try:
            comp_status = CompetitionStatus(status_filter)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status_filter}",
            )

    competitions = await service.list_competitions(
        session=session,
        status=comp_status,
        include_private=False,
        limit=limit,
        offset=offset,
    )

    # Get participant counts
    competition_responses = []
    for comp in competitions:
        count = await session.scalar(
            select(func.count(CompetitionParticipant.id)).where(
                CompetitionParticipant.competition_id == comp.id,
                CompetitionParticipant.is_active.is_(True),
            )
        )
        competition_responses.append(_competition_to_response(comp, count or 0))

    # Get total count for pagination
    total_query = select(func.count(Competition.id))
    if comp_status:
        total_query = total_query.where(Competition.status == comp_status)
    total = await session.scalar(total_query) or 0

    return CompetitionListResponse(
        competitions=competition_responses,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{competition_id}", response_model=CompetitionResponse)
async def get_competition(
    competition_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> CompetitionResponse:
    """Get competition details.

    Args:
        competition_id: Competition UUID
        session: Database session

    Returns:
        Competition details
    """
    logger.info("get_competition_request", competition_id=str(competition_id))

    service = CompetitionService()
    competition = await service.get_competition(session, competition_id)

    if not competition:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Competition not found")

    # Get participant count
    count = await session.scalar(
        select(func.count(CompetitionParticipant.id)).where(
            CompetitionParticipant.competition_id == competition_id,
            CompetitionParticipant.is_active.is_(True),
        )
    )

    return _competition_to_response(competition, count or 0)


@router.post("/{competition_id}/register", response_model=ParticipantResponse, status_code=status.HTTP_201_CREATED)
async def register_for_competition(
    competition_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> ParticipantResponse:
    """Register for a competition.

    Args:
        competition_id: Competition UUID
        session: Database session
        current_user: Authenticated user

    Returns:
        Participant record
    """
    logger.info(
        "register_for_competition_request",
        competition_id=str(competition_id),
        user_id=str(current_user.id),
    )

    service = CompetitionService()

    try:
        participant = await service.register_participant(
            session=session,
            competition_id=competition_id,
            user_id=current_user.id,
        )

        await session.commit()

        return ParticipantResponse(
            id=participant.id,
            competition_id=participant.competition_id,
            user_id=participant.user_id,
            portfolio_id=participant.portfolio_id,
            registered_at=participant.registered_at,
            is_active=participant.is_active,
            disqualified=participant.disqualified,
            disqualification_reason=participant.disqualification_reason,
            final_rank=participant.final_rank,
            final_return_pct=participant.final_return_pct,
            final_sharpe=participant.final_sharpe,
            prize_awarded=participant.prize_awarded,
        )

    except CompetitionValidationError as e:
        logger.error("registration_failed", error=str(e), competition_id=str(competition_id))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("unexpected_error_registering", error=str(e), competition_id=str(competition_id))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to register")


@router.delete("/{competition_id}/register", status_code=status.HTTP_204_NO_CONTENT)
async def withdraw_from_competition(
    competition_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> None:
    """Withdraw from competition (before it starts).

    Args:
        competition_id: Competition UUID
        session: Database session
        current_user: Authenticated user
    """
    logger.info(
        "withdraw_from_competition_request",
        competition_id=str(competition_id),
        user_id=str(current_user.id),
    )

    service = CompetitionService()

    try:
        await service.withdraw_participant(
            session=session,
            competition_id=competition_id,
            user_id=current_user.id,
        )

        await session.commit()

    except CompetitionValidationError as e:
        logger.error("withdrawal_failed", error=str(e), competition_id=str(competition_id))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("unexpected_error_withdrawing", error=str(e), competition_id=str(competition_id))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to withdraw")


@router.get("/{competition_id}/standings", response_model=CompetitionStandingsResponse)
async def get_standings(
    competition_id: UUID,
    *,
    session: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(100, ge=1, le=500, description="Maximum standings to return"),
) -> CompetitionStandingsResponse:
    """Get competition standings.

    Args:
        competition_id: Competition UUID
        limit: Maximum standings to return
        session: Database session

    Returns:
        Competition standings
    """
    logger.info("get_standings_request", competition_id=str(competition_id), limit=limit)

    service = CompetitionService()

    try:
        competition = await service.get_competition(session, competition_id)
        if not competition:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Competition not found")

        standings = await service.get_standings(session, competition_id, limit)

        # Get total participant count
        total = await session.scalar(
            select(func.count(CompetitionParticipant.id)).where(
                CompetitionParticipant.competition_id == competition_id,
                CompetitionParticipant.is_active.is_(True),
            )
        )

        return CompetitionStandingsResponse(
            competition_id=competition_id,
            competition_name=competition.name,
            status=competition.status.value,
            standings=[
                CompetitionStandingResponse(
                    rank=s.rank,
                    user_id=s.user_id,
                    username=s.username,
                    portfolio_id=s.portfolio_id,
                    total_return_pct=s.total_return_pct,
                    sharpe_ratio=s.sharpe_ratio,
                    total_trades=s.total_trades,
                    is_disqualified=s.is_disqualified,
                )
                for s in standings
            ],
            total_participants=total or 0,
            as_of=datetime.now(UTC),
        )

    except CompetitionValidationError as e:
        logger.error("get_standings_failed", error=str(e), competition_id=str(competition_id))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("unexpected_error_getting_standings", error=str(e), competition_id=str(competition_id))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get standings")


@router.get("/{competition_id}/my-standing", response_model=CompetitionStandingResponse)
async def get_my_standing(
    competition_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> CompetitionStandingResponse:
    """Get current user's standing in competition.

    Args:
        competition_id: Competition UUID
        session: Database session
        current_user: Authenticated user

    Returns:
        User's current standing
    """
    logger.info(
        "get_my_standing_request",
        competition_id=str(competition_id),
        user_id=str(current_user.id),
    )

    service = CompetitionService()

    try:
        standings = await service.get_standings(session, competition_id, limit=1000)

        # Find user's standing
        user_standing = next((s for s in standings if s.user_id == current_user.id), None)

        if not user_standing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not registered for this competition",
            )

        return CompetitionStandingResponse(
            rank=user_standing.rank,
            user_id=user_standing.user_id,
            username=user_standing.username,
            portfolio_id=user_standing.portfolio_id,
            total_return_pct=user_standing.total_return_pct,
            sharpe_ratio=user_standing.sharpe_ratio,
            total_trades=user_standing.total_trades,
            is_disqualified=user_standing.is_disqualified,
        )

    except CompetitionValidationError as e:
        logger.error("get_my_standing_failed", error=str(e), competition_id=str(competition_id))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("unexpected_error_getting_my_standing", error=str(e), competition_id=str(competition_id))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get standing")


@router.get("/me/active", response_model=list[UserCompetitionResponse])
async def get_my_active_competitions(
    session: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> list[UserCompetitionResponse]:
    """Get competitions user is participating in.

    Args:
        session: Database session
        current_user: Authenticated user

    Returns:
        List of user's competitions with participant data
    """
    logger.info("get_my_active_competitions_request", user_id=str(current_user.id))

    service = CompetitionService()

    competitions_data = await service.get_user_competitions(
        session=session,
        user_id=current_user.id,
    )

    responses = []
    for item in competitions_data:
        competition = item["competition"]
        participant = item["participant"]

        # Get participant count
        count = await session.scalar(
            select(func.count(CompetitionParticipant.id)).where(
                CompetitionParticipant.competition_id == competition.id,
                CompetitionParticipant.is_active.is_(True),
            )
        )

        comp_response = _competition_to_response(competition, count or 0)
        participant_response = ParticipantResponse(
            id=participant.id,
            competition_id=participant.competition_id,
            user_id=participant.user_id,
            portfolio_id=participant.portfolio_id,
            registered_at=participant.registered_at,
            is_active=participant.is_active,
            disqualified=participant.disqualified,
            disqualification_reason=participant.disqualification_reason,
            final_rank=participant.final_rank,
            final_return_pct=participant.final_return_pct,
            final_sharpe=participant.final_sharpe,
            prize_awarded=participant.prize_awarded,
        )

        responses.append(
            UserCompetitionResponse(
                competition=comp_response,
                participant=participant_response,
            )
        )

    return responses


@router.post("/{competition_id}/disqualify", status_code=status.HTTP_204_NO_CONTENT)
async def disqualify_participant(
    competition_id: UUID,
    data: CompetitionDisqualificationRequest,
    session: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> None:
    """Disqualify a participant for rule violation.

    Only competition creators can disqualify participants.

    Args:
        competition_id: Competition UUID
        data: Disqualification request data
        session: Database session
        current_user: Authenticated user
    """
    logger.info(
        "disqualify_participant_request",
        competition_id=str(competition_id),
        user_id=str(data.user_id),
        by_user=str(current_user.id),
    )

    service = CompetitionService()

    # Check if user is competition creator
    competition = await service.get_competition(session, competition_id)
    if not competition:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Competition not found")

    if competition.created_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only competition creator can disqualify participants",
        )

    try:
        await service.disqualify_participant(
            session=session,
            competition_id=competition_id,
            user_id=data.user_id,
            reason=data.reason,
        )

        await session.commit()

    except CompetitionValidationError as e:
        logger.error("disqualification_failed", error=str(e), competition_id=str(competition_id))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("unexpected_error_disqualifying", error=str(e), competition_id=str(competition_id))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to disqualify participant")

"""Celery tasks for paper trading."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from celery import shared_task

from signalforge.core.database import get_session_context
from signalforge.core.logging import get_logger
from signalforge.models.competition import Competition, CompetitionStatus
from signalforge.models.leaderboard import LeaderboardPeriod, LeaderboardType
from signalforge.paper_trading.competition_service import CompetitionService
from signalforge.paper_trading.leaderboard_service import LeaderboardService
from signalforge.paper_trading.snapshot_service import SnapshotService

logger = get_logger(__name__)


@shared_task(name="paper_trading.create_daily_snapshots")
def create_daily_snapshots() -> dict:
    """Create EOD snapshots for all active portfolios.

    This task should run at market close (4pm ET).

    Returns:
        Dictionary with task results
    """
    logger.info("task_started", task="create_daily_snapshots")

    async def _create_snapshots() -> dict:
        snapshot_service = SnapshotService()

        async with get_session_context() as session:
            snapshots = await snapshot_service.create_snapshots_for_all_portfolios(session)

            return {
                "success": True,
                "snapshots_created": len(snapshots),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If event loop is already running, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        result = loop.run_until_complete(_create_snapshots())
        logger.info("task_completed", task="create_daily_snapshots", result=result)
        return result
    except Exception as e:
        logger.error("task_failed", task="create_daily_snapshots", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }


@shared_task(name="paper_trading.update_position_prices")
def update_position_prices() -> dict:
    """Update current prices for all open positions.

    This task should run every minute during market hours.
    Uses current market data to update position values and unrealized P&L.

    Returns:
        Dictionary with task results
    """
    logger.info("task_started", task="update_position_prices")

    async def _update_prices() -> dict:
        from sqlalchemy import select

        from signalforge.models.paper_trading import PaperPosition

        # Import price fetching utility
        # This would use your existing price data service
        # For now, this is a placeholder implementation

        async with get_session_context() as session:
            # Fetch all positions
            result = await session.execute(select(PaperPosition))
            positions = result.scalars().all()

            updated_count = 0

            for position in positions:
                try:
                    # TODO: Integrate with your price service
                    # For now, we'll skip the actual price update
                    # current_price = await get_current_price(position.symbol)
                    # position.current_price = Decimal(str(current_price))

                    # Calculate unrealized P&L
                    # unrealized_pnl = (position.current_price - position.avg_entry_price) * Decimal(str(position.quantity))
                    # unrealized_pnl_pct = ((position.current_price - position.avg_entry_price) / position.avg_entry_price) * Decimal("100")

                    # position.unrealized_pnl = unrealized_pnl
                    # position.unrealized_pnl_pct = unrealized_pnl_pct

                    updated_count += 1
                except Exception as e:
                    logger.error(
                        "position_price_update_failed",
                        symbol=position.symbol,
                        error=str(e),
                    )

            await session.commit()

            return {
                "success": True,
                "positions_updated": updated_count,
                "timestamp": datetime.now(UTC).isoformat(),
            }

    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        result = loop.run_until_complete(_update_prices())
        logger.info("task_completed", task="update_position_prices", result=result)
        return result
    except Exception as e:
        logger.error("task_failed", task="update_position_prices", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }


@shared_task(name="paper_trading.check_pending_orders")
def check_pending_orders() -> dict:
    """Check and execute pending limit/stop orders.

    This task should run every minute during market hours.
    Checks if limit or stop orders should be executed based on current prices.

    Returns:
        Dictionary with task results
    """
    logger.info("task_started", task="check_pending_orders")

    async def _check_orders() -> dict:
        from sqlalchemy import select

        from signalforge.models.paper_trading import (
            OrderStatus,
            OrderType,
            PaperOrder,
        )

        async with get_session_context() as session:
            # Fetch all pending limit and stop orders
            result = await session.execute(
                select(PaperOrder).where(
                    PaperOrder.status == OrderStatus.PENDING,
                    PaperOrder.order_type.in_([OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT]),
                )
            )
            orders = result.scalars().all()

            executed_count = 0
            expired_count = 0

            for order in orders:
                try:
                    # Check if order has expired
                    if order.expires_at and datetime.now(UTC) > order.expires_at:
                        order.status = OrderStatus.EXPIRED
                        expired_count += 1
                        continue

                    # TODO: Integrate with your price service to check execution conditions
                    # current_price = await get_current_price(order.symbol)

                    # Check if limit order should execute
                    # if order.order_type == OrderType.LIMIT:
                    #     if order.side == OrderSide.BUY and current_price <= order.limit_price:
                    #         await execute_order(order, current_price, session)
                    #         executed_count += 1
                    #     elif order.side == OrderSide.SELL and current_price >= order.limit_price:
                    #         await execute_order(order, current_price, session)
                    #         executed_count += 1

                    # Check if stop order should execute
                    # elif order.order_type == OrderType.STOP:
                    #     if order.side == OrderSide.BUY and current_price >= order.stop_price:
                    #         await execute_order(order, current_price, session)
                    #         executed_count += 1
                    #     elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                    #         await execute_order(order, current_price, session)
                    #         executed_count += 1

                except Exception as e:
                    logger.error(
                        "order_check_failed",
                        order_id=str(order.id),
                        symbol=order.symbol,
                        error=str(e),
                    )

            await session.commit()

            return {
                "success": True,
                "orders_executed": executed_count,
                "orders_expired": expired_count,
                "timestamp": datetime.now(UTC).isoformat(),
            }

    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        result = loop.run_until_complete(_check_orders())
        logger.info("task_completed", task="check_pending_orders", result=result)
        return result
    except Exception as e:
        logger.error("task_failed", task="check_pending_orders", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }


@shared_task(name="paper_trading.calculate_leaderboards")
def calculate_leaderboards() -> dict:
    """Calculate all leaderboards.

    This task should run:
    - Daily: After daily snapshots are created (after market close)
    - Weekly: On Sunday evenings
    - Monthly: On the first day of each month

    Returns:
        Dictionary with task results
    """
    logger.info("task_started", task="calculate_leaderboards")

    async def _calculate_leaderboards() -> dict:
        async with get_session_context() as session:
            service = LeaderboardService(session)
            results = {}

            periods_to_calculate = [
                LeaderboardPeriod.DAILY,
                LeaderboardPeriod.WEEKLY,
                LeaderboardPeriod.MONTHLY,
                LeaderboardPeriod.ALL_TIME,
            ]

            leaderboard_types = [
                LeaderboardType.TOTAL_RETURN,
                LeaderboardType.SHARPE_RATIO,
                LeaderboardType.RISK_ADJUSTED,
            ]

            for period in periods_to_calculate:
                for lb_type in leaderboard_types:
                    try:
                        entries = await service.calculate_leaderboard(
                            period=period,
                            leaderboard_type=lb_type,
                        )
                        results[f"{period.value}_{lb_type.value}"] = {
                            "success": True,
                            "entries_count": len(entries),
                        }
                        logger.info(
                            "leaderboard_calculated",
                            period=period.value,
                            leaderboard_type=lb_type.value,
                            entries_count=len(entries),
                        )
                    except Exception as e:
                        results[f"{period.value}_{lb_type.value}"] = {
                            "success": False,
                            "error": str(e),
                        }
                        logger.error(
                            "leaderboard_calculation_failed",
                            period=period.value,
                            leaderboard_type=lb_type.value,
                            error=str(e),
                        )

            await session.commit()

            total_success = sum(
                1 for r in results.values() if r.get("success", False)
            )
            total_calculated = len(results)

            return {
                "success": True,
                "total_calculated": total_calculated,
                "successful": total_success,
                "failed": total_calculated - total_success,
                "details": results,
                "timestamp": datetime.now(UTC).isoformat(),
            }

    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        result = loop.run_until_complete(_calculate_leaderboards())
        logger.info("task_completed", task="calculate_leaderboards", result=result)
        return result
    except Exception as e:
        logger.error("task_failed", task="calculate_leaderboards", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }


@shared_task(name="paper_trading.update_competition_statuses")
def update_competition_statuses() -> dict:
    """Update competition statuses based on dates.

    Runs every hour to check if competitions need status updates.
    Handles transitions: DRAFT -> REGISTRATION_OPEN -> ACTIVE -> COMPLETED

    Returns:
        Dictionary with task results
    """
    logger.info("task_started", task="update_competition_statuses")

    async def _update_statuses() -> dict:
        from sqlalchemy import select

        results = {
            "updated": [],
            "errors": [],
            "total_checked": 0,
        }

        try:
            async with get_session_context() as session:
                service = CompetitionService()

                # Get all non-completed, non-cancelled competitions
                query = select(Competition).where(
                    Competition.status.in_([
                        CompetitionStatus.DRAFT,
                        CompetitionStatus.REGISTRATION_OPEN,
                        CompetitionStatus.ACTIVE,
                    ])
                )

                result = await session.execute(query)
                competitions = result.scalars().all()

                results["total_checked"] = len(competitions)

                for competition in competitions:
                    try:
                        old_status = competition.status
                        updated_competition = await service.update_competition_status(
                            session=session,
                            competition_id=competition.id,
                        )

                        if updated_competition.status != old_status:
                            results["updated"].append({
                                "competition_id": str(competition.id),
                                "old_status": old_status.value,
                                "new_status": updated_competition.status.value,
                            })
                            logger.info(
                                "competition_status_updated",
                                competition_id=str(competition.id),
                                old_status=old_status.value,
                                new_status=updated_competition.status.value,
                            )

                    except Exception as e:
                        logger.error(
                            "error_updating_competition_status",
                            competition_id=str(competition.id),
                            error=str(e),
                        )
                        results["errors"].append({
                            "competition_id": str(competition.id),
                            "error": str(e),
                        })

                await session.commit()

        except Exception as e:
            logger.error("update_statuses_failed", error=str(e))
            results["task_error"] = str(e)

        return results

    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        result = loop.run_until_complete(_update_statuses())
        logger.info(
            "task_completed",
            task="update_competition_statuses",
            updated_count=len(result.get("updated", [])),
            error_count=len(result.get("errors", [])),
        )
        return result
    except Exception as e:
        logger.error("task_failed", task="update_competition_statuses", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }


@shared_task(name="paper_trading.finalize_completed_competitions")
def finalize_completed_competitions() -> dict:
    """Finalize competitions that have ended.

    Runs hourly to check for competitions that have passed their end date
    and need final standings calculated and prizes awarded.

    Returns:
        Dictionary with task results
    """
    logger.info("task_started", task="finalize_completed_competitions")

    async def _finalize_competitions() -> dict:
        from sqlalchemy import func, select

        from signalforge.models.competition import CompetitionParticipant

        results = {
            "finalized": [],
            "errors": [],
            "total_checked": 0,
        }

        try:
            async with get_session_context() as session:
                service = CompetitionService()
                now = datetime.now(UTC)

                # Get competitions that are completed but not yet finalized
                query = (
                    select(Competition)
                    .where(
                        Competition.status == CompetitionStatus.COMPLETED,
                        Competition.competition_end <= now,
                    )
                    .order_by(Competition.competition_end.asc())
                )

                result = await session.execute(query)
                competitions = result.scalars().all()

                results["total_checked"] = len(competitions)

                for competition in competitions:
                    try:
                        # Check if already finalized
                        ranked_count = await session.scalar(
                            select(func.count(CompetitionParticipant.id)).where(
                                CompetitionParticipant.competition_id == competition.id,
                                CompetitionParticipant.final_rank.isnot(None),
                            )
                        )

                        if ranked_count and ranked_count > 0:
                            logger.debug(
                                "competition_already_finalized",
                                competition_id=str(competition.id),
                            )
                            continue

                        # Finalize the competition
                        await service.finalize_competition(
                            session=session,
                            competition_id=competition.id,
                        )

                        results["finalized"].append({
                            "competition_id": str(competition.id),
                            "competition_name": competition.name,
                            "ended_at": competition.competition_end.isoformat(),
                        })

                        logger.info(
                            "competition_finalized",
                            competition_id=str(competition.id),
                            competition_name=competition.name,
                        )

                    except Exception as e:
                        logger.error(
                            "error_finalizing_competition",
                            competition_id=str(competition.id),
                            error=str(e),
                        )
                        results["errors"].append({
                            "competition_id": str(competition.id),
                            "error": str(e),
                        })

                await session.commit()

        except Exception as e:
            logger.error("finalize_competitions_failed", error=str(e))
            results["task_error"] = str(e)

        return results

    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        result = loop.run_until_complete(_finalize_competitions())
        logger.info(
            "task_completed",
            task="finalize_completed_competitions",
            finalized_count=len(result.get("finalized", [])),
            error_count=len(result.get("errors", [])),
        )
        return result
    except Exception as e:
        logger.error("task_failed", task="finalize_completed_competitions", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }


@shared_task(name="paper_trading.calculate_competition_standings")
def calculate_competition_standings() -> dict:
    """Update standings for all active competitions.

    Runs every 15 minutes during market hours to update leaderboards.
    This task calculates standings but doesn't persist them - they are
    calculated on-demand via API for fresh results.

    Returns:
        Dictionary with task results
    """
    logger.info("task_started", task="calculate_competition_standings")

    async def _calculate_standings() -> dict:
        from sqlalchemy import select

        results = {
            "calculated": [],
            "errors": [],
            "total_competitions": 0,
        }

        try:
            async with get_session_context() as session:
                service = CompetitionService()

                # Get all active competitions
                query = select(Competition).where(Competition.status == CompetitionStatus.ACTIVE)

                result = await session.execute(query)
                competitions = result.scalars().all()

                results["total_competitions"] = len(competitions)

                for competition in competitions:
                    try:
                        # Calculate standings
                        standings = await service.get_standings(
                            session=session,
                            competition_id=competition.id,
                            limit=100,
                        )

                        results["calculated"].append({
                            "competition_id": str(competition.id),
                            "competition_name": competition.name,
                            "participant_count": len(standings),
                        })

                        logger.debug(
                            "competition_standings_calculated",
                            competition_id=str(competition.id),
                            participant_count=len(standings),
                        )

                    except Exception as e:
                        logger.error(
                            "error_calculating_standings",
                            competition_id=str(competition.id),
                            error=str(e),
                        )
                        results["errors"].append({
                            "competition_id": str(competition.id),
                            "error": str(e),
                        })

        except Exception as e:
            logger.error("calculate_standings_failed", error=str(e))
            results["task_error"] = str(e)

        return results

    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        result = loop.run_until_complete(_calculate_standings())
        logger.info(
            "task_completed",
            task="calculate_competition_standings",
            calculated_count=len(result.get("calculated", [])),
            error_count=len(result.get("errors", [])),
        )
        return result
    except Exception as e:
        logger.error("task_failed", task="calculate_competition_standings", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }

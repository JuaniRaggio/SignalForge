"""Database dependencies."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.core.database import async_session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency that yields database sessions."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

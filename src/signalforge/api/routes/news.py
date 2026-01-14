"""News routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.api.dependencies.auth import get_current_active_user
from signalforge.api.dependencies.database import get_db
from signalforge.models.news import NewsArticle
from signalforge.models.user import User
from signalforge.schemas.news import NewsArticleResponse, NewsListResponse

router = APIRouter()


@router.get("/", response_model=NewsListResponse)
async def get_news(
    db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
    source: str | None = Query(None, description="Filter by source"),
    limit: int = Query(20, ge=1, le=100, description="Number of articles"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
) -> NewsListResponse:
    """Get news articles."""
    query = select(NewsArticle)

    if source:
        query = query.where(NewsArticle.source == source)

    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    query = query.order_by(NewsArticle.published_at.desc().nulls_last())
    query = query.offset(offset).limit(limit)

    result = await db.execute(query)
    articles = result.scalars().all()

    return NewsListResponse(
        articles=[
            NewsArticleResponse(
                id=a.id,
                url=a.url,
                title=a.title,
                source=a.source,
                published_at=a.published_at,
                summary=a.summary,
                categories=a.categories,
                symbols=a.symbols,
                created_at=a.created_at,
            )
            for a in articles
        ],
        count=len(articles),
        total=total,
    )

"""News schemas."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class NewsArticleResponse(BaseModel):
    """Schema for news article response."""

    id: UUID
    url: str
    title: str
    source: str
    published_at: datetime | None
    summary: str | None
    categories: list[str]
    symbols: list[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class NewsListResponse(BaseModel):
    """Schema for news list response."""

    articles: list[NewsArticleResponse]
    count: int
    total: int

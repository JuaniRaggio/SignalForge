"""Pydantic schemas for API requests and responses."""

from signalforge.schemas.auth import (
    RefreshTokenRequest,
    TokenResponse,
    UserLogin,
    UserRegister,
    UserResponse,
)
from signalforge.schemas.base import (
    BaseResponse,
    CountResponse,
    EmptyResponse,
    ErrorDetail,
    ErrorResponse,
    HealthCheckResponse,
    IdResponse,
    PaginationInfo,
    PaginationParams,
    ResponseMetadata,
    create_error_response,
    create_success_response,
)
from signalforge.schemas.ingestion import (
    DataQualityStatus,
    RSSArticleRaw,
    RSSArticleValidated,
    RSSValidationResult,
    YahooDataValidationResult,
    YahooPriceDataRaw,
    YahooPriceDataValidated,
    validate_rss_articles,
    validate_yahoo_data,
)
from signalforge.schemas.market import (
    PriceData,
    PriceHistoryRequest,
    PriceHistoryResponse,
    SymbolInfo,
    SymbolListResponse,
)
from signalforge.schemas.news import NewsArticleResponse, NewsListResponse

__all__ = [
    # Base response patterns
    "BaseResponse",
    "EmptyResponse",
    "ErrorResponse",
    "ErrorDetail",
    "ResponseMetadata",
    "PaginationInfo",
    "PaginationParams",
    "HealthCheckResponse",
    "CountResponse",
    "IdResponse",
    "create_success_response",
    "create_error_response",
    # Auth
    "RefreshTokenRequest",
    "TokenResponse",
    "UserLogin",
    "UserRegister",
    "UserResponse",
    # Market
    "PriceData",
    "PriceHistoryRequest",
    "PriceHistoryResponse",
    "SymbolInfo",
    "SymbolListResponse",
    # News
    "NewsArticleResponse",
    "NewsListResponse",
    # Ingestion validation
    "DataQualityStatus",
    "YahooPriceDataRaw",
    "YahooPriceDataValidated",
    "YahooDataValidationResult",
    "RSSArticleRaw",
    "RSSArticleValidated",
    "RSSValidationResult",
    "validate_yahoo_data",
    "validate_rss_articles",
]

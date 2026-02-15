"""API Key manager for generation, validation, and revocation."""

import secrets
from datetime import UTC, datetime
from uuid import UUID

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.core.security import get_password_hash, verify_password
from signalforge.models.api_key import APIKey, SubscriptionTier

logger = structlog.get_logger(__name__)


class APIKeyManager:
    """
    Manages API key lifecycle operations.

    Provides methods for generating, validating, and revoking API keys.
    Uses bcrypt for secure key hashing.
    """

    @staticmethod
    def _generate_key() -> str:
        """
        Generate a cryptographically secure API key.

        Returns:
            A URL-safe base64-encoded random key (43 characters)
        """
        return secrets.token_urlsafe(32)

    @staticmethod
    async def generate_api_key(
        db: AsyncSession,
        user_id: UUID,
        name: str,
        tier: SubscriptionTier,
        scopes: list[str] | None = None,
        rate_limit_override: int | None = None,
        burst_limit_override: int | None = None,
        expires_at: datetime | None = None,
    ) -> tuple[str, APIKey]:
        """
        Generate a new API key for a user.

        Args:
            db: Database session
            user_id: User ID to associate the key with
            name: Descriptive name for the key
            tier: Subscription tier for the key
            scopes: List of allowed scopes (defaults to all)
            rate_limit_override: Custom rate limit (overrides tier default)
            burst_limit_override: Custom burst limit (overrides tier default)
            expires_at: Optional expiration datetime

        Returns:
            Tuple of (plain_key, api_key_record)
            The plain key should be shown to the user only once.

        Raises:
            ValueError: If invalid parameters are provided
        """
        if not name or not name.strip():
            raise ValueError("API key name cannot be empty")

        if scopes is None:
            scopes = ["read", "write"]

        plain_key = APIKeyManager._generate_key()
        key_hash = get_password_hash(plain_key)

        api_key = APIKey(
            user_id=user_id,
            key_hash=key_hash,
            name=name.strip(),
            tier=tier,
            scopes=scopes,
            rate_limit_override=rate_limit_override,
            burst_limit_override=burst_limit_override,
            expires_at=expires_at,
            is_active=True,
        )

        db.add(api_key)
        await db.flush()
        await db.refresh(api_key)

        logger.info(
            "api_key_generated",
            key_id=str(api_key.id),
            user_id=str(user_id),
            name=name,
            tier=tier.value,
        )

        return plain_key, api_key

    @staticmethod
    async def validate_api_key(db: AsyncSession, plain_key: str) -> APIKey | None:
        """
        Validate an API key and return the associated record.

        Args:
            db: Database session
            plain_key: The plain text API key to validate

        Returns:
            APIKey record if valid, None if invalid

        Side effects:
            Updates last_used_at timestamp if key is valid
        """
        if not plain_key:
            return None

        result = await db.execute(
            select(APIKey).where(APIKey.is_active == True)  # noqa: E712
        )
        api_keys = result.scalars().all()

        for api_key in api_keys:
            if verify_password(plain_key, api_key.key_hash):
                if api_key.expires_at and api_key.expires_at < datetime.now(UTC):
                    logger.warning(
                        "api_key_expired",
                        key_id=str(api_key.id),
                        expired_at=api_key.expires_at.isoformat(),
                    )
                    return None

                api_key.last_used_at = datetime.now(UTC)
                await db.flush()

                logger.debug(
                    "api_key_validated",
                    key_id=str(api_key.id),
                    user_id=str(api_key.user_id),
                )

                return api_key

        logger.warning("api_key_validation_failed")
        return None

    @staticmethod
    async def revoke_api_key(
        db: AsyncSession,
        key_id: UUID,
        user_id: UUID,
    ) -> bool:
        """
        Revoke an API key.

        Args:
            db: Database session
            key_id: ID of the key to revoke
            user_id: User ID (for authorization check)

        Returns:
            True if key was revoked, False if not found or unauthorized
        """
        result = await db.execute(
            select(APIKey).where(
                APIKey.id == key_id,
                APIKey.user_id == user_id,
            )
        )
        api_key = result.scalar_one_or_none()

        if not api_key:
            logger.warning(
                "api_key_revoke_failed_not_found",
                key_id=str(key_id),
                user_id=str(user_id),
            )
            return False

        api_key.is_active = False
        await db.flush()

        logger.info(
            "api_key_revoked",
            key_id=str(key_id),
            user_id=str(user_id),
        )

        return True

    @staticmethod
    async def list_user_keys(db: AsyncSession, user_id: UUID) -> list[APIKey]:
        """
        List all API keys for a user.

        Args:
            db: Database session
            user_id: User ID to list keys for

        Returns:
            List of APIKey records
        """
        result = await db.execute(
            select(APIKey)
            .where(APIKey.user_id == user_id)
            .order_by(APIKey.created_at.desc())
        )

        return list(result.scalars().all())

    @staticmethod
    async def get_key_by_id(
        db: AsyncSession,
        key_id: UUID,
        user_id: UUID,
    ) -> APIKey | None:
        """
        Get an API key by ID.

        Args:
            db: Database session
            key_id: API key ID
            user_id: User ID (for authorization check)

        Returns:
            APIKey record if found and authorized, None otherwise
        """
        result = await db.execute(
            select(APIKey).where(
                APIKey.id == key_id,
                APIKey.user_id == user_id,
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def check_scope(api_key: APIKey, required_scope: str) -> bool:
        """
        Check if an API key has a required scope.

        Args:
            api_key: APIKey record
            required_scope: Scope to check for

        Returns:
            True if key has the scope, False otherwise
        """
        return required_scope in api_key.scopes

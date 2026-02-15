"""Document version management."""

import difflib
from datetime import UTC, datetime

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from .schemas import FinancialDocument

logger = structlog.get_logger(__name__)


class DocumentVersionManager:
    """Manage document versions."""

    def __init__(self, session: AsyncSession | None = None) -> None:
        """
        Initialize version manager.

        Args:
            session: Optional database session for persistence
        """
        self.session = session
        # In-memory storage for demo purposes when no session provided
        self._versions: dict[str, list[tuple[int, datetime, str, str]]] = {}

    async def create_version(
        self,
        document_id: str,
        new_content: str,
        reason: str = "update",
    ) -> int:
        """
        Create new version of document.

        Args:
            document_id: Document identifier
            new_content: New content for this version
            reason: Reason for version creation

        Returns:
            New version number
        """
        if self.session:
            # TODO: Database persistence
            # For now, fallback to in-memory
            pass

        # Get existing versions
        versions = self._versions.get(document_id, [])

        # Calculate next version number
        next_version = max(v[0] for v in versions) + 1 if versions else 1

        # Store version
        timestamp = datetime.now(UTC)
        versions.append((next_version, timestamp, reason, new_content))
        self._versions[document_id] = versions

        logger.info(
            "created_version",
            document_id=document_id,
            version=next_version,
            reason=reason,
        )

        return next_version

    async def get_version(
        self, document_id: str, version: int
    ) -> FinancialDocument | None:
        """
        Get specific version of document.

        Args:
            document_id: Document identifier
            version: Version number to retrieve

        Returns:
            FinancialDocument if found, None otherwise
        """
        if self.session:
            # TODO: Database query
            pass

        versions = self._versions.get(document_id, [])
        for v_num, _timestamp, _reason, _content in versions:
            if v_num == version:
                logger.debug(
                    "retrieved_version",
                    document_id=document_id,
                    version=version,
                )
                # Note: This is a simplified return; in production,
                # would reconstruct full FinancialDocument from storage
                return None  # Placeholder

        logger.warning(
            "version_not_found",
            document_id=document_id,
            version=version,
        )
        return None

    async def get_latest(self, document_id: str) -> FinancialDocument | None:
        """
        Get latest version of document.

        Args:
            document_id: Document identifier

        Returns:
            Latest FinancialDocument if found, None otherwise
        """
        if self.session:
            # TODO: Database query
            pass

        versions = self._versions.get(document_id, [])
        if not versions:
            return None

        latest_version = max(v[0] for v in versions)
        return await self.get_version(document_id, latest_version)

    async def get_history(
        self, document_id: str
    ) -> list[tuple[int, datetime, str]]:
        """
        Get version history.

        Args:
            document_id: Document identifier

        Returns:
            List of (version, timestamp, reason) tuples
        """
        if self.session:
            # TODO: Database query
            pass

        versions = self._versions.get(document_id, [])
        history = [(v_num, timestamp, reason) for v_num, timestamp, reason, _ in versions]

        logger.debug(
            "retrieved_history",
            document_id=document_id,
            version_count=len(history),
        )

        return history

    def compute_diff(self, old_content: str, new_content: str) -> dict[str, object]:
        """
        Compute difference between versions.

        Args:
            old_content: Previous version content
            new_content: New version content

        Returns:
            Dictionary with diff information
        """
        # Split into lines for line-by-line diff
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        # Compute unified diff
        diff = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile="old",
                tofile="new",
                lineterm="",
            )
        )

        # Compute basic statistics
        additions = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
        deletions = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))

        # Compute similarity ratio
        similarity = difflib.SequenceMatcher(
            None,
            old_content,
            new_content,
        ).ratio()

        result = {
            "diff": diff,
            "additions": additions,
            "deletions": deletions,
            "similarity": similarity,
            "changed": old_content != new_content,
        }

        logger.debug(
            "computed_diff",
            additions=additions,
            deletions=deletions,
            similarity=similarity,
        )

        return result

"""Test configuration and fixtures."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from signalforge.api.dependencies.database import get_db
from signalforge.api.main import app
from signalforge.core.config import get_settings
from signalforge.core.redis import get_redis
from signalforge.core.security import create_access_token, get_password_hash
from signalforge.models.base import Base
from signalforge.models.user import User, UserType

settings = get_settings()

# Only replace the database name at the end of the URL, not the username
_base_url = settings.database_url
if _base_url.endswith("/signalforge"):
    TEST_DATABASE_URL = _base_url[:-12] + "/signalforge_test"
else:
    TEST_DATABASE_URL = _base_url


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


def _drop_all_tables(connection: Any) -> None:
    """Drop all tables and enum types."""
    from sqlalchemy import text

    # Drop all tables first
    Base.metadata.drop_all(connection)

    # Drop enum types to prevent conflicts on next create
    enum_types = [
        "activitytype",
        "competitionstatus",
        "competitiontype",
        "eventstatus",
        "eventtype",
        "experiencelevel",
        "investmenthorizon",
        "modeltype",
        "orderside",
        "orderstatus",
        "ordertype",
        "portfoliostatus",
        "risktolerance",
        "usertype",
    ]
    for enum_type in enum_types:
        try:
            connection.execute(text(f"DROP TYPE IF EXISTS {enum_type} CASCADE"))
        except Exception:
            pass


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        poolclass=StaticPool,
    )

    # Drop existing tables/types and create fresh
    async with engine.begin() as conn:
        await conn.run_sync(_drop_all_tables)
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(_drop_all_tables)

    await engine.dispose()


def create_mock_redis() -> MagicMock:
    """Create a mock Redis client for testing."""
    mock_redis = MagicMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.incr = AsyncMock(return_value=1)
    mock_redis.expire = AsyncMock(return_value=True)
    mock_redis.setex = AsyncMock(return_value=True)
    mock_redis.exists = AsyncMock(return_value=0)
    mock_redis.ping = AsyncMock(return_value=True)
    mock_redis.close = AsyncMock(return_value=None)
    return mock_redis


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession, redis_client: StatefulRedisMock) -> AsyncGenerator[AsyncClient, None]:
    """Create test client with database session and redis override."""

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    async def override_get_redis() -> StatefulRedisMock:
        return redis_client

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_redis] = override_get_redis

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest_asyncio.fixture(scope="function")
async def test_user(db_session: AsyncSession) -> User:
    """Create a test user in the database."""
    user = User(
        id=uuid4(),
        email="testuser@example.com",
        username="testuser",
        hashed_password=get_password_hash("testpassword123"),
        user_type=UserType.CASUAL_OBSERVER,
        is_active=True,
    )
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)
    return user


class StatefulRedisMock:
    """A stateful Redis mock that actually tracks values."""

    def __init__(self) -> None:
        self._data: dict[str, str | int | float | bytes] = {}
        self._hashes: dict[str, dict[str, str | int | float]] = {}
        self._lists: dict[str, list[str]] = {}
        self._sets: dict[str, set[str]] = {}
        self._zsets: dict[str, dict[str, float]] = {}

    async def get(self, key: str) -> str | bytes | None:
        return self._data.get(key)

    async def set(self, key: str, value: str | bytes, ex: int | None = None, px: int | None = None, nx: bool = False, xx: bool = False) -> bool:
        if nx and key in self._data:
            return False
        if xx and key not in self._data:
            return False
        self._data[key] = value
        return True

    async def setex(self, key: str, seconds: int, value: str | bytes) -> bool:
        self._data[key] = value
        return True

    async def delete(self, *keys: str) -> int:
        count = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                count += 1
        return count

    async def exists(self, *keys: str) -> int:
        return sum(1 for k in keys if k in self._data)

    async def expire(self, key: str, seconds: int) -> bool:
        return key in self._data

    async def ttl(self, key: str) -> int:
        return 60 if key in self._data else -2

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        pass

    async def incr(self, key: str) -> int:
        val = int(self._data.get(key, 0)) + 1
        self._data[key] = val
        return val

    async def incrby(self, key: str, amount: int) -> int:
        val = int(self._data.get(key, 0)) + amount
        self._data[key] = val
        return val

    async def incrbyfloat(self, key: str, amount: float) -> float:
        val = float(self._data.get(key, 0)) + amount
        self._data[key] = val
        return val

    async def decr(self, key: str) -> int:
        val = int(self._data.get(key, 0)) - 1
        self._data[key] = val
        return val

    async def decrby(self, key: str, amount: int) -> int:
        val = int(self._data.get(key, 0)) - amount
        self._data[key] = val
        return val

    async def hget(self, name: str, key: str) -> str | int | float | None:
        return self._hashes.get(name, {}).get(key)

    async def hset(self, name: str, key: str | None = None, value: str | int | float | None = None, mapping: dict | None = None) -> int:
        if name not in self._hashes:
            self._hashes[name] = {}
        if mapping:
            self._hashes[name].update(mapping)
            return len(mapping)
        if key is not None:
            self._hashes[name][key] = value
            return 1
        return 0

    async def hgetall(self, name: str) -> dict[str, str | int | float]:
        return self._hashes.get(name, {})

    async def hincrby(self, name: str, key: str, amount: int = 1) -> int:
        if name not in self._hashes:
            self._hashes[name] = {}
        val = int(self._hashes[name].get(key, 0)) + amount
        self._hashes[name][key] = val
        return val

    async def hincrbyfloat(self, name: str, key: str, amount: float) -> float:
        if name not in self._hashes:
            self._hashes[name] = {}
        val = float(self._hashes[name].get(key, 0)) + amount
        self._hashes[name][key] = val
        return val

    async def lpush(self, name: str, *values: str) -> int:
        if name not in self._lists:
            self._lists[name] = []
        for v in values:
            self._lists[name].insert(0, v)
        return len(self._lists[name])

    async def rpush(self, name: str, *values: str) -> int:
        if name not in self._lists:
            self._lists[name] = []
        self._lists[name].extend(values)
        return len(self._lists[name])

    async def lpop(self, name: str, count: int | None = None) -> str | list[str] | None:
        if name not in self._lists or not self._lists[name]:
            return None
        if count is None:
            return self._lists[name].pop(0)
        result = []
        for _ in range(min(count, len(self._lists[name]))):
            result.append(self._lists[name].pop(0))
        return result

    async def rpop(self, name: str, count: int | None = None) -> str | list[str] | None:
        if name not in self._lists or not self._lists[name]:
            return None
        if count is None:
            return self._lists[name].pop()
        result = []
        for _ in range(min(count, len(self._lists[name]))):
            result.append(self._lists[name].pop())
        return result

    async def lrange(self, name: str, start: int, end: int) -> list[str]:
        if name not in self._lists:
            return []
        if end == -1:
            return self._lists[name][start:]
        return self._lists[name][start:end + 1]

    async def llen(self, name: str) -> int:
        return len(self._lists.get(name, []))

    async def sadd(self, name: str, *values: str) -> int:
        if name not in self._sets:
            self._sets[name] = set()
        before = len(self._sets[name])
        self._sets[name].update(values)
        return len(self._sets[name]) - before

    async def smembers(self, name: str) -> set[str]:
        return self._sets.get(name, set())

    async def sismember(self, name: str, value: str) -> bool:
        return value in self._sets.get(name, set())

    async def zadd(self, name: str, mapping: dict[str, float], nx: bool = False, xx: bool = False, gt: bool = False, lt: bool = False) -> int:
        if name not in self._zsets:
            self._zsets[name] = {}
        added = 0
        for member, score in mapping.items():
            if member not in self._zsets[name]:
                added += 1
            self._zsets[name][member] = score
        return added

    async def zrange(self, name: str, start: int, end: int, withscores: bool = False) -> list:
        if name not in self._zsets:
            return []
        items = sorted(self._zsets[name].items(), key=lambda x: x[1])
        if end == -1:
            items = items[start:]
        else:
            items = items[start:end + 1]
        if withscores:
            return [(k, v) for k, v in items]
        return [k for k, v in items]

    async def zrangebyscore(self, name: str, min_score: float, max_score: float, withscores: bool = False) -> list:
        if name not in self._zsets:
            return []
        items = [(k, v) for k, v in self._zsets[name].items() if min_score <= v <= max_score]
        items.sort(key=lambda x: x[1])
        if withscores:
            return items
        return [k for k, v in items]

    async def zrem(self, name: str, *members: str) -> int:
        if name not in self._zsets:
            return 0
        removed = 0
        for member in members:
            if member in self._zsets[name]:
                del self._zsets[name][member]
                removed += 1
        return removed

    async def zscore(self, name: str, member: str) -> float | None:
        return self._zsets.get(name, {}).get(member)

    async def scan(self, cursor: int = 0, match: str | None = None, count: int = 10) -> tuple[int, list[str]]:
        """Scan keys matching a pattern."""
        import fnmatch
        keys = list(self._data.keys())
        if match:
            pattern = match.replace("*", ".*")
            keys = [k for k in keys if fnmatch.fnmatch(k, match)]
        return (0, keys)

    def pipeline(self, transaction: bool = True) -> "RedisPipelineMock":
        return RedisPipelineMock(self)


class RedisPipelineMock:
    """Mock Redis pipeline."""

    def __init__(self, redis: StatefulRedisMock) -> None:
        self._redis = redis
        self._commands: list[tuple[str, tuple, dict]] = []

    def __enter__(self) -> "RedisPipelineMock":
        return self

    def __exit__(self, *args) -> None:
        pass

    async def __aenter__(self) -> "RedisPipelineMock":
        return self

    async def __aexit__(self, *args) -> None:
        pass

    def incr(self, key: str) -> "RedisPipelineMock":
        self._commands.append(("incr", (key,), {}))
        return self

    def expire(self, key: str, seconds: int) -> "RedisPipelineMock":
        self._commands.append(("expire", (key, seconds), {}))
        return self

    def hincrby(self, name: str, key: str, amount: int = 1) -> "RedisPipelineMock":
        self._commands.append(("hincrby", (name, key, amount), {}))
        return self

    def hincrbyfloat(self, name: str, key: str, amount: float) -> "RedisPipelineMock":
        self._commands.append(("hincrbyfloat", (name, key, amount), {}))
        return self

    async def execute(self) -> list:
        results = []
        for cmd, args, kwargs in self._commands:
            method = getattr(self._redis, cmd)
            result = await method(*args, **kwargs)
            results.append(result)
        self._commands = []
        return results


@pytest_asyncio.fixture(scope="function")
async def redis_client() -> StatefulRedisMock:
    """Create a stateful Redis mock for testing."""
    return StatefulRedisMock()


@pytest_asyncio.fixture(scope="function")
async def authenticated_user_token(test_user: User) -> str:
    """Create an access token for the test user."""
    return create_access_token(data={"sub": str(test_user.id)})

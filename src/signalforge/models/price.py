"""Price model for OHLCV data with TimescaleDB hypertable support."""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import BigInteger, DateTime, Numeric, String, text
from sqlalchemy.orm import Mapped, mapped_column

from signalforge.models.base import Base


class Price(Base):
    """OHLCV price data model.

    This table uses a composite primary key (symbol, timestamp) and
    should be converted to a TimescaleDB hypertable after creation.
    """

    __tablename__ = "prices"

    symbol: Mapped[str] = mapped_column(
        String(20),
        primary_key=True,
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        primary_key=True,
        nullable=False,
    )
    open: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=6),
        nullable=False,
    )
    high: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=6),
        nullable=False,
    )
    low: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=6),
        nullable=False,
    )
    close: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=6),
        nullable=False,
    )
    volume: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )
    adj_close: Mapped[Decimal | None] = mapped_column(
        Numeric(precision=18, scale=6),
        nullable=True,
    )


# SQL to convert prices table to TimescaleDB hypertable
HYPERTABLE_SQL = text("""
SELECT create_hypertable(
    'prices',
    by_range('timestamp'),
    if_not_exists => TRUE
);
""")

# SQL to add compression policy (compress data older than 30 days)
COMPRESSION_POLICY_SQL = text("""
ALTER TABLE prices SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);

SELECT add_compression_policy(
    'prices',
    compress_after => INTERVAL '30 days',
    if_not_exists => TRUE
);
""")

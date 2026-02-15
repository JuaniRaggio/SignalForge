"""Add paper trading tables for positions and portfolio snapshots.

This migration:
1. Creates the paper_positions table for tracking individual trades
2. Creates the portfolio_snapshots table for daily portfolio state
3. Creates indexes for efficient querying

Revision ID: 003_add_paper_trading_tables
Revises: 002_add_document_embeddings
Create Date: 2026-01-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "003_add_paper_trading_tables"
down_revision: str = "002_add_document_embeddings"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create paper trading tables and indexes."""
    # Create paper_positions table
    op.create_table(
        "paper_positions",
        sa.Column(
            "id",
            sa.Integer(),
            nullable=False,
            autoincrement=True,
        ),
        sa.Column(
            "signal_id",
            sa.String(36),
            nullable=False,
            comment="UUID of originating signal",
        ),
        sa.Column(
            "symbol",
            sa.String(10),
            nullable=False,
            comment="Stock symbol",
        ),
        sa.Column(
            "direction",
            sa.String(10),
            nullable=False,
            comment="long or short",
        ),
        sa.Column(
            "entry_date",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="Position entry date",
        ),
        sa.Column(
            "entry_price",
            sa.Numeric(12, 4),
            nullable=False,
            comment="Entry price",
        ),
        sa.Column(
            "shares",
            sa.Integer(),
            nullable=False,
            comment="Number of shares",
        ),
        sa.Column(
            "stop_loss",
            sa.Numeric(12, 4),
            nullable=False,
            comment="Stop loss price",
        ),
        sa.Column(
            "take_profit",
            sa.Numeric(12, 4),
            nullable=False,
            comment="Take profit price",
        ),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="open",
            comment="open or closed",
        ),
        sa.Column(
            "exit_date",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "exit_price",
            sa.Numeric(12, 4),
            nullable=True,
        ),
        sa.Column(
            "exit_reason",
            sa.String(50),
            nullable=True,
            comment="stop_loss, take_profit, timeout, manual",
        ),
        sa.Column(
            "pnl",
            sa.Numeric(14, 2),
            nullable=True,
            comment="Profit/Loss in dollars",
        ),
        sa.Column(
            "pnl_pct",
            sa.Numeric(8, 4),
            nullable=True,
            comment="Profit/Loss percentage",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_paper_positions"),
    )

    # Create indexes for paper_positions
    op.create_index(
        "ix_paper_positions_signal_id",
        "paper_positions",
        ["signal_id"],
    )
    op.create_index(
        "ix_paper_positions_symbol",
        "paper_positions",
        ["symbol"],
    )
    op.create_index(
        "ix_paper_positions_status",
        "paper_positions",
        ["status"],
    )
    op.create_index(
        "ix_paper_positions_symbol_status",
        "paper_positions",
        ["symbol", "status"],
    )
    op.create_index(
        "ix_paper_positions_entry_date",
        "paper_positions",
        ["entry_date"],
    )

    # Create portfolio_snapshots table
    op.create_table(
        "portfolio_snapshots",
        sa.Column(
            "id",
            sa.Integer(),
            nullable=False,
            autoincrement=True,
        ),
        sa.Column(
            "snapshot_date",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="Date of snapshot (EOD)",
        ),
        sa.Column(
            "equity_value",
            sa.Numeric(14, 2),
            nullable=False,
            comment="Total portfolio value",
        ),
        sa.Column(
            "cash_balance",
            sa.Numeric(14, 2),
            nullable=False,
            comment="Cash not in positions",
        ),
        sa.Column(
            "positions_value",
            sa.Numeric(14, 2),
            nullable=False,
            comment="Value of open positions",
        ),
        sa.Column(
            "positions_count",
            sa.Integer(),
            nullable=False,
            comment="Number of open positions",
        ),
        sa.Column(
            "daily_return_pct",
            sa.Numeric(8, 4),
            nullable=True,
            comment="Daily return %",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_portfolio_snapshots"),
        sa.UniqueConstraint("snapshot_date", name="uq_portfolio_snapshots_snapshot_date"),
    )

    # Create indexes for portfolio_snapshots
    op.create_index(
        "ix_portfolio_snapshots_snapshot_date",
        "portfolio_snapshots",
        ["snapshot_date"],
    )


def downgrade() -> None:
    """Drop paper trading tables and indexes."""
    # Drop portfolio_snapshots table (indexes will be dropped automatically)
    op.drop_table("portfolio_snapshots")

    # Drop paper_positions table (indexes will be dropped automatically)
    op.drop_table("paper_positions")

"""Comprehensive tests for paper portfolio engine."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from signalforge.benchmark.paper_portfolio import (
    PaperPortfolio,
    PortfolioConfig,
    PositionState,
)


class TestPortfolioConfig:
    """Tests for PortfolioConfig dataclass."""

    def test_valid_config(self) -> None:
        """Test creating valid portfolio configuration."""
        config = PortfolioConfig(
            initial_capital=Decimal("100000"),
            max_positions=10,
            max_position_size_pct=Decimal("20"),
            stop_loss_pct=Decimal("5"),
            take_profit_pct=Decimal("10"),
        )

        assert config.initial_capital == Decimal("100000")
        assert config.max_positions == 10
        assert config.max_position_size_pct == Decimal("20")
        assert config.stop_loss_pct == Decimal("5")
        assert config.take_profit_pct == Decimal("10")

    def test_config_without_optional_params(self) -> None:
        """Test config creation without optional stop-loss/take-profit."""
        config = PortfolioConfig(
            initial_capital=Decimal("50000"),
            max_positions=5,
            max_position_size_pct=Decimal("15"),
        )

        assert config.initial_capital == Decimal("50000")
        assert config.max_positions == 5
        assert config.max_position_size_pct == Decimal("15")
        assert config.stop_loss_pct is None
        assert config.take_profit_pct is None

    def test_negative_initial_capital(self) -> None:
        """Test that negative initial capital raises error."""
        with pytest.raises(ValueError, match="initial_capital must be positive"):
            PortfolioConfig(
                initial_capital=Decimal("-1000"),
                max_positions=5,
                max_position_size_pct=Decimal("20"),
            )

    def test_zero_initial_capital(self) -> None:
        """Test that zero initial capital raises error."""
        with pytest.raises(ValueError, match="initial_capital must be positive"):
            PortfolioConfig(
                initial_capital=Decimal("0"),
                max_positions=5,
                max_position_size_pct=Decimal("20"),
            )

    def test_negative_max_positions(self) -> None:
        """Test that negative max positions raises error."""
        with pytest.raises(ValueError, match="max_positions must be positive"):
            PortfolioConfig(
                initial_capital=Decimal("100000"),
                max_positions=-1,
                max_position_size_pct=Decimal("20"),
            )

    def test_zero_max_positions(self) -> None:
        """Test that zero max positions raises error."""
        with pytest.raises(ValueError, match="max_positions must be positive"):
            PortfolioConfig(
                initial_capital=Decimal("100000"),
                max_positions=0,
                max_position_size_pct=Decimal("20"),
            )

    def test_invalid_max_position_size_pct_zero(self) -> None:
        """Test that zero max position size raises error."""
        with pytest.raises(
            ValueError, match="max_position_size_pct must be between 0 and 100"
        ):
            PortfolioConfig(
                initial_capital=Decimal("100000"),
                max_positions=5,
                max_position_size_pct=Decimal("0"),
            )

    def test_invalid_max_position_size_pct_over_100(self) -> None:
        """Test that position size over 100% raises error."""
        with pytest.raises(
            ValueError, match="max_position_size_pct must be between 0 and 100"
        ):
            PortfolioConfig(
                initial_capital=Decimal("100000"),
                max_positions=5,
                max_position_size_pct=Decimal("101"),
            )

    def test_invalid_stop_loss_pct_zero(self) -> None:
        """Test that zero stop loss percentage raises error."""
        with pytest.raises(
            ValueError, match="stop_loss_pct must be between 0 and 100"
        ):
            PortfolioConfig(
                initial_capital=Decimal("100000"),
                max_positions=5,
                max_position_size_pct=Decimal("20"),
                stop_loss_pct=Decimal("0"),
            )

    def test_invalid_stop_loss_pct_over_100(self) -> None:
        """Test that stop loss over 100% raises error."""
        with pytest.raises(
            ValueError, match="stop_loss_pct must be between 0 and 100"
        ):
            PortfolioConfig(
                initial_capital=Decimal("100000"),
                max_positions=5,
                max_position_size_pct=Decimal("20"),
                stop_loss_pct=Decimal("100"),
            )

    def test_invalid_take_profit_pct(self) -> None:
        """Test that zero or negative take profit raises error."""
        with pytest.raises(ValueError, match="take_profit_pct must be positive"):
            PortfolioConfig(
                initial_capital=Decimal("100000"),
                max_positions=5,
                max_position_size_pct=Decimal("20"),
                take_profit_pct=Decimal("-5"),
            )


class TestPositionState:
    """Tests for PositionState dataclass."""

    def test_valid_position_state(self) -> None:
        """Test creating valid position state."""
        position = PositionState(
            symbol="AAPL",
            quantity=100,
            entry_price=Decimal("150.00"),
            entry_date=datetime.now(UTC),
            current_price=Decimal("155.00"),
            stop_loss=Decimal("145.00"),
            take_profit=Decimal("160.00"),
        )

        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.entry_price == Decimal("150.00")
        assert position.current_price == Decimal("155.00")
        assert position.stop_loss == Decimal("145.00")
        assert position.take_profit == Decimal("160.00")
        assert position.unrealized_pnl == Decimal("500.0000")
        assert position.unrealized_pnl_pct == Decimal("3.3333")

    def test_position_without_stop_take_profit(self) -> None:
        """Test position creation without stop-loss/take-profit."""
        position = PositionState(
            symbol="TSLA",
            quantity=50,
            entry_price=Decimal("200.00"),
            entry_date=datetime.now(UTC),
            current_price=Decimal("200.00"),
            stop_loss=None,
            take_profit=None,
        )

        assert position.stop_loss is None
        assert position.take_profit is None
        assert position.unrealized_pnl == Decimal("0")
        assert position.unrealized_pnl_pct == Decimal("0")

    def test_empty_symbol(self) -> None:
        """Test that empty symbol raises error."""
        with pytest.raises(ValueError, match="symbol cannot be empty"):
            PositionState(
                symbol="",
                quantity=100,
                entry_price=Decimal("150.00"),
                entry_date=datetime.now(UTC),
                current_price=Decimal("150.00"),
                stop_loss=None,
                take_profit=None,
            )

    def test_negative_quantity(self) -> None:
        """Test that negative quantity raises error."""
        with pytest.raises(ValueError, match="quantity must be positive"):
            PositionState(
                symbol="AAPL",
                quantity=-100,
                entry_price=Decimal("150.00"),
                entry_date=datetime.now(UTC),
                current_price=Decimal("150.00"),
                stop_loss=None,
                take_profit=None,
            )

    def test_zero_quantity(self) -> None:
        """Test that zero quantity raises error."""
        with pytest.raises(ValueError, match="quantity must be positive"):
            PositionState(
                symbol="AAPL",
                quantity=0,
                entry_price=Decimal("150.00"),
                entry_date=datetime.now(UTC),
                current_price=Decimal("150.00"),
                stop_loss=None,
                take_profit=None,
            )

    def test_negative_entry_price(self) -> None:
        """Test that negative entry price raises error."""
        with pytest.raises(ValueError, match="entry_price must be positive"):
            PositionState(
                symbol="AAPL",
                quantity=100,
                entry_price=Decimal("-150.00"),
                entry_date=datetime.now(UTC),
                current_price=Decimal("150.00"),
                stop_loss=None,
                take_profit=None,
            )

    def test_zero_entry_price(self) -> None:
        """Test that zero entry price raises error."""
        with pytest.raises(ValueError, match="entry_price must be positive"):
            PositionState(
                symbol="AAPL",
                quantity=100,
                entry_price=Decimal("0"),
                entry_date=datetime.now(UTC),
                current_price=Decimal("150.00"),
                stop_loss=None,
                take_profit=None,
            )

    def test_negative_current_price(self) -> None:
        """Test that negative current price raises error."""
        with pytest.raises(ValueError, match="current_price cannot be negative"):
            PositionState(
                symbol="AAPL",
                quantity=100,
                entry_price=Decimal("150.00"),
                entry_date=datetime.now(UTC),
                current_price=Decimal("-150.00"),
                stop_loss=None,
                take_profit=None,
            )

    def test_update_price(self) -> None:
        """Test updating position price recalculates P&L."""
        position = PositionState(
            symbol="AAPL",
            quantity=100,
            entry_price=Decimal("150.00"),
            entry_date=datetime.now(UTC),
            current_price=Decimal("150.00"),
            stop_loss=None,
            take_profit=None,
        )

        assert position.unrealized_pnl == Decimal("0")

        position.update_price(Decimal("155.00"))

        assert position.current_price == Decimal("155.00")
        assert position.unrealized_pnl == Decimal("500.0000")
        assert position.unrealized_pnl_pct == Decimal("3.3333")

    def test_update_price_negative(self) -> None:
        """Test that updating to negative price raises error."""
        position = PositionState(
            symbol="AAPL",
            quantity=100,
            entry_price=Decimal("150.00"),
            entry_date=datetime.now(UTC),
            current_price=Decimal("150.00"),
            stop_loss=None,
            take_profit=None,
        )

        with pytest.raises(ValueError, match="new_price cannot be negative"):
            position.update_price(Decimal("-10.00"))

    def test_check_stop_loss_triggered(self) -> None:
        """Test stop-loss detection when triggered."""
        position = PositionState(
            symbol="AAPL",
            quantity=100,
            entry_price=Decimal("150.00"),
            entry_date=datetime.now(UTC),
            current_price=Decimal("144.00"),
            stop_loss=Decimal("145.00"),
            take_profit=None,
        )

        assert position.check_stop_loss() is True

    def test_check_stop_loss_not_triggered(self) -> None:
        """Test stop-loss detection when not triggered."""
        position = PositionState(
            symbol="AAPL",
            quantity=100,
            entry_price=Decimal("150.00"),
            entry_date=datetime.now(UTC),
            current_price=Decimal("148.00"),
            stop_loss=Decimal("145.00"),
            take_profit=None,
        )

        assert position.check_stop_loss() is False

    def test_check_stop_loss_none(self) -> None:
        """Test stop-loss detection when no stop-loss set."""
        position = PositionState(
            symbol="AAPL",
            quantity=100,
            entry_price=Decimal("150.00"),
            entry_date=datetime.now(UTC),
            current_price=Decimal("140.00"),
            stop_loss=None,
            take_profit=None,
        )

        assert position.check_stop_loss() is False

    def test_check_take_profit_triggered(self) -> None:
        """Test take-profit detection when triggered."""
        position = PositionState(
            symbol="AAPL",
            quantity=100,
            entry_price=Decimal("150.00"),
            entry_date=datetime.now(UTC),
            current_price=Decimal("161.00"),
            stop_loss=None,
            take_profit=Decimal("160.00"),
        )

        assert position.check_take_profit() is True

    def test_check_take_profit_not_triggered(self) -> None:
        """Test take-profit detection when not triggered."""
        position = PositionState(
            symbol="AAPL",
            quantity=100,
            entry_price=Decimal("150.00"),
            entry_date=datetime.now(UTC),
            current_price=Decimal("158.00"),
            stop_loss=None,
            take_profit=Decimal("160.00"),
        )

        assert position.check_take_profit() is False

    def test_check_take_profit_none(self) -> None:
        """Test take-profit detection when no take-profit set."""
        position = PositionState(
            symbol="AAPL",
            quantity=100,
            entry_price=Decimal("150.00"),
            entry_date=datetime.now(UTC),
            current_price=Decimal("170.00"),
            stop_loss=None,
            take_profit=None,
        )

        assert position.check_take_profit() is False

    def test_get_market_value(self) -> None:
        """Test calculating market value of position."""
        position = PositionState(
            symbol="AAPL",
            quantity=100,
            entry_price=Decimal("150.00"),
            entry_date=datetime.now(UTC),
            current_price=Decimal("155.50"),
            stop_loss=None,
            take_profit=None,
        )

        market_value = position.get_market_value()
        assert market_value == Decimal("15550.0000")


class TestPaperPortfolio:
    """Tests for PaperPortfolio class."""

    def test_portfolio_initialization(self) -> None:
        """Test portfolio initialization with config."""
        config = PortfolioConfig(
            initial_capital=Decimal("100000"),
            max_positions=5,
            max_position_size_pct=Decimal("20"),
        )
        portfolio = PaperPortfolio(config)

        assert portfolio.get_cash() == Decimal("100000")
        assert portfolio.get_equity() == Decimal("100000")
        assert len(portfolio.get_positions()) == 0

    def test_open_position_success(self) -> None:
        """Test successfully opening a position."""
        config = PortfolioConfig(
            initial_capital=Decimal("100000"),
            max_positions=5,
            max_position_size_pct=Decimal("20"),
        )
        portfolio = PaperPortfolio(config)

        position = portfolio.open_position(
            symbol="AAPL",
            quantity=100,
            price=Decimal("150.00"),
            stop_loss=Decimal("145.00"),
            take_profit=Decimal("160.00"),
        )

        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.entry_price == Decimal("150.00")
        assert portfolio.get_cash() == Decimal("85000.0000")
        assert len(portfolio.get_positions()) == 1

    def test_open_position_duplicate_symbol(self) -> None:
        """Test that opening duplicate position raises error."""
        config = PortfolioConfig(
            initial_capital=Decimal("100000"),
            max_positions=5,
            max_position_size_pct=Decimal("50"),
        )
        portfolio = PaperPortfolio(config)

        portfolio.open_position(
            symbol="AAPL",
            quantity=100,
            price=Decimal("150.00"),
        )

        with pytest.raises(ValueError, match="Position already exists for symbol: AAPL"):
            portfolio.open_position(
                symbol="AAPL",
                quantity=50,
                price=Decimal("155.00"),
            )

    def test_open_position_max_positions_exceeded(self) -> None:
        """Test that exceeding max positions raises error."""
        config = PortfolioConfig(
            initial_capital=Decimal("100000"),
            max_positions=2,
            max_position_size_pct=Decimal("30"),
        )
        portfolio = PaperPortfolio(config)

        portfolio.open_position("AAPL", 50, Decimal("100.00"))
        portfolio.open_position("TSLA", 50, Decimal("200.00"))

        with pytest.raises(ValueError, match="max positions"):
            portfolio.open_position("MSFT", 50, Decimal("300.00"))

    def test_open_position_insufficient_cash(self) -> None:
        """Test that insufficient cash raises error."""
        config = PortfolioConfig(
            initial_capital=Decimal("10000"),
            max_positions=5,
            max_position_size_pct=Decimal("100"),
        )
        portfolio = PaperPortfolio(config)

        with pytest.raises(ValueError, match="Insufficient cash"):
            portfolio.open_position("AAPL", 100, Decimal("150.00"))

    def test_open_position_exceeds_max_size(self) -> None:
        """Test that position exceeding max size raises error."""
        config = PortfolioConfig(
            initial_capital=Decimal("100000"),
            max_positions=5,
            max_position_size_pct=Decimal("10"),
        )
        portfolio = PaperPortfolio(config)

        with pytest.raises(ValueError, match="exceeds maximum"):
            portfolio.open_position("AAPL", 100, Decimal("150.00"))

    def test_open_position_empty_symbol(self) -> None:
        """Test that empty symbol raises error."""
        config = PortfolioConfig(
            initial_capital=Decimal("100000"),
            max_positions=5,
            max_position_size_pct=Decimal("20"),
        )
        portfolio = PaperPortfolio(config)

        with pytest.raises(ValueError, match="symbol cannot be empty"):
            portfolio.open_position("", 100, Decimal("150.00"))

    def test_close_position_success(self) -> None:
        """Test successfully closing a position."""
        config = PortfolioConfig(
            initial_capital=Decimal("100000"),
            max_positions=5,
            max_position_size_pct=Decimal("20"),
        )
        portfolio = PaperPortfolio(config)

        portfolio.open_position("AAPL", 100, Decimal("150.00"))
        initial_cash = portfolio.get_cash()

        pnl = portfolio.close_position("AAPL", Decimal("155.00"))

        assert pnl == Decimal("500.0000")
        assert portfolio.get_cash() == initial_cash + Decimal("15500.0000")
        assert len(portfolio.get_positions()) == 0
        assert not portfolio.has_position("AAPL")

    def test_close_position_loss(self) -> None:
        """Test closing position at a loss."""
        config = PortfolioConfig(
            initial_capital=Decimal("100000"),
            max_positions=5,
            max_position_size_pct=Decimal("20"),
        )
        portfolio = PaperPortfolio(config)

        portfolio.open_position("AAPL", 100, Decimal("150.00"))
        pnl = portfolio.close_position("AAPL", Decimal("145.00"))

        assert pnl == Decimal("-500.0000")

    def test_close_position_nonexistent(self) -> None:
        """Test that closing nonexistent position raises error."""
        config = PortfolioConfig(
            initial_capital=Decimal("100000"),
            max_positions=5,
            max_position_size_pct=Decimal("20"),
        )
        portfolio = PaperPortfolio(config)

        with pytest.raises(ValueError, match="No open position found"):
            portfolio.close_position("AAPL", Decimal("150.00"))

    def test_update_prices(self) -> None:
        """Test updating prices for multiple positions."""
        config = PortfolioConfig(
            initial_capital=Decimal("100000"),
            max_positions=5,
            max_position_size_pct=Decimal("30"),
        )
        portfolio = PaperPortfolio(config)

        portfolio.open_position("AAPL", 100, Decimal("150.00"))
        portfolio.open_position("TSLA", 50, Decimal("200.00"))

        portfolio.update_prices({
            "AAPL": Decimal("155.00"),
            "TSLA": Decimal("210.00"),
        })

        aapl_position = portfolio.get_position("AAPL")
        tsla_position = portfolio.get_position("TSLA")

        assert aapl_position is not None
        assert aapl_position.current_price == Decimal("155.00")
        assert aapl_position.unrealized_pnl == Decimal("500.0000")

        assert tsla_position is not None
        assert tsla_position.current_price == Decimal("210.00")
        assert tsla_position.unrealized_pnl == Decimal("500.0000")

    def test_check_stop_loss_take_profit_stop_loss_triggered(self) -> None:
        """Test automatic stop-loss execution."""
        config = PortfolioConfig(
            initial_capital=Decimal("100000"),
            max_positions=5,
            max_position_size_pct=Decimal("20"),
        )
        portfolio = PaperPortfolio(config)

        portfolio.open_position(
            "AAPL",
            100,
            Decimal("150.00"),
            stop_loss=Decimal("145.00"),
        )

        portfolio.update_prices({"AAPL": Decimal("144.00")})
        closed = portfolio.check_stop_loss_take_profit()

        assert "AAPL" in closed
        assert len(portfolio.get_positions()) == 0

    def test_check_stop_loss_take_profit_take_profit_triggered(self) -> None:
        """Test automatic take-profit execution."""
        config = PortfolioConfig(
            initial_capital=Decimal("100000"),
            max_positions=5,
            max_position_size_pct=Decimal("20"),
        )
        portfolio = PaperPortfolio(config)

        portfolio.open_position(
            "AAPL",
            100,
            Decimal("150.00"),
            take_profit=Decimal("160.00"),
        )

        portfolio.update_prices({"AAPL": Decimal("161.00")})
        closed = portfolio.check_stop_loss_take_profit()

        assert "AAPL" in closed
        assert len(portfolio.get_positions()) == 0

    def test_check_stop_loss_take_profit_not_triggered(self) -> None:
        """Test that no positions close when thresholds not reached."""
        config = PortfolioConfig(
            initial_capital=Decimal("100000"),
            max_positions=5,
            max_position_size_pct=Decimal("20"),
        )
        portfolio = PaperPortfolio(config)

        portfolio.open_position(
            "AAPL",
            100,
            Decimal("150.00"),
            stop_loss=Decimal("145.00"),
            take_profit=Decimal("160.00"),
        )

        portfolio.update_prices({"AAPL": Decimal("152.00")})
        closed = portfolio.check_stop_loss_take_profit()

        assert len(closed) == 0
        assert len(portfolio.get_positions()) == 1

    def test_get_equity(self) -> None:
        """Test calculating total portfolio equity."""
        config = PortfolioConfig(
            initial_capital=Decimal("100000"),
            max_positions=5,
            max_position_size_pct=Decimal("30"),
        )
        portfolio = PaperPortfolio(config)

        portfolio.open_position("AAPL", 100, Decimal("150.00"))
        portfolio.update_prices({"AAPL": Decimal("155.00")})

        equity = portfolio.get_equity()
        cash = portfolio.get_cash()
        positions_value = portfolio.get_positions_value()

        assert equity == cash + positions_value
        assert equity == Decimal("100500.0000")

    def test_get_positions_value(self) -> None:
        """Test calculating total value of positions."""
        config = PortfolioConfig(
            initial_capital=Decimal("100000"),
            max_positions=5,
            max_position_size_pct=Decimal("30"),
        )
        portfolio = PaperPortfolio(config)

        portfolio.open_position("AAPL", 100, Decimal("150.00"))
        portfolio.open_position("TSLA", 50, Decimal("200.00"))

        portfolio.update_prices({
            "AAPL": Decimal("155.00"),
            "TSLA": Decimal("210.00"),
        })

        positions_value = portfolio.get_positions_value()
        assert positions_value == Decimal("26000.0000")

    def test_get_total_pnl(self) -> None:
        """Test calculating total P&L (realized + unrealized)."""
        config = PortfolioConfig(
            initial_capital=Decimal("100000"),
            max_positions=5,
            max_position_size_pct=Decimal("30"),
        )
        portfolio = PaperPortfolio(config)

        portfolio.open_position("AAPL", 100, Decimal("150.00"))
        portfolio.open_position("TSLA", 50, Decimal("200.00"))

        portfolio.update_prices({
            "AAPL": Decimal("155.00"),
            "TSLA": Decimal("210.00"),
        })

        portfolio.close_position("TSLA", Decimal("210.00"))

        total_pnl = portfolio.get_total_pnl()
        assert total_pnl == Decimal("1000.0000")

    def test_has_position(self) -> None:
        """Test checking if position exists."""
        config = PortfolioConfig(
            initial_capital=Decimal("100000"),
            max_positions=5,
            max_position_size_pct=Decimal("20"),
        )
        portfolio = PaperPortfolio(config)

        assert not portfolio.has_position("AAPL")

        portfolio.open_position("AAPL", 100, Decimal("150.00"))

        assert portfolio.has_position("AAPL")
        assert not portfolio.has_position("TSLA")

    def test_to_snapshot(self) -> None:
        """Test creating portfolio snapshot."""
        config = PortfolioConfig(
            initial_capital=Decimal("100000"),
            max_positions=5,
            max_position_size_pct=Decimal("20"),
        )
        portfolio = PaperPortfolio(config)

        portfolio.open_position("AAPL", 100, Decimal("150.00"))
        portfolio.update_prices({"AAPL": Decimal("155.00")})

        snapshot = portfolio.to_snapshot()

        assert snapshot.equity_value == Decimal("100500.00")
        assert snapshot.cash_balance == Decimal("85000.00")
        assert snapshot.positions_value == Decimal("15500.00")
        assert snapshot.positions_count == 1
        assert snapshot.daily_return_pct is None

    def test_multiple_positions_workflow(self) -> None:
        """Test complex workflow with multiple positions."""
        config = PortfolioConfig(
            initial_capital=Decimal("100000"),
            max_positions=5,
            max_position_size_pct=Decimal("25"),
        )
        portfolio = PaperPortfolio(config)

        portfolio.open_position("AAPL", 100, Decimal("150.00"))
        portfolio.open_position("TSLA", 50, Decimal("200.00"))
        portfolio.open_position("MSFT", 75, Decimal("300.00"))

        assert len(portfolio.get_positions()) == 3
        assert portfolio.get_cash() == Decimal("52500.0000")

        portfolio.update_prices({
            "AAPL": Decimal("155.00"),
            "TSLA": Decimal("210.00"),
            "MSFT": Decimal("305.00"),
        })

        portfolio.close_position("TSLA", Decimal("210.00"))

        assert len(portfolio.get_positions()) == 2
        assert portfolio.has_position("AAPL")
        assert not portfolio.has_position("TSLA")
        assert portfolio.has_position("MSFT")

    def test_edge_case_zero_unrealized_pnl(self) -> None:
        """Test position with zero unrealized P&L."""
        config = PortfolioConfig(
            initial_capital=Decimal("100000"),
            max_positions=5,
            max_position_size_pct=Decimal("20"),
        )
        portfolio = PaperPortfolio(config)

        portfolio.open_position("AAPL", 100, Decimal("150.00"))

        position = portfolio.get_position("AAPL")
        assert position is not None
        assert position.unrealized_pnl == Decimal("0")
        assert position.unrealized_pnl_pct == Decimal("0")

    def test_position_size_at_exact_max(self) -> None:
        """Test opening position at exactly max size."""
        config = PortfolioConfig(
            initial_capital=Decimal("100000"),
            max_positions=5,
            max_position_size_pct=Decimal("20"),
        )
        portfolio = PaperPortfolio(config)

        portfolio.open_position("AAPL", 133, Decimal("150.00"))

        assert len(portfolio.get_positions()) == 1
        position = portfolio.get_position("AAPL")
        assert position is not None

    def test_sequential_open_close_reopen(self) -> None:
        """Test opening, closing, and reopening same symbol."""
        config = PortfolioConfig(
            initial_capital=Decimal("100000"),
            max_positions=5,
            max_position_size_pct=Decimal("20"),
        )
        portfolio = PaperPortfolio(config)

        portfolio.open_position("AAPL", 100, Decimal("150.00"))
        portfolio.close_position("AAPL", Decimal("155.00"))

        assert not portfolio.has_position("AAPL")

        portfolio.open_position("AAPL", 100, Decimal("155.00"))

        assert portfolio.has_position("AAPL")
        position = portfolio.get_position("AAPL")
        assert position is not None
        assert position.entry_price == Decimal("155.00")

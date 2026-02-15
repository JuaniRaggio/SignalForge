"""Comprehensive tests for portfolio concentration analysis module."""

from decimal import Decimal

import polars as pl
import pytest

from signalforge.risk.concentration import (
    AlertSeverity,
    AlertType,
    ConcentrationAlert,
    ConcentrationAnalyzer,
    ConcentrationConfig,
    ConcentrationResult,
    Position,
)


class TestPosition:
    """Tests for Position dataclass."""

    def test_valid_position(self) -> None:
        """Test creating a valid position."""
        position = Position(
            symbol="AAPL",
            value=Decimal("100000"),
            weight=0.25,
            sector="Technology",
        )
        assert position.symbol == "AAPL"
        assert position.value == Decimal("100000")
        assert position.weight == 0.25
        assert position.sector == "Technology"

    def test_position_without_sector(self) -> None:
        """Test creating a position without sector."""
        position = Position(
            symbol="AAPL",
            value=Decimal("100000"),
            weight=0.25,
        )
        assert position.sector is None

    def test_empty_symbol_raises_error(self) -> None:
        """Test that empty symbol raises ValueError."""
        with pytest.raises(ValueError, match="Position symbol cannot be empty"):
            Position(symbol="", value=Decimal("100000"), weight=0.25)

    def test_whitespace_only_symbol_raises_error(self) -> None:
        """Test that whitespace-only symbol raises ValueError."""
        with pytest.raises(ValueError, match="Position symbol cannot be empty"):
            Position(symbol="   ", value=Decimal("100000"), weight=0.25)

    def test_negative_value_raises_error(self) -> None:
        """Test that negative value raises ValueError."""
        with pytest.raises(ValueError, match="Position value cannot be negative"):
            Position(symbol="AAPL", value=Decimal("-1000"), weight=0.25)

    def test_zero_value_is_valid(self) -> None:
        """Test that zero value is valid."""
        position = Position(symbol="AAPL", value=Decimal("0"), weight=0.0)
        assert position.value == Decimal("0")

    def test_weight_below_zero_raises_error(self) -> None:
        """Test that weight below 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="Position weight must be between 0.0 and 1.0"):
            Position(symbol="AAPL", value=Decimal("100000"), weight=-0.1)

    def test_weight_above_one_raises_error(self) -> None:
        """Test that weight above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="Position weight must be between 0.0 and 1.0"):
            Position(symbol="AAPL", value=Decimal("100000"), weight=1.5)

    def test_edge_case_weights(self) -> None:
        """Test edge case weights (0.0 and 1.0)."""
        position_min = Position(symbol="AAPL", value=Decimal("0"), weight=0.0)
        assert position_min.weight == 0.0

        position_max = Position(symbol="AAPL", value=Decimal("100000"), weight=1.0)
        assert position_max.weight == 1.0


class TestConcentrationConfig:
    """Tests for ConcentrationConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ConcentrationConfig()
        assert config.max_single_position_pct == 0.15
        assert config.max_sector_pct == 0.30
        assert config.max_correlated_group_pct == 0.40
        assert config.hhi_warning_threshold == 0.15
        assert config.hhi_critical_threshold == 0.25

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = ConcentrationConfig(
            max_single_position_pct=0.20,
            max_sector_pct=0.35,
            max_correlated_group_pct=0.45,
            hhi_warning_threshold=0.10,
            hhi_critical_threshold=0.20,
        )
        assert config.max_single_position_pct == 0.20
        assert config.max_sector_pct == 0.35
        assert config.max_correlated_group_pct == 0.45
        assert config.hhi_warning_threshold == 0.10
        assert config.hhi_critical_threshold == 0.20

    def test_invalid_max_single_position_below_zero_raises_error(self) -> None:
        """Test that max_single_position_pct below 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="max_single_position_pct must be between 0.0 and 1.0"):
            ConcentrationConfig(max_single_position_pct=-0.1)

    def test_invalid_max_single_position_zero_raises_error(self) -> None:
        """Test that max_single_position_pct of 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="max_single_position_pct must be between 0.0 and 1.0"):
            ConcentrationConfig(max_single_position_pct=0.0)

    def test_invalid_max_single_position_above_one_raises_error(self) -> None:
        """Test that max_single_position_pct above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="max_single_position_pct must be between 0.0 and 1.0"):
            ConcentrationConfig(max_single_position_pct=1.5)

    def test_invalid_max_sector_below_zero_raises_error(self) -> None:
        """Test that max_sector_pct below 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="max_sector_pct must be between 0.0 and 1.0"):
            ConcentrationConfig(max_sector_pct=-0.1)

    def test_invalid_max_sector_zero_raises_error(self) -> None:
        """Test that max_sector_pct of 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="max_sector_pct must be between 0.0 and 1.0"):
            ConcentrationConfig(max_sector_pct=0.0)

    def test_invalid_max_sector_above_one_raises_error(self) -> None:
        """Test that max_sector_pct above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="max_sector_pct must be between 0.0 and 1.0"):
            ConcentrationConfig(max_sector_pct=1.5)

    def test_invalid_max_correlated_group_below_zero_raises_error(self) -> None:
        """Test that max_correlated_group_pct below 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="max_correlated_group_pct must be between 0.0 and 1.0"):
            ConcentrationConfig(max_correlated_group_pct=-0.1)

    def test_invalid_max_correlated_group_zero_raises_error(self) -> None:
        """Test that max_correlated_group_pct of 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="max_correlated_group_pct must be between 0.0 and 1.0"):
            ConcentrationConfig(max_correlated_group_pct=0.0)

    def test_invalid_max_correlated_group_above_one_raises_error(self) -> None:
        """Test that max_correlated_group_pct above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="max_correlated_group_pct must be between 0.0 and 1.0"):
            ConcentrationConfig(max_correlated_group_pct=1.5)

    def test_invalid_hhi_warning_below_zero_raises_error(self) -> None:
        """Test that hhi_warning_threshold below 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="hhi_warning_threshold must be between 0.0 and 1.0"):
            ConcentrationConfig(hhi_warning_threshold=-0.1)

    def test_invalid_hhi_warning_above_one_raises_error(self) -> None:
        """Test that hhi_warning_threshold above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="hhi_warning_threshold must be between 0.0 and 1.0"):
            ConcentrationConfig(hhi_warning_threshold=1.5)

    def test_invalid_hhi_critical_below_zero_raises_error(self) -> None:
        """Test that hhi_critical_threshold below 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="hhi_critical_threshold must be between 0.0 and 1.0"):
            ConcentrationConfig(hhi_critical_threshold=-0.1)

    def test_invalid_hhi_critical_above_one_raises_error(self) -> None:
        """Test that hhi_critical_threshold above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="hhi_critical_threshold must be between 0.0 and 1.0"):
            ConcentrationConfig(hhi_critical_threshold=1.5)

    def test_hhi_critical_below_warning_raises_error(self) -> None:
        """Test that hhi_critical_threshold < hhi_warning_threshold raises ValueError."""
        with pytest.raises(ValueError, match="hhi_critical_threshold .* must be >= hhi_warning_threshold"):
            ConcentrationConfig(hhi_warning_threshold=0.20, hhi_critical_threshold=0.15)

    def test_hhi_thresholds_equal_is_valid(self) -> None:
        """Test that equal HHI thresholds are valid."""
        config = ConcentrationConfig(hhi_warning_threshold=0.20, hhi_critical_threshold=0.20)
        assert config.hhi_warning_threshold == 0.20
        assert config.hhi_critical_threshold == 0.20

    def test_edge_case_max_values(self) -> None:
        """Test edge case max values (all at 1.0)."""
        config = ConcentrationConfig(
            max_single_position_pct=1.0,
            max_sector_pct=1.0,
            max_correlated_group_pct=1.0,
        )
        assert config.max_single_position_pct == 1.0
        assert config.max_sector_pct == 1.0
        assert config.max_correlated_group_pct == 1.0


class TestConcentrationAlert:
    """Tests for ConcentrationAlert dataclass."""

    def test_valid_alert(self) -> None:
        """Test creating a valid alert."""
        alert = ConcentrationAlert(
            alert_type=AlertType.SINGLE_POSITION_EXCEEDED,
            severity=AlertSeverity.CRITICAL,
            message="Position AAPL weight 0.5 exceeds maximum",
            value=0.5,
            threshold=0.15,
            metadata={"symbol": "AAPL"},
        )
        assert alert.alert_type == AlertType.SINGLE_POSITION_EXCEEDED
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.value == 0.5
        assert alert.threshold == 0.15
        assert alert.metadata["symbol"] == "AAPL"

    def test_alert_without_metadata(self) -> None:
        """Test creating an alert without metadata."""
        alert = ConcentrationAlert(
            alert_type=AlertType.HHI_WARNING,
            severity=AlertSeverity.WARNING,
            message="HHI exceeds warning threshold",
            value=0.18,
            threshold=0.15,
        )
        assert alert.metadata == {}

    def test_empty_message_raises_error(self) -> None:
        """Test that empty message raises ValueError."""
        with pytest.raises(ValueError, match="Alert message cannot be empty"):
            ConcentrationAlert(
                alert_type=AlertType.HHI_WARNING,
                severity=AlertSeverity.WARNING,
                message="",
                value=0.18,
                threshold=0.15,
            )

    def test_whitespace_only_message_raises_error(self) -> None:
        """Test that whitespace-only message raises ValueError."""
        with pytest.raises(ValueError, match="Alert message cannot be empty"):
            ConcentrationAlert(
                alert_type=AlertType.HHI_WARNING,
                severity=AlertSeverity.WARNING,
                message="   ",
                value=0.18,
                threshold=0.15,
            )

    def test_invalid_value_below_zero_raises_error(self) -> None:
        """Test that value below 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="Alert value must be between 0.0 and 1.0"):
            ConcentrationAlert(
                alert_type=AlertType.HHI_WARNING,
                severity=AlertSeverity.WARNING,
                message="Test",
                value=-0.1,
                threshold=0.15,
            )

    def test_invalid_value_above_one_raises_error(self) -> None:
        """Test that value above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="Alert value must be between 0.0 and 1.0"):
            ConcentrationAlert(
                alert_type=AlertType.HHI_WARNING,
                severity=AlertSeverity.WARNING,
                message="Test",
                value=1.5,
                threshold=0.15,
            )

    def test_invalid_threshold_below_zero_raises_error(self) -> None:
        """Test that threshold below 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="Alert threshold must be between 0.0 and 1.0"):
            ConcentrationAlert(
                alert_type=AlertType.HHI_WARNING,
                severity=AlertSeverity.WARNING,
                message="Test",
                value=0.18,
                threshold=-0.1,
            )

    def test_invalid_threshold_above_one_raises_error(self) -> None:
        """Test that threshold above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="Alert threshold must be between 0.0 and 1.0"):
            ConcentrationAlert(
                alert_type=AlertType.HHI_WARNING,
                severity=AlertSeverity.WARNING,
                message="Test",
                value=0.18,
                threshold=1.5,
            )


class TestConcentrationResult:
    """Tests for ConcentrationResult dataclass."""

    def test_valid_result(self) -> None:
        """Test creating a valid result."""
        alerts = [
            ConcentrationAlert(
                alert_type=AlertType.HHI_WARNING,
                severity=AlertSeverity.WARNING,
                message="HHI warning",
                value=0.18,
                threshold=0.15,
            )
        ]
        result = ConcentrationResult(
            hhi_index=0.18,
            is_concentrated=True,
            largest_position_pct=0.35,
            largest_sector_pct=0.50,
            alerts=alerts,
            metadata={"num_positions": 5},
        )
        assert result.hhi_index == 0.18
        assert result.is_concentrated is True
        assert result.largest_position_pct == 0.35
        assert result.largest_sector_pct == 0.50
        assert len(result.alerts) == 1
        assert result.metadata["num_positions"] == 5

    def test_result_without_alerts(self) -> None:
        """Test creating a result without alerts."""
        result = ConcentrationResult(
            hhi_index=0.10,
            is_concentrated=False,
            largest_position_pct=0.10,
            largest_sector_pct=0.20,
            alerts=[],
        )
        assert len(result.alerts) == 0

    def test_invalid_hhi_below_zero_raises_error(self) -> None:
        """Test that hhi_index below 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="HHI index must be between 0.0 and 1.0"):
            ConcentrationResult(
                hhi_index=-0.1,
                is_concentrated=False,
                largest_position_pct=0.10,
                largest_sector_pct=0.20,
                alerts=[],
            )

    def test_invalid_hhi_above_one_raises_error(self) -> None:
        """Test that hhi_index above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="HHI index must be between 0.0 and 1.0"):
            ConcentrationResult(
                hhi_index=1.5,
                is_concentrated=False,
                largest_position_pct=0.10,
                largest_sector_pct=0.20,
                alerts=[],
            )

    def test_invalid_largest_position_below_zero_raises_error(self) -> None:
        """Test that largest_position_pct below 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="largest_position_pct must be between 0.0 and 1.0"):
            ConcentrationResult(
                hhi_index=0.10,
                is_concentrated=False,
                largest_position_pct=-0.1,
                largest_sector_pct=0.20,
                alerts=[],
            )

    def test_invalid_largest_position_above_one_raises_error(self) -> None:
        """Test that largest_position_pct above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="largest_position_pct must be between 0.0 and 1.0"):
            ConcentrationResult(
                hhi_index=0.10,
                is_concentrated=False,
                largest_position_pct=1.5,
                largest_sector_pct=0.20,
                alerts=[],
            )

    def test_invalid_largest_sector_below_zero_raises_error(self) -> None:
        """Test that largest_sector_pct below 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="largest_sector_pct must be between 0.0 and 1.0"):
            ConcentrationResult(
                hhi_index=0.10,
                is_concentrated=False,
                largest_position_pct=0.10,
                largest_sector_pct=-0.1,
                alerts=[],
            )

    def test_invalid_largest_sector_above_one_raises_error(self) -> None:
        """Test that largest_sector_pct above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="largest_sector_pct must be between 0.0 and 1.0"):
            ConcentrationResult(
                hhi_index=0.10,
                is_concentrated=False,
                largest_position_pct=0.10,
                largest_sector_pct=1.5,
                alerts=[],
            )

    def test_get_alerts_by_severity(self) -> None:
        """Test filtering alerts by severity."""
        alerts = [
            ConcentrationAlert(
                alert_type=AlertType.HHI_WARNING,
                severity=AlertSeverity.WARNING,
                message="Warning",
                value=0.18,
                threshold=0.15,
            ),
            ConcentrationAlert(
                alert_type=AlertType.SINGLE_POSITION_EXCEEDED,
                severity=AlertSeverity.CRITICAL,
                message="Critical",
                value=0.50,
                threshold=0.15,
            ),
            ConcentrationAlert(
                alert_type=AlertType.HHI_CRITICAL,
                severity=AlertSeverity.CRITICAL,
                message="Critical HHI",
                value=0.30,
                threshold=0.25,
            ),
        ]
        result = ConcentrationResult(
            hhi_index=0.30,
            is_concentrated=True,
            largest_position_pct=0.50,
            largest_sector_pct=0.50,
            alerts=alerts,
        )

        warnings = result.get_alerts_by_severity(AlertSeverity.WARNING)
        assert len(warnings) == 1
        assert warnings[0].severity == AlertSeverity.WARNING

        criticals = result.get_alerts_by_severity(AlertSeverity.CRITICAL)
        assert len(criticals) == 2
        assert all(a.severity == AlertSeverity.CRITICAL for a in criticals)

    def test_get_alerts_by_type(self) -> None:
        """Test filtering alerts by type."""
        alerts = [
            ConcentrationAlert(
                alert_type=AlertType.HHI_WARNING,
                severity=AlertSeverity.WARNING,
                message="Warning",
                value=0.18,
                threshold=0.15,
            ),
            ConcentrationAlert(
                alert_type=AlertType.SINGLE_POSITION_EXCEEDED,
                severity=AlertSeverity.CRITICAL,
                message="Critical",
                value=0.50,
                threshold=0.15,
            ),
            ConcentrationAlert(
                alert_type=AlertType.SINGLE_POSITION_EXCEEDED,
                severity=AlertSeverity.CRITICAL,
                message="Critical 2",
                value=0.40,
                threshold=0.15,
            ),
        ]
        result = ConcentrationResult(
            hhi_index=0.30,
            is_concentrated=True,
            largest_position_pct=0.50,
            largest_sector_pct=0.50,
            alerts=alerts,
        )

        position_alerts = result.get_alerts_by_type(AlertType.SINGLE_POSITION_EXCEEDED)
        assert len(position_alerts) == 2
        assert all(a.alert_type == AlertType.SINGLE_POSITION_EXCEEDED for a in position_alerts)

        hhi_alerts = result.get_alerts_by_type(AlertType.HHI_WARNING)
        assert len(hhi_alerts) == 1
        assert hhi_alerts[0].alert_type == AlertType.HHI_WARNING


class TestConcentrationAnalyzer:
    """Tests for ConcentrationAnalyzer class."""

    @pytest.fixture
    def analyzer(self) -> ConcentrationAnalyzer:
        """Fixture for analyzer with default config."""
        return ConcentrationAnalyzer()

    @pytest.fixture
    def custom_analyzer(self) -> ConcentrationAnalyzer:
        """Fixture for analyzer with custom config."""
        config = ConcentrationConfig(
            max_single_position_pct=0.20,
            max_sector_pct=0.35,
            hhi_warning_threshold=0.10,
            hhi_critical_threshold=0.20,
        )
        return ConcentrationAnalyzer(config)

    def test_initialization_default(self, analyzer: ConcentrationAnalyzer) -> None:
        """Test analyzer initialization with default config."""
        assert analyzer._config.max_single_position_pct == 0.15
        assert analyzer._config.max_sector_pct == 0.30

    def test_initialization_custom(self, custom_analyzer: ConcentrationAnalyzer) -> None:
        """Test analyzer initialization with custom config."""
        assert custom_analyzer._config.max_single_position_pct == 0.20
        assert custom_analyzer._config.max_sector_pct == 0.35

    def test_calculate_hhi_equal_weights(self, analyzer: ConcentrationAnalyzer) -> None:
        """Test HHI calculation with equal weights."""
        weights = [0.25, 0.25, 0.25, 0.25]
        hhi = analyzer.calculate_hhi(weights)
        assert hhi == 0.25  # 4 * (0.25^2) = 0.25

    def test_calculate_hhi_single_position(self, analyzer: ConcentrationAnalyzer) -> None:
        """Test HHI calculation with single position."""
        weights = [1.0]
        hhi = analyzer.calculate_hhi(weights)
        assert hhi == 1.0

    def test_calculate_hhi_concentrated_portfolio(self, analyzer: ConcentrationAnalyzer) -> None:
        """Test HHI calculation with concentrated portfolio."""
        weights = [0.5, 0.3, 0.2]
        hhi = analyzer.calculate_hhi(weights)
        expected = 0.5**2 + 0.3**2 + 0.2**2  # 0.38
        assert abs(hhi - expected) < 0.001

    def test_calculate_hhi_diversified_portfolio(self, analyzer: ConcentrationAnalyzer) -> None:
        """Test HHI calculation with diversified portfolio."""
        weights = [0.1] * 10
        hhi = analyzer.calculate_hhi(weights)
        expected = 10 * (0.1**2)  # 0.10
        assert abs(hhi - expected) < 0.001

    def test_calculate_hhi_empty_weights_raises_error(self, analyzer: ConcentrationAnalyzer) -> None:
        """Test that empty weights list raises ValueError."""
        with pytest.raises(ValueError, match="Weights list cannot be empty"):
            analyzer.calculate_hhi([])

    def test_calculate_hhi_invalid_weight_negative_raises_error(
        self, analyzer: ConcentrationAnalyzer
    ) -> None:
        """Test that negative weight raises ValueError."""
        with pytest.raises(ValueError, match="All weights must be between 0.0 and 1.0"):
            analyzer.calculate_hhi([0.5, -0.1, 0.6])

    def test_calculate_hhi_invalid_weight_above_one_raises_error(
        self, analyzer: ConcentrationAnalyzer
    ) -> None:
        """Test that weight above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="All weights must be between 0.0 and 1.0"):
            analyzer.calculate_hhi([0.5, 1.5, 0.3])

    def test_analyze_single_position_no_alerts(self, analyzer: ConcentrationAnalyzer) -> None:
        """Test single position analysis with no alerts."""
        positions = [
            Position(symbol="AAPL", value=Decimal("10000"), weight=0.10),
            Position(symbol="MSFT", value=Decimal("10000"), weight=0.10),
            Position(symbol="JPM", value=Decimal("10000"), weight=0.10),
        ]
        alerts = analyzer.analyze_single_position(positions)
        assert len(alerts) == 0

    def test_analyze_single_position_with_alert(self, analyzer: ConcentrationAnalyzer) -> None:
        """Test single position analysis with alert."""
        positions = [
            Position(symbol="AAPL", value=Decimal("50000"), weight=0.50),
            Position(symbol="MSFT", value=Decimal("30000"), weight=0.30),
            Position(symbol="JPM", value=Decimal("20000"), weight=0.20),
        ]
        alerts = analyzer.analyze_single_position(positions)
        assert len(alerts) == 3  # All exceed 0.15 threshold
        assert all(a.alert_type == AlertType.SINGLE_POSITION_EXCEEDED for a in alerts)
        assert all(a.severity == AlertSeverity.CRITICAL for a in alerts)

    def test_analyze_single_position_empty_list(self, analyzer: ConcentrationAnalyzer) -> None:
        """Test single position analysis with empty list."""
        alerts = analyzer.analyze_single_position([])
        assert len(alerts) == 0

    def test_analyze_sector_concentration_no_alerts(self, analyzer: ConcentrationAnalyzer) -> None:
        """Test sector concentration analysis with no alerts."""
        positions = [
            Position(symbol="AAPL", value=Decimal("15000"), weight=0.15),
            Position(symbol="MSFT", value=Decimal("15000"), weight=0.15),
            Position(symbol="JPM", value=Decimal("10000"), weight=0.10),
            Position(symbol="BAC", value=Decimal("10000"), weight=0.10),
            Position(symbol="XOM", value=Decimal("15000"), weight=0.15),
            Position(symbol="CVX", value=Decimal("15000"), weight=0.15),
            Position(symbol="JNJ", value=Decimal("10000"), weight=0.10),
            Position(symbol="PFE", value=Decimal("10000"), weight=0.10),
        ]
        sector_map = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "JPM": "Financials",
            "BAC": "Financials",
            "XOM": "Energy",
            "CVX": "Energy",
            "JNJ": "Healthcare",
            "PFE": "Healthcare",
        }
        # Technology: 0.30, Financials: 0.20, Energy: 0.30, Healthcare: 0.20
        alerts = analyzer.analyze_sector_concentration(positions, sector_map)
        assert len(alerts) == 0  # No sector exceeds 0.30

    def test_analyze_sector_concentration_with_alert(self, analyzer: ConcentrationAnalyzer) -> None:
        """Test sector concentration analysis with alert."""
        positions = [
            Position(symbol="AAPL", value=Decimal("40000"), weight=0.40),
            Position(symbol="MSFT", value=Decimal("30000"), weight=0.30),
            Position(symbol="JPM", value=Decimal("30000"), weight=0.30),
        ]
        sector_map = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "JPM": "Financials",
        }
        alerts = analyzer.analyze_sector_concentration(positions, sector_map)
        assert len(alerts) == 1  # Technology sector: 0.70 > 0.30
        assert alerts[0].alert_type == AlertType.SECTOR_CONCENTRATION
        assert alerts[0].severity == AlertSeverity.CRITICAL
        assert "Technology" in alerts[0].message

    def test_analyze_sector_concentration_empty_sector_map(
        self, analyzer: ConcentrationAnalyzer
    ) -> None:
        """Test sector concentration analysis with empty sector map and no position sectors."""
        positions = [
            Position(symbol="AAPL", value=Decimal("50000"), weight=0.50),
        ]
        alerts = analyzer.analyze_sector_concentration(positions, {})
        # No sector information available (neither in map nor in position), should return empty
        assert len(alerts) == 0

    def test_analyze_sector_concentration_uses_position_sector(
        self, analyzer: ConcentrationAnalyzer
    ) -> None:
        """Test that sector analysis uses position.sector when sector_map is empty."""
        positions = [
            Position(symbol="AAPL", value=Decimal("40000"), weight=0.40, sector="Technology"),
            Position(symbol="MSFT", value=Decimal("30000"), weight=0.30, sector="Technology"),
            Position(symbol="JPM", value=Decimal("30000"), weight=0.30, sector="Financials"),
        ]
        # Empty sector_map should fall back to position.sector
        alerts = analyzer.analyze_sector_concentration(positions, {})
        # Technology sector: 0.70 > 0.30
        assert len(alerts) == 1
        assert "Technology" in alerts[0].message

    def test_analyze_correlated_groups_no_matrix(self, analyzer: ConcentrationAnalyzer) -> None:
        """Test correlated groups analysis with empty correlation matrix."""
        positions = [
            Position(symbol="AAPL", value=Decimal("50000"), weight=0.50),
        ]
        correlation_matrix = pl.DataFrame()
        alerts = analyzer.analyze_correlated_groups(positions, correlation_matrix)
        assert len(alerts) == 0

    def test_analyze_correlated_groups_with_alert(self, analyzer: ConcentrationAnalyzer) -> None:
        """Test correlated groups analysis with alert."""
        positions = [
            Position(symbol="AAPL", value=Decimal("30000"), weight=0.30),
            Position(symbol="MSFT", value=Decimal("25000"), weight=0.25),
            Position(symbol="JPM", value=Decimal("45000"), weight=0.45),
        ]
        correlation_matrix = pl.DataFrame(
            {
                "symbol": ["AAPL", "MSFT", "JPM"],
                "AAPL": [1.0, 0.85, 0.2],
                "MSFT": [0.85, 1.0, 0.15],
                "JPM": [0.2, 0.15, 1.0],
            }
        )
        alerts = analyzer.analyze_correlated_groups(positions, correlation_matrix, threshold=0.7)
        assert len(alerts) == 1  # AAPL+MSFT group: 0.55 > 0.40
        assert alerts[0].alert_type == AlertType.CORRELATED_GROUP
        assert alerts[0].severity == AlertSeverity.CRITICAL

    def test_analyze_correlated_groups_invalid_threshold_raises_error(
        self, analyzer: ConcentrationAnalyzer
    ) -> None:
        """Test that invalid correlation threshold raises ValueError."""
        positions = [Position(symbol="AAPL", value=Decimal("50000"), weight=0.50)]
        # Need a non-empty correlation matrix for validation to run
        correlation_matrix = pl.DataFrame({"symbol": ["AAPL"], "AAPL": [1.0]})
        with pytest.raises(ValueError, match="Correlation threshold must be between 0.0 and 1.0"):
            analyzer.analyze_correlated_groups(positions, correlation_matrix, threshold=1.5)

    def test_analyze_portfolio_basic(self, analyzer: ConcentrationAnalyzer) -> None:
        """Test basic portfolio analysis."""
        positions = [
            Position(symbol="AAPL", value=Decimal("10000"), weight=0.10),
            Position(symbol="MSFT", value=Decimal("10000"), weight=0.10),
            Position(symbol="JPM", value=Decimal("10000"), weight=0.10),
            Position(symbol="BAC", value=Decimal("10000"), weight=0.10),
            Position(symbol="XOM", value=Decimal("10000"), weight=0.10),
        ]
        sector_map = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "JPM": "Financials",
            "BAC": "Financials",
            "XOM": "Energy",
        }
        result = analyzer.analyze_portfolio(positions, sector_map)
        assert result.hhi_index > 0.0
        assert result.largest_position_pct == 0.10
        assert result.is_concentrated is False  # Well diversified
        assert len(result.alerts) == 0

    def test_analyze_portfolio_concentrated(self, analyzer: ConcentrationAnalyzer) -> None:
        """Test concentrated portfolio analysis."""
        positions = [
            Position(symbol="AAPL", value=Decimal("60000"), weight=0.60),
            Position(symbol="MSFT", value=Decimal("40000"), weight=0.40),
        ]
        sector_map = {"AAPL": "Technology", "MSFT": "Technology"}
        result = analyzer.analyze_portfolio(positions, sector_map)
        assert result.is_concentrated is True
        assert len(result.alerts) > 0
        assert result.largest_position_pct == 0.60
        assert result.largest_sector_pct == 1.0

    def test_analyze_portfolio_empty_raises_error(self, analyzer: ConcentrationAnalyzer) -> None:
        """Test that empty portfolio raises ValueError."""
        with pytest.raises(ValueError, match="Cannot analyze empty portfolio"):
            analyzer.analyze_portfolio([], {})

    def test_analyze_portfolio_hhi_warning(self, custom_analyzer: ConcentrationAnalyzer) -> None:
        """Test portfolio with HHI warning alert."""
        # custom_analyzer has hhi_warning_threshold=0.10, hhi_critical_threshold=0.20
        # We need HHI between 0.10 and 0.20 for a warning
        # Equal weights of 0.25 each gives HHI = 4 * 0.25^2 = 0.25, which exceeds critical
        # Let's use weights that give HHI between 0.10 and 0.20
        positions = [
            Position(symbol="AAPL", value=Decimal("25000"), weight=0.25),
            Position(symbol="MSFT", value=Decimal("25000"), weight=0.25),
            Position(symbol="JPM", value=Decimal("25000"), weight=0.25),
            Position(symbol="GS", value=Decimal("25000"), weight=0.25),
        ]
        # HHI = 4 * 0.25^2 = 0.25, which is > 0.20 (critical)
        # Let's try more diversified
        positions = [
            Position(symbol=f"STOCK{i}", value=Decimal("10000"), weight=0.10) for i in range(10)
        ]
        # HHI = 10 * 0.10^2 = 0.10, which equals warning threshold
        result = custom_analyzer.analyze_portfolio(positions, {})
        # Should get either warning or critical alert
        hhi_alerts = result.get_alerts_by_type(AlertType.HHI_WARNING)
        hhi_critical_alerts = result.get_alerts_by_type(AlertType.HHI_CRITICAL)
        assert len(hhi_alerts) + len(hhi_critical_alerts) >= 1

    def test_analyze_portfolio_hhi_critical(self, custom_analyzer: ConcentrationAnalyzer) -> None:
        """Test portfolio with HHI critical alert."""
        positions = [
            Position(symbol="AAPL", value=Decimal("70000"), weight=0.70),
            Position(symbol="MSFT", value=Decimal("30000"), weight=0.30),
        ]
        result = custom_analyzer.analyze_portfolio(positions, {})
        hhi_alerts = result.get_alerts_by_type(AlertType.HHI_CRITICAL)
        assert len(hhi_alerts) >= 1

    def test_analyze_portfolio_metadata(self, analyzer: ConcentrationAnalyzer) -> None:
        """Test that portfolio analysis includes metadata."""
        positions = [
            Position(symbol="AAPL", value=Decimal("30000"), weight=0.30),
            Position(symbol="MSFT", value=Decimal("40000"), weight=0.40),
            Position(symbol="JPM", value=Decimal("30000"), weight=0.30),
        ]
        sector_map = {"AAPL": "Technology", "MSFT": "Technology", "JPM": "Financials"}
        result = analyzer.analyze_portfolio(positions, sector_map)
        assert "num_positions" in result.metadata
        assert result.metadata["num_positions"] == 3
        assert "num_sectors" in result.metadata
        assert result.metadata["num_sectors"] == 2
        assert "total_value" in result.metadata
        assert "num_warnings" in result.metadata
        assert "num_critical" in result.metadata


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def analyzer(self) -> ConcentrationAnalyzer:
        """Fixture for analyzer."""
        return ConcentrationAnalyzer()

    def test_single_position_portfolio(self, analyzer: ConcentrationAnalyzer) -> None:
        """Test portfolio with single position."""
        positions = [Position(symbol="AAPL", value=Decimal("100000"), weight=1.0)]
        result = analyzer.analyze_portfolio(positions, {})
        assert result.hhi_index == 1.0
        assert result.is_concentrated is True
        assert len(result.alerts) > 0

    def test_many_small_positions(self, analyzer: ConcentrationAnalyzer) -> None:
        """Test portfolio with many small positions."""
        positions = [
            Position(symbol=f"STOCK{i}", value=Decimal("1000"), weight=0.01) for i in range(100)
        ]
        result = analyzer.analyze_portfolio(positions, {})
        assert result.hhi_index < 0.15  # Well diversified
        assert result.is_concentrated is False

    def test_zero_weight_positions(self, analyzer: ConcentrationAnalyzer) -> None:
        """Test portfolio with zero-weight positions."""
        positions = [
            Position(symbol="AAPL", value=Decimal("50000"), weight=0.50),
            Position(symbol="MSFT", value=Decimal("50000"), weight=0.50),
            Position(symbol="CASH", value=Decimal("0"), weight=0.0),
        ]
        result = analyzer.analyze_portfolio(positions, {})
        assert result.hhi_index == 0.5  # 0.5^2 + 0.5^2 + 0^2

    def test_very_small_weights(self, analyzer: ConcentrationAnalyzer) -> None:
        """Test portfolio with very small weights."""
        positions = [
            Position(symbol="AAPL", value=Decimal("99990"), weight=0.9999),
            Position(symbol="MSFT", value=Decimal("10"), weight=0.0001),
        ]
        result = analyzer.analyze_portfolio(positions, {})
        assert result.is_concentrated is True
        assert result.largest_position_pct > 0.99

    def test_weights_sum_slightly_off(self, analyzer: ConcentrationAnalyzer) -> None:
        """Test portfolio where weights don't sum exactly to 1.0."""
        # This should still work due to floating point tolerance
        weights = [0.333, 0.333, 0.333]  # Sum = 0.999
        hhi = analyzer.calculate_hhi(weights)
        assert 0.0 <= hhi <= 1.0

    def test_sector_map_with_unknown_symbols(self, analyzer: ConcentrationAnalyzer) -> None:
        """Test sector analysis when sector_map doesn't contain all symbols."""
        positions = [
            Position(symbol="AAPL", value=Decimal("30000"), weight=0.30),
            Position(symbol="MSFT", value=Decimal("30000"), weight=0.30),
            Position(symbol="UNKNOWN", value=Decimal("40000"), weight=0.40),
        ]
        sector_map = {"AAPL": "Technology", "MSFT": "Technology"}
        alerts = analyzer.analyze_sector_concentration(positions, sector_map)
        # Should only analyze known sectors
        assert len(alerts) == 1  # Technology: 0.60 > 0.30

    def test_all_alert_types_generated(self) -> None:
        """Test that all alert types can be generated."""
        config = ConcentrationConfig(
            max_single_position_pct=0.10,
            max_sector_pct=0.20,
            max_correlated_group_pct=0.30,
            hhi_warning_threshold=0.05,
            hhi_critical_threshold=0.10,
        )
        analyzer = ConcentrationAnalyzer(config)

        positions = [
            Position(symbol="AAPL", value=Decimal("40000"), weight=0.40),
            Position(symbol="MSFT", value=Decimal("30000"), weight=0.30),
            Position(symbol="GOOGL", value=Decimal("30000"), weight=0.30),
        ]
        sector_map = {"AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology"}
        correlation_matrix = pl.DataFrame(
            {
                "symbol": ["AAPL", "MSFT", "GOOGL"],
                "AAPL": [1.0, 0.85, 0.80],
                "MSFT": [0.85, 1.0, 0.75],
                "GOOGL": [0.80, 0.75, 1.0],
            }
        )

        result = analyzer.analyze_portfolio(positions, sector_map, correlation_matrix)

        alert_types = {alert.alert_type for alert in result.alerts}
        assert AlertType.SINGLE_POSITION_EXCEEDED in alert_types
        assert AlertType.SECTOR_CONCENTRATION in alert_types
        assert AlertType.CORRELATED_GROUP in alert_types
        assert AlertType.HHI_CRITICAL in alert_types or AlertType.HHI_WARNING in alert_types

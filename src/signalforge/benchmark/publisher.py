"""Benchmark report publishing and MLflow integration.

This module provides comprehensive report publishing capabilities for backtesting
and benchmark comparison results. It supports multiple output formats (JSON, CSV),
MLflow integration for experiment tracking, and flexible report generation.

The publisher handles:
- Performance metrics export
- Benchmark comparison results
- Equity curve data
- Trade history
- Full consolidated reports
- MLflow experiment tracking

Examples:
    Basic usage with JSON output:

    >>> from pathlib import Path
    >>> from signalforge.benchmark.publisher import BenchmarkPublisher, PublishConfig
    >>> from signalforge.ml.backtesting.metrics import BacktestMetrics
    >>>
    >>> config = PublishConfig(
    ...     output_dir=Path("./reports"),
    ...     format="json",
    ...     include_trades=True,
    ...     mlflow_tracking=False,
    ... )
    >>> publisher = BenchmarkPublisher(config)
    >>> metrics = BacktestMetrics(...)
    >>> report_path = publisher.publish_metrics(metrics, "my_strategy")

    Full report with MLflow tracking:

    >>> config = PublishConfig(
    ...     output_dir=Path("./reports"),
    ...     format="json",
    ...     include_trades=True,
    ...     include_equity_curve=True,
    ...     mlflow_tracking=True,
    ... )
    >>> publisher = BenchmarkPublisher(config)
    >>> report = publisher.publish_full_report(
    ...     metrics=metrics,
    ...     comparisons=comparisons,
    ...     equity_curve=equity_curve,
    ...     trades=trades,
    ...     strategy_name="momentum_strategy",
    ... )
    >>> print(f"Report ID: {report.report_id}")
    >>> print(f"MLflow Run: {report.mlflow_run_id}")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl

from signalforge.core.logging import get_logger

if TYPE_CHECKING:
    from signalforge.benchmark.comparator import ComparisonResult
    from signalforge.benchmark.performance import PerformanceMetrics
    from signalforge.ml.backtesting.engine import Trade
    from signalforge.ml.backtesting.metrics import BacktestMetrics

logger = get_logger(__name__)


@dataclass
class PublishConfig:
    """Configuration for benchmark report publishing.

    This dataclass encapsulates all configuration parameters for the publisher,
    including output paths, format preferences, and feature toggles.

    Attributes:
        output_dir: Directory path for saving report files
        format: Output format for structured data ("json" or "csv")
        include_trades: Whether to include trade history in reports
        include_equity_curve: Whether to include equity curve data
        mlflow_tracking: Whether to log results to MLflow

    Note:
        The output_dir will be created if it doesn't exist.
        CSV format is recommended for equity curves and trade data.
        JSON format is recommended for metrics and comparisons.
    """

    output_dir: Path
    format: Literal["json", "csv"] = "json"
    include_trades: bool = True
    include_equity_curve: bool = True
    mlflow_tracking: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not isinstance(self.output_dir, Path):
            raise TypeError(f"output_dir must be Path, got {type(self.output_dir)}")
        if self.format not in ("json", "csv"):
            raise ValueError(f"format must be 'json' or 'csv', got {self.format}")

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "PublishConfig initialized",
            output_dir=str(self.output_dir),
            format=self.format,
            mlflow_tracking=self.mlflow_tracking,
        )


@dataclass
class PublishedReport:
    """Metadata for a published benchmark report.

    This dataclass tracks all files and metadata associated with a published
    benchmark report, including MLflow run information.

    Attributes:
        report_id: Unique identifier for the report (timestamp-based)
        timestamp: When the report was generated
        file_paths: List of all generated file paths
        mlflow_run_id: MLflow run ID if tracking is enabled, None otherwise

    Note:
        The report_id uses ISO format timestamp for uniqueness and sorting.
    """

    report_id: str
    timestamp: datetime
    file_paths: list[Path] = field(default_factory=list)
    mlflow_run_id: str | None = None

    def __post_init__(self) -> None:
        """Validate published report after initialization."""
        if not self.report_id:
            raise ValueError("report_id cannot be empty")
        if not isinstance(self.timestamp, datetime):
            raise TypeError(f"timestamp must be datetime, got {type(self.timestamp)}")

    def to_dict(self) -> dict[str, str | list[str] | None]:
        """Convert published report to dictionary.

        Returns:
            Dictionary representation of the report metadata.
        """
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp.isoformat(),
            "file_paths": [str(p) for p in self.file_paths],
            "mlflow_run_id": self.mlflow_run_id,
        }


class BenchmarkPublisher:
    """Publisher for benchmark reports and MLflow integration.

    This class provides comprehensive report publishing capabilities for
    backtesting results, benchmark comparisons, and performance metrics.
    It supports multiple output formats and optional MLflow tracking.

    Attributes:
        config: Publishing configuration

    Examples:
        Create publisher and publish metrics:

        >>> from pathlib import Path
        >>> config = PublishConfig(output_dir=Path("./reports"))
        >>> publisher = BenchmarkPublisher(config)
        >>> metrics_path = publisher.publish_metrics(metrics, "my_strategy")

        Publish full report with MLflow:

        >>> config = PublishConfig(
        ...     output_dir=Path("./reports"),
        ...     mlflow_tracking=True,
        ... )
        >>> publisher = BenchmarkPublisher(config)
        >>> report = publisher.publish_full_report(
        ...     metrics=metrics,
        ...     comparisons=comparisons,
        ...     equity_curve=equity_curve,
        ...     trades=trades,
        ...     strategy_name="momentum",
        ... )
    """

    def __init__(self, config: PublishConfig) -> None:
        """Initialize benchmark publisher.

        Args:
            config: Publishing configuration
        """
        self.config = config
        logger.info(
            "BenchmarkPublisher initialized",
            output_dir=str(config.output_dir),
            format=config.format,
        )

    def _generate_report_id(self, strategy_name: str) -> str:
        """Generate unique report ID.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Unique report identifier
        """
        timestamp = datetime.utcnow()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        return f"{strategy_name}_{timestamp_str}"

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize strategy name for use in filenames.

        Args:
            name: Strategy name to sanitize

        Returns:
            Sanitized filename-safe string
        """
        # Replace spaces and special characters with underscores
        sanitized = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)
        return sanitized.lower()

    def _metrics_to_dict(
        self, metrics: BacktestMetrics | PerformanceMetrics
    ) -> dict[str, float | int | str]:
        """Convert metrics object to dictionary.

        Args:
            metrics: Either BacktestMetrics or PerformanceMetrics

        Returns:
            Dictionary with metric names and values
        """
        # Both BacktestMetrics and PerformanceMetrics have to_dict method or similar attributes
        from signalforge.benchmark.performance import PerformanceMetrics
        from signalforge.ml.backtesting.metrics import BacktestMetrics

        result: dict[str, float | int | str]
        if isinstance(metrics, BacktestMetrics):
            # BacktestMetrics.to_dict() returns dict[str, float | int]
            # We need to cast it to our return type
            return dict(metrics.to_dict())
        elif isinstance(metrics, PerformanceMetrics):
            # PerformanceMetrics doesn't have to_dict, so create one
            result = {
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "calmar_ratio": metrics.calmar_ratio,
                "max_drawdown": metrics.max_drawdown,
                "max_drawdown_duration": metrics.max_drawdown_duration,
                "cagr": metrics.cagr,
                "volatility": metrics.volatility,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "avg_win": metrics.avg_win,
                "avg_loss": metrics.avg_loss,
                "expectancy": metrics.expectancy,
                "total_trades": metrics.total_trades,
                "winning_trades": metrics.winning_trades,
                "losing_trades": metrics.losing_trades,
            }
            return result
        else:
            raise TypeError(f"Unsupported metrics type: {type(metrics)}")

    def publish_metrics(
        self, metrics: BacktestMetrics | PerformanceMetrics, strategy_name: str
    ) -> Path:
        """Publish performance metrics to file.

        Args:
            metrics: Performance metrics from backtest
            strategy_name: Name of the strategy

        Returns:
            Path to the published metrics file

        Examples:
            >>> metrics = BacktestMetrics(
            ...     total_return=15.5,
            ...     sharpe_ratio=1.8,
            ...     max_drawdown=8.2,
            ...     # ... other metrics
            ... )
            >>> path = publisher.publish_metrics(metrics, "momentum")
        """
        sanitized_name = self._sanitize_filename(strategy_name)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        if self.config.format == "json":
            filename = f"{sanitized_name}_metrics_{timestamp}.json"
            file_path = self.config.output_dir / filename

            metrics_dict = self._metrics_to_dict(metrics)
            metrics_dict["strategy_name"] = strategy_name
            metrics_dict["generated_at"] = datetime.utcnow().isoformat()

            with open(file_path, "w") as f:
                json.dump(metrics_dict, f, indent=2)

        else:  # csv
            filename = f"{sanitized_name}_metrics_{timestamp}.csv"
            file_path = self.config.output_dir / filename

            # Convert metrics to DataFrame for CSV export
            metrics_dict = self._metrics_to_dict(metrics)
            metrics_dict["strategy_name"] = strategy_name
            metrics_dict["generated_at"] = datetime.utcnow().isoformat()

            df = pl.DataFrame([metrics_dict])
            df.write_csv(file_path)

        logger.info(
            "Metrics published",
            strategy_name=strategy_name,
            file_path=str(file_path),
            format=self.config.format,
        )

        return file_path

    def publish_comparison(
        self, comparisons: list[ComparisonResult], strategy_name: str
    ) -> Path:
        """Publish benchmark comparison results to file.

        Args:
            comparisons: List of comparison results against benchmarks
            strategy_name: Name of the strategy

        Returns:
            Path to the published comparison file

        Examples:
            >>> comparisons = [
            ...     ComparisonResult(
            ...         strategy_name="momentum",
            ...         benchmark_name="SPY",
            ...         alpha=2.5,
            ...         beta=1.1,
            ...         # ... other metrics
            ...     ),
            ... ]
            >>> path = publisher.publish_comparison(comparisons, "momentum")
        """
        if not comparisons:
            logger.warning("No comparisons to publish", strategy_name=strategy_name)
            raise ValueError("comparisons list cannot be empty")

        sanitized_name = self._sanitize_filename(strategy_name)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        if self.config.format == "json":
            filename = f"{sanitized_name}_comparisons_{timestamp}.json"
            file_path = self.config.output_dir / filename

            comparisons_data = {
                "strategy_name": strategy_name,
                "generated_at": datetime.utcnow().isoformat(),
                "comparisons": [comp.to_dict() for comp in comparisons],
            }

            with open(file_path, "w") as f:
                json.dump(comparisons_data, f, indent=2)

        else:  # csv
            filename = f"{sanitized_name}_comparisons_{timestamp}.csv"
            file_path = self.config.output_dir / filename

            # Convert comparisons to DataFrame
            comparison_dicts = [comp.to_dict() for comp in comparisons]
            df = pl.DataFrame(comparison_dicts)
            df.write_csv(file_path)

        logger.info(
            "Comparisons published",
            strategy_name=strategy_name,
            num_comparisons=len(comparisons),
            file_path=str(file_path),
        )

        return file_path

    def publish_equity_curve(
        self,
        equity_curve: list[Decimal],
        timestamps: list[datetime],
        strategy_name: str,
    ) -> Path:
        """Publish equity curve data to file.

        Args:
            equity_curve: List of equity values over time
            timestamps: Corresponding timestamps for equity values
            strategy_name: Name of the strategy

        Returns:
            Path to the published equity curve file

        Raises:
            ValueError: If equity_curve and timestamps have different lengths

        Examples:
            >>> from decimal import Decimal
            >>> from datetime import datetime
            >>> equity = [Decimal("100000"), Decimal("105000"), Decimal("103000")]
            >>> times = [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)]
            >>> path = publisher.publish_equity_curve(equity, times, "momentum")
        """
        if len(equity_curve) != len(timestamps):
            raise ValueError(
                f"equity_curve length ({len(equity_curve)}) must match "
                f"timestamps length ({len(timestamps)})"
            )

        if not equity_curve:
            raise ValueError("equity_curve cannot be empty")

        sanitized_name = self._sanitize_filename(strategy_name)
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # CSV is the preferred format for time series data
        filename = f"{sanitized_name}_equity_curve_{timestamp_str}.csv"
        file_path = self.config.output_dir / filename

        # Create DataFrame with timestamps and equity values
        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "equity": [float(eq) for eq in equity_curve],
            }
        )
        df.write_csv(file_path)

        logger.info(
            "Equity curve published",
            strategy_name=strategy_name,
            data_points=len(equity_curve),
            file_path=str(file_path),
        )

        return file_path

    def publish_trades(
        self, trades: list[Trade], strategy_name: str
    ) -> Path:
        """Publish trade history to file.

        Args:
            trades: List of completed trades
            strategy_name: Name of the strategy

        Returns:
            Path to the published trades file

        Examples:
            >>> from signalforge.ml.backtesting.engine import Trade
            >>> trades = [
            ...     Trade(
            ...         entry_date=datetime(2024, 1, 1),
            ...         exit_date=datetime(2024, 1, 5),
            ...         entry_price=100.0,
            ...         exit_price=105.0,
            ...         position_size=100.0,
            ...         direction="long",
            ...         pnl=500.0,
            ...         return_pct=5.0,
            ...     ),
            ... ]
            >>> path = publisher.publish_trades(trades, "momentum")
        """
        if not trades:
            logger.warning("No trades to publish", strategy_name=strategy_name)
            raise ValueError("trades list cannot be empty")

        sanitized_name = self._sanitize_filename(strategy_name)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # CSV is the preferred format for trade data
        filename = f"{sanitized_name}_trades_{timestamp}.csv"
        file_path = self.config.output_dir / filename

        # Convert trades to DataFrame
        trade_dicts = [
            {
                "entry_date": trade.entry_date.isoformat(),
                "exit_date": trade.exit_date.isoformat(),
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "position_size": trade.position_size,
                "direction": trade.direction,
                "pnl": trade.pnl,
                "return_pct": trade.return_pct,
            }
            for trade in trades
        ]
        df = pl.DataFrame(trade_dicts)
        df.write_csv(file_path)

        logger.info(
            "Trades published",
            strategy_name=strategy_name,
            num_trades=len(trades),
            file_path=str(file_path),
        )

        return file_path

    def publish_full_report(
        self,
        metrics: BacktestMetrics | PerformanceMetrics,
        comparisons: list[ComparisonResult],
        equity_curve: list[Decimal],
        timestamps: list[datetime],
        trades: list[Trade],
        strategy_name: str,
    ) -> PublishedReport:
        """Publish complete benchmark report with all components.

        This method publishes a comprehensive report including metrics,
        comparisons, equity curve, and trades. It optionally logs everything
        to MLflow if tracking is enabled.

        Args:
            metrics: Performance metrics from backtest
            comparisons: List of benchmark comparison results
            equity_curve: List of equity values over time
            timestamps: Corresponding timestamps for equity values
            trades: List of completed trades
            strategy_name: Name of the strategy

        Returns:
            PublishedReport with metadata about all generated files

        Examples:
            >>> report = publisher.publish_full_report(
            ...     metrics=metrics,
            ...     comparisons=comparisons,
            ...     equity_curve=equity_curve,
            ...     timestamps=timestamps,
            ...     trades=trades,
            ...     strategy_name="momentum_strategy",
            ... )
            >>> print(f"Generated {len(report.file_paths)} files")
            >>> if report.mlflow_run_id:
            ...     print(f"MLflow Run: {report.mlflow_run_id}")
        """
        report_id = self._generate_report_id(strategy_name)
        timestamp = datetime.utcnow()
        file_paths: list[Path] = []
        mlflow_run_id: str | None = None

        logger.info(
            "Publishing full report",
            strategy_name=strategy_name,
            report_id=report_id,
        )

        # Publish metrics
        metrics_path = self.publish_metrics(metrics, strategy_name)
        file_paths.append(metrics_path)

        # Publish comparisons
        if comparisons:
            comparisons_path = self.publish_comparison(comparisons, strategy_name)
            file_paths.append(comparisons_path)

        # Publish equity curve if enabled
        if self.config.include_equity_curve and equity_curve and timestamps:
            equity_path = self.publish_equity_curve(equity_curve, timestamps, strategy_name)
            file_paths.append(equity_path)

        # Publish trades if enabled
        if self.config.include_trades and trades:
            trades_path = self.publish_trades(trades, strategy_name)
            file_paths.append(trades_path)

        # Log to MLflow if enabled
        if self.config.mlflow_tracking:
            try:
                mlflow_run_id = self.log_to_mlflow(metrics, comparisons, strategy_name)
            except Exception as e:
                logger.error(
                    "MLflow logging failed",
                    error=str(e),
                    strategy_name=strategy_name,
                    exc_info=True,
                )

        # Create summary report
        summary_path = self._publish_summary(
            report_id, strategy_name, metrics, comparisons, file_paths, mlflow_run_id
        )
        file_paths.append(summary_path)

        report = PublishedReport(
            report_id=report_id,
            timestamp=timestamp,
            file_paths=file_paths,
            mlflow_run_id=mlflow_run_id,
        )

        logger.info(
            "Full report published",
            strategy_name=strategy_name,
            report_id=report_id,
            num_files=len(file_paths),
            mlflow_run_id=mlflow_run_id,
        )

        return report

    def _publish_summary(
        self,
        report_id: str,
        strategy_name: str,
        metrics: BacktestMetrics | PerformanceMetrics,
        comparisons: list[ComparisonResult],
        file_paths: list[Path],
        mlflow_run_id: str | None,
    ) -> Path:
        """Publish summary report with links to all components.

        Args:
            report_id: Unique report identifier
            strategy_name: Name of the strategy
            metrics: Performance metrics
            comparisons: Benchmark comparisons
            file_paths: List of generated file paths
            mlflow_run_id: MLflow run ID if tracking is enabled

        Returns:
            Path to the summary report file
        """
        sanitized_name = self._sanitize_filename(strategy_name)
        filename = f"{sanitized_name}_summary_{report_id}.json"
        file_path = self.config.output_dir / filename

        # Build metrics summary based on metric type
        from signalforge.ml.backtesting.metrics import BacktestMetrics

        if isinstance(metrics, BacktestMetrics):
            metrics_summary = {
                "total_return": metrics.total_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown,
                "total_trades": metrics.total_trades,
            }
        else:  # PerformanceMetrics
            metrics_summary = {
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "max_drawdown": metrics.max_drawdown,
                "cagr": metrics.cagr,
                "total_trades": metrics.total_trades,
            }

        summary = {
            "report_id": report_id,
            "strategy_name": strategy_name,
            "generated_at": datetime.utcnow().isoformat(),
            "metrics_summary": metrics_summary,
            "benchmark_summary": [
                {
                    "benchmark": comp.benchmark_name,
                    "alpha": comp.alpha,
                    "information_ratio": comp.information_ratio,
                }
                for comp in comparisons
            ]
            if comparisons
            else [],
            "files": [str(p) for p in file_paths],
            "mlflow_run_id": mlflow_run_id,
        }

        with open(file_path, "w") as f:
            json.dump(summary, f, indent=2)

        return file_path

    def log_to_mlflow(
        self,
        metrics: BacktestMetrics | PerformanceMetrics,
        comparisons: list[ComparisonResult],
        strategy_name: str,
    ) -> str:
        """Log metrics and comparisons to MLflow.

        This method logs all performance metrics and benchmark comparisons
        to MLflow for experiment tracking. It requires an active MLflow run.

        Args:
            metrics: Performance metrics from backtest
            comparisons: List of benchmark comparison results
            strategy_name: Name of the strategy

        Returns:
            MLflow run ID

        Raises:
            RuntimeError: If MLflow is not installed or no active run exists
            ImportError: If MLflow package is not available

        Examples:
            >>> import mlflow
            >>> from signalforge.ml.training.mlflow_config import start_run
            >>>
            >>> with start_run(run_name="backtest_momentum"):
            ...     run_id = publisher.log_to_mlflow(metrics, comparisons, "momentum")
            ...     print(f"Logged to run: {run_id}")
        """
        try:
            import mlflow
        except ImportError as e:
            logger.error("MLflow not installed")
            raise ImportError("MLflow is required for tracking. Install with: pip install mlflow") from e

        active_run = mlflow.active_run()
        if active_run is None:
            error_msg = "No active MLflow run. Use mlflow.start_run() context manager first."
            logger.error("log_to_mlflow_failed", error=error_msg)
            raise RuntimeError(error_msg)

        run_id: str = active_run.info.run_id

        logger.info(
            "Logging to MLflow",
            strategy_name=strategy_name,
            run_id=run_id,
        )

        # Log strategy name as parameter
        mlflow.log_param("strategy_name", strategy_name)

        # Log all metrics
        metrics_dict = self._metrics_to_dict(metrics)
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"backtest_{key}", value)

        # Log comparison metrics for each benchmark
        for comp in comparisons:
            benchmark_prefix = f"vs_{comp.benchmark_name}"
            mlflow.log_metric(f"{benchmark_prefix}_alpha", comp.alpha)
            mlflow.log_metric(f"{benchmark_prefix}_beta", comp.beta)
            mlflow.log_metric(f"{benchmark_prefix}_correlation", comp.correlation)
            mlflow.log_metric(f"{benchmark_prefix}_tracking_error", comp.tracking_error)
            mlflow.log_metric(f"{benchmark_prefix}_information_ratio", comp.information_ratio)
            mlflow.log_metric(f"{benchmark_prefix}_up_capture", comp.up_capture)
            mlflow.log_metric(f"{benchmark_prefix}_down_capture", comp.down_capture)

        logger.info(
            "MLflow logging complete",
            strategy_name=strategy_name,
            run_id=run_id,
            num_metrics=len(metrics_dict),
            num_comparisons=len(comparisons),
        )

        return run_id

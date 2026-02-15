# Benchmark Comparator

The Benchmark Comparator module provides comprehensive tools for comparing trading strategy performance against market benchmarks. It calculates key relative performance metrics including alpha, beta, correlation, tracking error, information ratio, and capture ratios.

## Features

- Calculate alpha (excess return over benchmark)
- Calculate beta (systematic risk)
- Measure correlation with benchmarks
- Calculate tracking error
- Compute information ratio
- Determine up/down capture ratios
- Support for multiple benchmarks
- Built on Polars for high performance
- Full type safety with strict mypy compliance

## Installation

The module is part of the SignalForge package. Ensure you have the required dependencies:

```bash
pip install polars>=0.20.0
```

## Quick Start

```python
from decimal import Decimal
import polars as pl
from signalforge.benchmark.comparator import BenchmarkComparator

# Initialize with risk-free rate
comparator = BenchmarkComparator(risk_free_rate=0.02)

# Add a benchmark
spy_returns = pl.Series([0.01, 0.02, -0.01, 0.015])
spy_equity = [
    Decimal("10000"),
    Decimal("10100"),
    Decimal("10302"),
    Decimal("10199"),
    Decimal("10352")
]
comparator.add_benchmark("SPY", spy_returns, spy_equity)

# Compare your strategy
strategy_returns = pl.Series([0.012, 0.022, -0.008, 0.018])
result = comparator.compare(strategy_returns, "SPY", "MyStrategy")

print(f"Alpha: {result.alpha:.2f}%")
print(f"Beta: {result.beta:.2f}")
print(f"Information Ratio: {result.information_ratio:.2f}")
```

## Core Components

### BenchmarkData

Dataclass for storing benchmark information:

```python
@dataclass
class BenchmarkData:
    name: str  # Benchmark identifier
    returns: pl.Series  # Period returns (as decimals)
    equity_curve: list[Decimal]  # Equity values over time
```

### ComparisonResult

Dataclass containing all comparison metrics:

```python
@dataclass
class ComparisonResult:
    strategy_name: str
    benchmark_name: str
    alpha: float  # Annualized excess return (%)
    beta: float  # Systematic risk
    correlation: float  # Correlation coefficient [-1, 1]
    tracking_error: float  # Annualized (%)
    information_ratio: float  # Alpha / Tracking Error
    up_capture: float  # % of benchmark gains captured
    down_capture: float  # % of benchmark losses captured
    relative_drawdown: float  # Maximum relative drawdown (%)
```

### BenchmarkComparator

Main class for benchmark comparison:

```python
class BenchmarkComparator:
    def __init__(self, risk_free_rate: float = 0.0) -> None
    def add_benchmark(self, name: str, returns: pl.Series,
                     equity_curve: list[Decimal]) -> None
    def calculate_alpha(self, strategy_returns: pl.Series,
                       benchmark_returns: pl.Series) -> float
    def calculate_beta(self, strategy_returns: pl.Series,
                      benchmark_returns: pl.Series) -> float
    def calculate_correlation(self, strategy_returns: pl.Series,
                             benchmark_returns: pl.Series) -> float
    def calculate_tracking_error(self, strategy_returns: pl.Series,
                                benchmark_returns: pl.Series) -> float
    def calculate_information_ratio(self, strategy_returns: pl.Series,
                                   benchmark_returns: pl.Series) -> float
    def calculate_capture_ratios(self, strategy_returns: pl.Series,
                                benchmark_returns: pl.Series) -> tuple[float, float]
    def compare(self, strategy_returns: pl.Series, benchmark_name: str,
               strategy_name: str = "Strategy") -> ComparisonResult
    def compare_all(self, strategy_returns: pl.Series,
                   strategy_name: str = "Strategy") -> list[ComparisonResult]
```

## Metric Definitions

### Alpha (Jensen's Alpha)

Measures excess return over the benchmark after adjusting for systematic risk:

```
Alpha = Rs - (Rf + Beta * (Rb - Rf))
```

Where:
- Rs = Strategy return (annualized)
- Rb = Benchmark return (annualized)
- Rf = Risk-free rate
- Beta = Strategy beta

**Interpretation:**
- Alpha > 0: Outperformance
- Alpha = 0: Matches expected return
- Alpha < 0: Underperformance

### Beta

Measures systematic risk relative to the benchmark:

```
Beta = Cov(Strategy, Benchmark) / Var(Benchmark)
```

**Interpretation:**
- Beta = 1.0: Same volatility as benchmark
- Beta > 1.0: More volatile than benchmark
- Beta < 1.0: Less volatile than benchmark
- Beta < 0: Inversely correlated with benchmark

### Correlation

Pearson correlation coefficient measuring linear relationship:

```
Correlation = Cov(Strategy, Benchmark) / (Std(Strategy) * Std(Benchmark))
```

**Interpretation:**
- 1.0: Perfect positive correlation
- 0.0: No linear relationship
- -1.0: Perfect negative correlation

### Tracking Error

Standard deviation of excess returns (annualized):

```
Tracking Error = Std(Strategy Returns - Benchmark Returns) * sqrt(252)
```

**Interpretation:**
- < 2%: Passive/index strategy
- 2-5%: Enhanced index strategy
- > 5%: Active management

### Information Ratio

Risk-adjusted excess return:

```
Information Ratio = Alpha / Tracking Error
```

**Interpretation:**
- IR > 0.5: Good performance
- IR > 1.0: Excellent performance
- IR < 0: Underperformance

### Capture Ratios

Measures performance in up and down markets:

```
Up Capture = Avg(Strategy Returns | Benchmark > 0) / Avg(Benchmark Returns | Benchmark > 0) * 100
Down Capture = Avg(Strategy Returns | Benchmark < 0) / Avg(Benchmark Returns | Benchmark < 0) * 100
```

**Interpretation:**
- Up Capture > 100%: Captures more upside
- Down Capture < 100%: Better downside protection
- Ideal: Up > 100%, Down < 100%

## Advanced Usage

### Compare Against Multiple Benchmarks

```python
comparator = BenchmarkComparator(risk_free_rate=0.02)

# Add multiple benchmarks
comparator.add_benchmark("SPY", spy_returns, spy_equity)
comparator.add_benchmark("QQQ", qqq_returns, qqq_equity)
comparator.add_benchmark("IWM", iwm_returns, iwm_equity)

# Compare against all
results = comparator.compare_all(strategy_returns, "MyStrategy")

# Find best information ratio
best = max(results, key=lambda r: r.information_ratio)
print(f"Best vs {best.benchmark_name}: IR={best.information_ratio:.2f}")
```

### Individual Metric Calculations

```python
comparator = BenchmarkComparator()

# Calculate specific metrics
beta = comparator.calculate_beta(strategy_returns, benchmark_returns)
alpha = comparator.calculate_alpha(strategy_returns, benchmark_returns)
correlation = comparator.calculate_correlation(strategy_returns, benchmark_returns)
tracking_error = comparator.calculate_tracking_error(strategy_returns, benchmark_returns)
ir = comparator.calculate_information_ratio(strategy_returns, benchmark_returns)
up_capture, down_capture = comparator.calculate_capture_ratios(
    strategy_returns, benchmark_returns
)
```

### Export Results

```python
result = comparator.compare(strategy_returns, "SPY", "MyStrategy")

# Convert to dictionary for logging/storage
result_dict = result.to_dict()

# Use with MLflow or other tracking systems
import mlflow
mlflow.log_metrics({
    f"benchmark_{result.benchmark_name}_alpha": result.alpha,
    f"benchmark_{result.benchmark_name}_beta": result.beta,
    f"benchmark_{result.benchmark_name}_ir": result.information_ratio,
})
```

## Best Practices

1. **Use Consistent Time Periods**: Ensure strategy and benchmark returns cover the same time period.

2. **Handle Missing Data**: The module automatically drops null values, but ensure data quality.

3. **Choose Appropriate Benchmarks**: Select benchmarks that match your strategy's asset class and style.

4. **Risk-Free Rate**: Use the appropriate risk-free rate for your analysis period (e.g., 3-month T-Bill rate).

5. **Sufficient Data**: Ensure at least 252 data points (1 trading year) for reliable annualized metrics.

6. **Multiple Benchmarks**: Compare against multiple benchmarks to understand strategy positioning.

## Error Handling

The module handles edge cases gracefully:

```python
# Empty series returns 0.0
empty_series = pl.Series([], dtype=pl.Float64)
beta = comparator.calculate_beta(empty_series, benchmark_returns)
# beta = 0.0

# Mismatched lengths returns 0.0
short_series = pl.Series([0.01, 0.02])
long_series = pl.Series([0.01, 0.02, 0.03, 0.04])
beta = comparator.calculate_beta(short_series, long_series)
# beta = 0.0

# Zero variance benchmark returns 0.0
constant_benchmark = pl.Series([0.01, 0.01, 0.01])
beta = comparator.calculate_beta(strategy_returns, constant_benchmark)
# beta = 0.0
```

## Performance Considerations

- Built on Polars for optimal performance
- Efficient vectorized operations
- Minimal memory overhead
- Suitable for large datasets (millions of data points)

## Testing

The module includes comprehensive tests (61 tests covering):
- Valid inputs
- Edge cases
- Error conditions
- Null handling
- Numerical accuracy
- Type safety

Run tests:
```bash
pytest tests/test_comparator.py -v
```

## Type Safety

Fully type-annotated and passes strict mypy checks:
```bash
mypy src/signalforge/benchmark/comparator.py --strict
```

## License

Part of the SignalForge project. See main project LICENSE for details.

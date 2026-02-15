# Execution Quality Module

## Overview

The execution quality module provides comprehensive tools for assessing whether trading signals are executable under real market conditions. It includes liquidity assessment and slippage estimation to ensure signals can be executed profitably.

## Features

### Liquidity Assessment (`liquidity.py`)

- **Average Daily Volume (ADV) Calculation**: 20-day rolling average of trading volume
- **Volume Volatility Measurement**: Standard deviation of volume for consistency analysis
- **Liquidity Scoring System**: 0-100 score combining dollar volume and consistency
- **Redis Caching**: 1-hour TTL for performance optimization

**Liquidity Score Guidelines:**
- Score > 70: High liquidity (suitable for most trading strategies)
- Score 40-70: Medium liquidity (acceptable with caution)
- Score < 40: Low liquidity (high execution risk)

### Slippage Estimation (`slippage.py`)

- **Market Impact Modeling**: Square-root impact function based on academic literature
- **Volatility-Adjusted Estimates**: Accounts for market conditions
- **Execution Risk Classification**:
  - Low Risk: Order < 1% of ADV
  - Medium Risk: Order 1-5% of ADV
  - High Risk: Order > 5% of ADV

**Slippage Formula:**
```
slippage_pct = base_impact * sqrt(order_size / ADV) * volatility_factor

Where:
- base_impact = 0.1% (10 basis points)
- volatility_factor = current_volatility / 2% (normalized)
```

## Usage Examples

### Basic Liquidity Assessment

```python
import polars as pl
from signalforge.execution import assess_liquidity

# Your OHLCV data
df = pl.DataFrame({
    "timestamp": [...],
    "close": [...],
    "volume": [...],
})

# Assess liquidity
metrics = assess_liquidity(df, symbol="AAPL")

print(f"Liquidity Score: {metrics.liquidity_score:.2f}/100")
print(f"Is Liquid: {metrics.is_liquid}")
```

### Slippage Estimation

```python
from signalforge.execution import estimate_slippage

# Estimate slippage for a $100,000 order
estimate = estimate_slippage(
    order_size_usd=100_000,
    avg_daily_volume=5_000_000,  # shares
    current_price=150.0,
    volatility=0.02,  # 2% daily volatility
    symbol="AAPL",
)

print(f"Estimated Slippage: {estimate.estimated_slippage_pct:.4f}%")
print(f"Cost: ${estimate.estimated_slippage_usd:.2f}")
print(f"Risk Level: {estimate.execution_risk}")
```

### Using Redis Cache

```python
from signalforge.core.redis import get_redis
from signalforge.execution import get_cached_liquidity_metrics

redis = await get_redis()
metrics = await get_cached_liquidity_metrics(df, "AAPL", redis)
```

### Complete Signal Validation Workflow

```python
from signalforge.execution import assess_liquidity, estimate_slippage

# Step 1: Check liquidity
liquidity = assess_liquidity(price_data, "AAPL")
if not liquidity.is_liquid:
    print("Signal rejected: Insufficient liquidity")
    return

# Step 2: Estimate slippage
current_price = float(price_data["close"][-1])
slippage = estimate_slippage(
    order_size_usd=signal_order_size,
    avg_daily_volume=liquidity.avg_daily_volume,
    current_price=current_price,
    volatility=0.02,
)

# Step 3: Validate against thresholds
if slippage.estimated_slippage_pct > max_acceptable_slippage:
    print("Signal rejected: Excessive slippage")
    return

if slippage.execution_risk == "high":
    print("Warning: High execution risk - consider splitting order")

print("Signal approved for execution")
```

## Architecture

### Data Classes

**LiquidityMetrics:**
- `symbol`: Asset symbol
- `avg_daily_volume`: Average daily volume (shares)
- `volume_volatility`: Standard deviation of volume
- `liquidity_score`: 0-100 composite score
- `is_liquid`: Boolean flag for quick filtering

**SlippageEstimate:**
- `symbol`: Asset symbol
- `order_size`: Order size in USD
- `estimated_slippage_pct`: Slippage percentage
- `estimated_slippage_usd`: Slippage cost in dollars
- `adv_ratio`: Order size / ADV ratio
- `execution_risk`: Risk level (low/medium/high)

### Functions

**Liquidity Module:**
- `calculate_avg_daily_volume()`: Calculate ADV over window
- `calculate_liquidity_score()`: Compute composite liquidity score
- `assess_liquidity()`: Complete liquidity assessment
- `get_cached_liquidity_metrics()`: Cached assessment with Redis

**Slippage Module:**
- `calculate_execution_risk()`: Classify execution risk
- `estimate_slippage()`: Estimate slippage and costs

## Testing

The module includes comprehensive test coverage:

- **63 total tests** across all functionality
- **Unit tests** for individual functions
- **Integration tests** for complete workflows
- **Edge case tests** for error handling
- **Validation tests** for data classes

Run tests:
```bash
pytest tests/test_execution_quality.py -v
```

## Performance Considerations

### Redis Caching

Liquidity metrics are cached with a 1-hour TTL to reduce database load:
- First call: Calculates and caches (~50ms)
- Subsequent calls: Returns from cache (~5ms)

### Computational Complexity

- **Liquidity Assessment**: O(n) where n = window size (default 20)
- **Slippage Estimation**: O(1) - constant time calculation

### Best Practices

1. **Cache frequently accessed symbols** using `get_cached_liquidity_metrics()`
2. **Use appropriate window sizes** (20 days is standard, but adjust for your needs)
3. **Update volatility estimates** regularly based on recent market data
4. **Monitor execution results** to calibrate slippage model parameters

## Integration with SignalForge

This module integrates seamlessly with:

- **Data Ingestion**: Uses OHLCV data from TimescaleDB
- **ML Pipeline**: Validates signals before backtesting
- **Risk Management**: Provides inputs for position sizing
- **Logging**: Uses structured logging (structlog) throughout

## References

### Academic Literature

- Almgren, R., & Chriss, N. (2000). "Optimal execution of portfolio transactions"
- Grinold, R., & Kahn, R. (2000). "Active Portfolio Management"
- Kyle, A. S. (1985). "Continuous auctions and insider trading"

### Industry Standards

- CFA Institute - Transaction Cost Analysis
- Market microstructure best practices
- Institutional trading guidelines

## Future Enhancements

Potential improvements for future versions:

1. **Historical Slippage Tracking**: Learn from actual execution data
2. **Intraday Patterns**: Incorporate time-of-day effects
3. **Spread Analysis**: Add bid-ask spread estimation
4. **Market Depth Integration**: Use order book data when available
5. **Smart Order Routing**: Optimize execution venue selection
6. **Adaptive Parameters**: Machine learning for parameter tuning

## License

Part of SignalForge - Algorithmic Trading Platform

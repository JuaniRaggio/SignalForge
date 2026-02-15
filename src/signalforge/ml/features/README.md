# Feature Engineering Module

This module provides feature engineering capabilities for SignalForge's ML pipelines, with a focus on technical indicators for financial time series analysis.

## Technical Indicators

The `technical.py` module implements efficient computation of common technical indicators using Polars DataFrames.

### Supported Indicators

| Indicator | Description | Default Period(s) |
|-----------|-------------|-------------------|
| **SMA** | Simple Moving Average | 10, 20, 50, 200 |
| **EMA** | Exponential Moving Average | 12, 26 |
| **RSI** | Relative Strength Index | 14 |
| **MACD** | Moving Average Convergence Divergence | 12, 26, 9 |
| **Bollinger Bands** | Volatility bands around SMA | 20, 2 std |
| **ATR** | Average True Range | 14 |
| **Volume SMA** | Volume moving average | 20 |

### Usage

#### Basic Usage with Default Configuration

```python
import polars as pl
from signalforge.ml.features.technical import compute_technical_indicators

# Assuming you have OHLCV data in a Polars DataFrame
df = pl.read_parquet("price_data.parquet")

# Compute all indicators with default configuration
df_with_indicators = compute_technical_indicators(df)

# The result includes all original columns plus 15 new indicator columns
print(df_with_indicators.columns)
```

#### Custom Configuration

```python
from signalforge.ml.features.technical import IndicatorConfig, compute_technical_indicators

# Create custom configuration for day trading
config = IndicatorConfig(
    sma_periods=(5, 10, 20),      # Shorter SMAs for day trading
    ema_periods=(8, 13, 21),       # Fibonacci EMAs
    rsi_period=9,                  # Faster RSI
    macd_fast=8,
    macd_slow=17,
    macd_signal=9,
    bb_period=15,
    atr_period=10,
)

df_with_indicators = compute_technical_indicators(df, config)
```

#### Computing Individual Indicators

```python
from signalforge.ml.features.technical import (
    compute_sma,
    compute_ema,
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_atr,
    compute_volume_sma,
)

# Compute individual indicators
df = compute_sma(df, period=20)
df = compute_ema(df, period=12)
df = compute_rsi(df, period=14)
df = compute_macd(df, fast_period=12, slow_period=26, signal_period=9)
df = compute_bollinger_bands(df, period=20, std_multiplier=2.0)
df = compute_atr(df, period=14)
df = compute_volume_sma(df, period=20)
```

### Input Data Requirements

The input DataFrame must contain these columns:

- `symbol`: str - Stock ticker symbol
- `timestamp`: datetime - Timestamp with timezone info
- `open`: float - Opening price
- `high`: float - Highest price
- `low`: float - Lowest price
- `close`: float - Closing price
- `volume`: int - Trading volume

### Output Columns

With default configuration, the following columns are added:

- `sma_10`, `sma_20`, `sma_50`, `sma_200`: Simple moving averages
- `ema_12`, `ema_26`: Exponential moving averages
- `rsi_14`: Relative Strength Index
- `macd`, `macd_signal`, `macd_histogram`: MACD components
- `bb_upper`, `bb_middle`, `bb_lower`: Bollinger Bands
- `atr_14`: Average True Range
- `volume_sma_20`: Volume moving average

### Performance Characteristics

- **Efficient**: Uses Polars for fast computation on large datasets
- **Memory-efficient**: Streaming operations where possible
- **Null handling**: Properly handles missing values for initial periods
- **Type-safe**: Full mypy strict compliance

### Example: Full Pipeline

```python
import asyncio
from datetime import datetime, timedelta
import polars as pl
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.core.database import get_async_session_maker
from signalforge.core.config import get_settings
from signalforge.ml.features.technical import compute_technical_indicators
from signalforge.models.price import Price


async def process_symbol(symbol: str) -> pl.DataFrame:
    """Fetch and process technical indicators for a symbol."""
    settings = get_settings()
    async_session_maker = get_async_session_maker(settings.database_url)

    async with async_session_maker() as session:
        # Fetch price data from TimescaleDB
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        stmt = (
            select(Price)
            .where(Price.symbol == symbol.upper())
            .where(Price.timestamp >= start_date)
            .where(Price.timestamp <= end_date)
            .order_by(Price.timestamp)
        )

        result = await session.execute(stmt)
        prices = result.scalars().all()

        # Convert to Polars DataFrame
        df = pl.DataFrame({
            "symbol": [p.symbol for p in prices],
            "timestamp": [p.timestamp for p in prices],
            "open": [float(p.open) for p in prices],
            "high": [float(p.high) for p in prices],
            "low": [float(p.low) for p in prices],
            "close": [float(p.close) for p in prices],
            "volume": [p.volume for p in prices],
        })

        # Compute indicators
        df_with_indicators = compute_technical_indicators(df)

        # Save for downstream ML processing
        df_with_indicators.write_parquet(f"/data/features/{symbol}_indicators.parquet")

        return df_with_indicators


# Run for multiple symbols
symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
results = await asyncio.gather(*[process_symbol(s) for s in symbols])
```

### Testing

Comprehensive tests are available in `tests/test_technical_indicators.py`:

```bash
pytest tests/test_technical_indicators.py -v
```

All tests include:
- Validation of indicator formulas
- Edge case handling (empty data, single row, nulls)
- Extreme volatility scenarios
- Multiple symbols processing
- Configuration validation

### References

Technical indicator implementations follow standard financial analysis formulas:

- **SMA**: Simple arithmetic mean over n periods
- **EMA**: Exponential smoothing with alpha = 2/(n+1)
- **RSI**: J. Welles Wilder Jr.'s Relative Strength Index formula
- **MACD**: Gerald Appel's MACD calculation
- **Bollinger Bands**: John Bollinger's volatility bands
- **ATR**: J. Welles Wilder Jr.'s Average True Range

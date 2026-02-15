# Order Flow Module

El módulo Order Flow detecta flujo institucional y actividad inusual en el mercado.

## Componentes

### 1. Dark Pools (dark_pools.py)

Procesador de actividad en dark pools (Alternative Trading Systems).

#### DarkPoolProcessor

Analiza prints de dark pool y detecta patrones institucionales.

**Métodos principales:**

- `process_ats_data(data)`: Procesa datos de ATS y calcula z-scores
- `get_dark_pool_summary(symbol, days)`: Resumen de actividad en dark pools
- `detect_large_prints(symbol, threshold_usd)`: Detecta prints grandes
- `calculate_dark_pool_ratio(symbol)`: Calcula porcentaje de volumen en dark pools
- `get_institutional_bias(symbol)`: Determina sesgo institucional

### 2. Options Activity (options_activity.py)

Detector de actividad inusual en opciones.

#### OptionsActivityDetector

Analiza flujo de opciones y detecta actividad institucional.

**Métodos principales:**

- `detect_unusual_activity(symbol, volume_threshold)`: Detecta actividad inusual
- `get_put_call_ratio(symbol)`: Calcula ratio put/call
- `get_options_flow_summary(symbol, days)`: Resumen de flujo de opciones
- `detect_large_premium_trades(symbol, threshold_usd)`: Detecta trades grandes
- `calculate_options_sentiment(symbol)`: Calcula sentimiento de opciones
- `get_expiry_concentration(symbol)`: Concentración por fecha de vencimiento

### 3. Short Interest (short_interest.py)

Rastreador de short interest y detección de squeeze candidates.

#### ShortInterestTracker

Monitorea datos de short interest y detecta oportunidades de short squeeze.

**Métodos principales:**

- `get_current_short_interest(symbol)`: Obtiene short interest actual
- `get_short_interest_history(symbol, reports)`: Historial de short interest
- `detect_short_squeeze_candidates(min_short_percent, min_days_to_cover)`: Detecta candidatos
- `calculate_short_interest_change(symbol)`: Calcula cambio en short interest
- `get_most_shorted_stocks(top_n)`: Stocks más shorteados

### 4. Aggregator (aggregator.py)

Agregador de flujos desde múltiples fuentes.

#### FlowAggregator

Combina y analiza flujos de dark pools, opciones y short interest.

**Métodos principales:**

- `aggregate_flows(symbol, days)`: Agrega todos los flujos
- `calculate_net_flow(symbol)`: Calcula flujo neto (bullish - bearish)
- `calculate_flow_z_score(symbol, lookback_days)`: Z-score del flujo
- `get_flow_momentum(symbol)`: Momentum del cambio en dirección
- `rank_by_institutional_interest(symbols)`: Rankea por interés institucional

### 5. Anomaly Detector (anomaly_detector.py)

Detector de anomalías y patrones en order flow.

#### FlowAnomalyDetector

Identifica patrones anómalos que indican actividad institucional inusual.

**Métodos principales:**

- `detect_anomalies(symbol, sensitivity)`: Detecta todas las anomalías
- `detect_volume_spike(symbol, threshold_std)`: Detecta spikes de volumen
- `detect_options_sweep(symbol)`: Detecta sweeps de opciones
- `detect_accumulation_pattern(symbol, days)`: Detecta acumulación
- `detect_distribution_pattern(symbol, days)`: Detecta distribución
- `get_anomaly_score(symbol)`: Score de anomalía (0-100)

## Modelos de Datos

### OrderFlowRecord

Registro general de actividad de order flow.

**Campos:**
- symbol: Símbolo del activo
- flow_type: Tipo de flujo (dark_pool, options, short_interest, etc.)
- direction: Dirección (bullish, bearish, neutral)
- value: Valor en dólares
- volume: Volumen de acciones
- is_unusual: Flag de actividad inusual
- z_score: Qué tan inusual es (desviaciones estándar)

### OptionsActivity

Actividad específica de opciones.

**Campos:**
- option_type: call/put
- strike: Precio de strike
- expiry: Fecha de vencimiento
- volume: Volumen del contrato
- open_interest: Open interest
- premium: Premium total
- implied_volatility: IV
- delta: Delta del contrato

### ShortInterest

Datos de short interest.

**Campos:**
- short_interest: Acciones en short
- shares_outstanding: Acciones en circulación
- short_percent: Porcentaje shorteado
- days_to_cover: Días para cubrir
- change_percent: Cambio porcentual

## Schemas Pydantic

### FlowAggregation

Agregación de flujos para un símbolo.

```python
FlowAggregation(
    symbol="AAPL",
    period_days=5,
    net_flow=1500000.0,
    bullish_flow=2000000.0,
    bearish_flow=500000.0,
    bias=FlowDirection.BULLISH,
    z_score=2.5,
    flow_momentum=0.7,
    dark_pool_volume=150000,
    options_premium=500000.0,
    short_interest_change=-5.0
)
```

### FlowAnomaly

Anomalía detectada en el flujo.

```python
FlowAnomaly(
    symbol="GME",
    timestamp=datetime.now(UTC),
    anomaly_type="volume_spike",
    severity=AnomalySeverity.HIGH,
    description="Unusual volume spike detected",
    z_score=4.2,
    metadata={"spike_ratio": 5.0}
)
```

## Uso Típico

```python
from signalforge.orderflow import (
    DarkPoolProcessor,
    OptionsActivityDetector,
    ShortInterestTracker,
    FlowAggregator,
    FlowAnomalyDetector
)

async def analyze_symbol(session, symbol: str):
    # Dark pools
    dp_processor = DarkPoolProcessor(session)
    dp_summary = await dp_processor.get_dark_pool_summary(symbol, days=30)

    # Options
    opt_detector = OptionsActivityDetector(session)
    unusual = await opt_detector.detect_unusual_activity(symbol)

    # Short interest
    si_tracker = ShortInterestTracker(session)
    current_si = await si_tracker.get_current_short_interest(symbol)

    # Agregación
    aggregator = FlowAggregator(session)
    flows = await aggregator.aggregate_flows(symbol, days=10)

    # Anomalías
    detector = FlowAnomalyDetector(session)
    anomalies = await detector.detect_anomalies(symbol, sensitivity=2.0)
    score = await detector.get_anomaly_score(symbol)

    return {
        "dark_pool": dp_summary,
        "options": unusual,
        "short_interest": current_si,
        "flows": flows,
        "anomalies": anomalies,
        "anomaly_score": score
    }
```

## Tests

El módulo incluye más de 120 tests comprehensivos:

- `test_dark_pools.py`: 12 tests para dark pool processing
- `test_options_activity.py`: 36 tests para opciones
- `test_short_interest.py`: 24 tests para short interest
- `test_flow_aggregator.py`: 24 tests para agregación
- `test_flow_anomaly.py`: 24 tests para detección de anomalías

Ejecutar tests:

```bash
pytest tests/test_dark_pools.py -v
pytest tests/test_options_activity.py -v
pytest tests/test_short_interest.py -v
pytest tests/test_flow_aggregator.py -v
pytest tests/test_flow_anomaly.py -v
```

## Características

- Procesamiento asíncrono con SQLAlchemy 2.0
- Análisis de datos con Polars (optimizado para rendimiento)
- Type hints completos (mypy strict mode)
- Logging estructurado con structlog
- Docstrings Google style
- Edge case handling comprehensivo
- Detección de anomalías multi-criterio
- Z-score calculation para identificar actividad inusual

## Dependencias

- SQLAlchemy 2.0+ (async)
- Polars 0.20+
- structlog 24.1+
- Pydantic 2.5+

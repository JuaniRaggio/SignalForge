"""Market regime detection module.

This module provides Hidden Markov Model (HMM) based regime detection
for financial markets. It identifies distinct market states such as:
- Bull markets (uptrends)
- Bear markets (downtrends)
- Range-bound markets (sideways/consolidation)
- Crisis periods (high volatility and sharp declines)

The module uses multiple features including returns, volatility, volume,
and trend indicators to train and predict market regimes.
"""

from __future__ import annotations

from signalforge.regime.detector import Regime, RegimeConfig, RegimeDetector

__all__ = [
    "Regime",
    "RegimeConfig",
    "RegimeDetector",
]

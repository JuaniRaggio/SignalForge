"""NLP signal generation and aggregation."""

from signalforge.nlp.signals.aggregator import AggregatedSignal, SignalAggregator
from signalforge.nlp.signals.generator import (
    AnalystConsensus,
    NLPSignalGenerator,
    NLPSignalOutput,
    SentimentOutput,
)

__all__ = [
    "AnalystConsensus",
    "AggregatedSignal",
    "NLPSignalGenerator",
    "NLPSignalOutput",
    "SentimentOutput",
    "SignalAggregator",
]

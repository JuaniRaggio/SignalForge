# Named Entity Recognition (NER) for Financial Text

The NER module provides state-of-the-art named entity recognition capabilities specifically tuned for financial documents and market intelligence.

## Features

- **spaCy-based extraction**: Leverages spaCy's pre-trained NER models for standard entities
- **Financial entity patterns**: Custom patterns for tickers, money amounts, and percentages
- **Lazy model loading**: Models are loaded on-demand to optimize resource usage
- **Batch processing**: Efficient batch processing with configurable batch sizes
- **Mock-friendly design**: Easy to test with dependency injection support
- **Confidence scoring**: Filter entities by confidence threshold

## Installation

```bash
pip install spacy>=3.7.0
python -m spacy download en_core_web_sm
```

## Quick Start

### Basic Entity Extraction

```python
from signalforge.nlp.ner import extract_entities

text = "Apple Inc. (AAPL) reported revenue of $1.5B, up 15% from last year."
result = extract_entities(text)

for entity in result.entities:
    print(f"{entity.text} ({entity.label})")
```

### Extract Ticker Symbols

```python
from signalforge.nlp.ner import extract_tickers

text = "Trading $AAPL and MSFT today."
tickers = extract_tickers(text)
print(tickers)  # ['AAPL', 'MSFT']
```

### Batch Processing

```python
from signalforge.nlp.ner import get_entity_extractor, NERConfig

config = NERConfig(batch_size=32, confidence_threshold=0.8)
extractor = get_entity_extractor(config)

texts = [
    "Microsoft reported revenue of $50B.",
    "Tesla's stock rose 20% today.",
]
results = extractor.extract_batch(texts)
```

## Entity Types

### Standard Entities (from spaCy)

- **PERSON**: People names
- **ORG**: Organizations, companies
- **GPE**: Geopolitical entities (cities, countries)
- **DATE**: Date expressions
- **TIME**: Time expressions
- **MONEY**: Monetary amounts (enhanced with custom patterns)
- **PERCENT**: Percentages (enhanced with custom patterns)
- **CARDINAL**: Numerals
- **ORDINAL**: Ordinal numbers

### Custom Financial Entities

- **TICKER**: Stock ticker symbols
  - Patterns: `$AAPL`, `MSFT`, `GOOGL`
  - Extracted as uppercase symbols between 2-5 characters

- **MONEY**: Monetary amounts (enhanced)
  - Patterns: `$1.5B`, `50M USD`, `$100.25`
  - Supports: K (thousands), M (millions), B (billions), T (trillions)

- **PERCENT**: Percentages (enhanced)
  - Patterns: `15%`, `3.5 percent`, `25.5%`
  - Supports: `%`, `percent`, `percentage`, `pct`

## Data Classes

### NamedEntity

Represents a single extracted entity.

```python
@dataclass
class NamedEntity:
    text: str           # Entity text
    label: str          # Entity type (ORG, TICKER, etc.)
    start: int          # Character offset (start)
    end: int            # Character offset (end)
    confidence: float   # Confidence score (0.0-1.0)
```

### EntityExtractionResult

Result of entity extraction on a document.

```python
@dataclass
class EntityExtractionResult:
    text: str                      # Original text
    entities: list[NamedEntity]    # Extracted entities
    entity_counts: dict[str, int]  # Count by label
```

### NERConfig

Configuration for entity extraction.

```python
@dataclass
class NERConfig:
    model_name: str = "en_core_web_sm"       # spaCy model
    include_custom_patterns: bool = True      # Add financial patterns
    confidence_threshold: float = 0.5         # Minimum confidence
    batch_size: int = 32                      # Batch size
    enable_ner: bool = True                   # Enable NER pipeline
    enable_entity_ruler: bool = True          # Enable entity ruler
    cache_model: bool = True                  # Cache loaded model
```

## Classes

### BaseEntityExtractor (ABC)

Abstract base class defining the extractor interface.

```python
class BaseEntityExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> EntityExtractionResult:
        pass

    @abstractmethod
    def extract_batch(self, texts: list[str]) -> list[EntityExtractionResult]:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass
```

### SpaCyEntityExtractor

Standard spaCy-based entity extractor.

```python
extractor = SpaCyEntityExtractor()
result = extractor.extract("Apple Inc. is in Cupertino.")
```

**Features:**
- Lazy model loading
- Model caching across instances
- Confidence threshold filtering
- Batch processing with `nlp.pipe()`

### FinancialEntityExtractor

Extends SpaCyEntityExtractor with custom financial patterns.

```python
extractor = FinancialEntityExtractor()
result = extractor.extract("$AAPL rose 15% to $150.25.")
```

**Additional Features:**
- Custom ticker patterns
- Enhanced money amount detection
- Percentage extraction
- All features from SpaCyEntityExtractor

## Helper Functions

### get_entity_extractor(config)

Factory function for creating entity extractors.

```python
# Get default singleton instance
extractor = get_entity_extractor()

# Create custom instance
config = NERConfig(confidence_threshold=0.8)
extractor = get_entity_extractor(config)
```

**Behavior:**
- Returns singleton for default config (cached)
- Creates new instance for custom config

### extract_entities(text)

Convenience function for quick entity extraction.

```python
result = extract_entities("Apple Inc. reported $1.5B revenue.")
print(f"Found {len(result.entities)} entities")
```

### extract_tickers(text)

Extract only ticker symbols from text.

```python
tickers = extract_tickers("Trading $AAPL and MSFT today.")
# Returns: ['AAPL', 'MSFT']
```

**Features:**
- Regex-based extraction
- Automatic deduplication
- Filters single-letter symbols
- Returns sorted list

## Advanced Usage

### Custom Configuration

```python
config = NERConfig(
    model_name="en_core_web_lg",           # Use larger model
    include_custom_patterns=True,           # Enable financial patterns
    confidence_threshold=0.8,               # Higher precision
    batch_size=64,                          # Larger batches
    cache_model=True,                       # Cache model
)

extractor = FinancialEntityExtractor(config=config)
```

### Dependency Injection (Testing)

```python
from unittest.mock import Mock

# Inject mock model for testing
mock_model = Mock()
extractor = SpaCyEntityExtractor(model=mock_model)
```

### Filter by Entity Type

```python
result = extract_entities(text)

# Extract specific entity types
orgs = [e for e in result.entities if e.label == "ORG"]
tickers = [e for e in result.entities if e.label == "TICKER"]
money = [e for e in result.entities if e.label == "MONEY"]
```

### Confidence Filtering

```python
config = NERConfig(confidence_threshold=0.9)
extractor = get_entity_extractor(config)

result = extractor.extract(text)
# Only entities with confidence >= 0.9 are returned
```

## Performance Considerations

### Model Caching

Models are cached by default to avoid reloading:

```python
# First instance loads model
extractor1 = FinancialEntityExtractor()
extractor1.extract(text1)

# Second instance uses cached model
extractor2 = FinancialEntityExtractor()
extractor2.extract(text2)
```

### Batch Processing

Use batch processing for multiple texts:

```python
# Efficient: Uses spaCy's pipe for parallelization
extractor = get_entity_extractor()
results = extractor.extract_batch(texts)

# Inefficient: Processes texts sequentially
results = [extractor.extract(text) for text in texts]
```

### Lazy Loading

Models are loaded on first use, not at import:

```python
# No model loaded yet
extractor = FinancialEntityExtractor()

# Model loaded here
result = extractor.extract(text)
```

## Error Handling

```python
from signalforge.nlp.ner import extract_entities

try:
    result = extract_entities("")
except ValueError as e:
    print(f"Invalid input: {e}")

try:
    result = extract_entities(text)
except RuntimeError as e:
    print(f"Extraction failed: {e}")
```

## Testing

The module is designed to be mock-friendly:

```python
from unittest.mock import Mock
from signalforge.nlp.ner import SpaCyEntityExtractor

# Create mock spaCy model
mock_model = Mock()
mock_doc = Mock()
mock_doc.ents = []
mock_model.return_value = mock_doc

# Inject mock for testing
extractor = SpaCyEntityExtractor(model=mock_model)
result = extractor.extract("test text")
```

## Examples

See `/examples/ner_example.py` for comprehensive usage examples:

```bash
python examples/ner_example.py
```

## Integration with Other Modules

### With Preprocessing

```python
from signalforge.nlp.preprocessing import TextPreprocessor
from signalforge.nlp.ner import extract_entities

preprocessor = TextPreprocessor()
text = preprocessor.clean_text(raw_text)
result = extract_entities(text)
```

### With Sentiment Analysis

```python
from signalforge.nlp.ner import extract_entities
from signalforge.nlp.sentiment import analyze_financial_text

# Extract entities
result = extract_entities(text)
tickers = [e.text for e in result.entities if e.label == "TICKER"]

# Analyze sentiment
sentiment = analyze_financial_text(text)

# Combine results
print(f"Sentiment: {sentiment.label} for tickers: {', '.join(tickers)}")
```

## Troubleshooting

### Model Not Found

```
RuntimeError: Failed to load spaCy model en_core_web_sm
```

**Solution:**
```bash
python -m spacy download en_core_web_sm
```

### Import Error

```
ModuleNotFoundError: No module named 'spacy'
```

**Solution:**
```bash
pip install spacy>=3.7.0
```

### Poor Entity Detection

**Solutions:**
1. Use a larger model: `en_core_web_lg` or `en_core_web_trf`
2. Lower confidence threshold: `NERConfig(confidence_threshold=0.3)`
3. Enable custom patterns: `NERConfig(include_custom_patterns=True)`

### Memory Issues with Large Batches

**Solution:**
```python
config = NERConfig(batch_size=16)  # Reduce batch size
extractor = get_entity_extractor(config)
```

## API Reference

For complete API documentation, see the module docstrings:

```python
from signalforge.nlp import ner
help(ner)
```

## License

Apache 2.0

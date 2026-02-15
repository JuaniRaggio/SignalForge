# GICS Sector Classifier

The sector classifier module provides automatic classification of financial documents into Global Industry Classification Standard (GICS) sectors using embedding-based similarity.

## Overview

The sector classifier analyzes financial text and assigns it to one of 11 GICS sectors:

1. **Energy** - Oil, gas, drilling, renewable energy
2. **Materials** - Chemicals, metals, mining, construction materials
3. **Industrials** - Manufacturing, machinery, transportation, aerospace
4. **Consumer Discretionary** - Retail, automobiles, entertainment, luxury
5. **Consumer Staples** - Food, beverage, household products, tobacco
6. **Health Care** - Pharmaceuticals, biotechnology, medical devices
7. **Financials** - Banking, insurance, investment services
8. **Information Technology** - Software, hardware, semiconductors, IT services
9. **Communication Services** - Telecommunications, media, social platforms
10. **Utilities** - Electric, water, gas utilities
11. **Real Estate** - REITs, property management, real estate services

## Key Features

- **Embedding-based Classification**: Uses sentence embeddings for semantic similarity
- **Confidence Scoring**: Returns confidence scores for all sectors
- **Batch Processing**: Efficiently classify multiple documents
- **Configurable Thresholds**: Customize similarity thresholds
- **Pre-defined Descriptions**: Each sector has multiple reference descriptions
- **Caching**: Sector embeddings are cached for performance

## Installation

The sector classifier requires the following dependencies:

```bash
pip install sentence-transformers torch
```

## Quick Start

### Basic Classification

```python
from signalforge.nlp.sector_classifier import classify_sector

text = "Apple announced new AI features in their software products"
prediction = classify_sector(text)

print(f"Sector: {prediction.sector}")
print(f"Confidence: {prediction.confidence:.2f}")
```

### Batch Classification

```python
from signalforge.nlp.sector_classifier import get_sector_classifier

classifier = get_sector_classifier()

texts = [
    "Banking profits increased significantly",
    "Oil production expanded in the region",
    "New pharmaceutical drug approved by FDA",
]

predictions = classifier.classify_batch(texts)

for text, pred in zip(texts, predictions):
    print(f"{text[:50]}: {pred.sector} ({pred.confidence:.2f})")
```

### Custom Configuration

```python
from signalforge.nlp.sector_classifier import (
    get_sector_classifier,
    SectorClassifierConfig,
)

config = SectorClassifierConfig(
    model_name="all-mpnet-base-v2",  # Higher quality model
    similarity_threshold=0.5,          # Require higher confidence
    top_k=5,                           # Return top 5 sectors
)

classifier = get_sector_classifier(config)
prediction = classifier.classify("Financial text here...")
```

## API Reference

### Classes

#### `SectorPrediction`

Result dataclass containing classification results.

**Attributes:**
- `text: str` - Original input text
- `sector: str` - Primary predicted sector
- `confidence: float` - Confidence score (0.0 to 1.0)
- `all_scores: dict[str, float]` - Scores for all sectors
- `metadata: dict[str, Any]` - Additional metadata

**Methods:**
- `get_top_k_sectors(k: int) -> list[tuple[str, float]]` - Get top K sectors

#### `SectorClassifierConfig`

Configuration dataclass for the classifier.

**Attributes:**
- `model_name: str` - Sentence-transformer model (default: "all-MiniLM-L6-v2")
- `similarity_threshold: float` - Minimum similarity score (default: 0.3)
- `top_k: int` - Number of top sectors to include (default: 3)
- `normalize_embeddings: bool` - Normalize embeddings (default: True)
- `cache_embeddings: bool` - Cache sector embeddings (default: True)

#### `EmbeddingSectorClassifier`

Main classifier implementation using embedding similarity.

**Methods:**
- `classify(text: str) -> SectorPrediction` - Classify single text
- `classify_batch(texts: list[str]) -> list[SectorPrediction]` - Classify multiple texts

### Functions

#### `classify_sector(text: str) -> SectorPrediction`

Convenience function for quick classification using defaults.

**Args:**
- `text: str` - Text to classify

**Returns:**
- `SectorPrediction` - Classification result

**Raises:**
- `ValueError` - If text is empty

**Example:**
```python
prediction = classify_sector("Goldman Sachs investment banking revenue grew")
print(prediction.sector)  # "Financials"
```

#### `get_sector_classifier(config: SectorClassifierConfig | None = None) -> EmbeddingSectorClassifier`

Factory function to get a classifier instance.

**Args:**
- `config: SectorClassifierConfig | None` - Configuration (None uses defaults)

**Returns:**
- `EmbeddingSectorClassifier` - Classifier instance

**Example:**
```python
classifier = get_sector_classifier()
prediction = classifier.classify("Technology company launches AI product")
```

#### `get_all_sectors() -> list[str]`

Get list of all GICS sector names.

**Returns:**
- `list[str]` - List of 11 GICS sectors

**Example:**
```python
sectors = get_all_sectors()
print(len(sectors))  # 11
```

## How It Works

### Embedding-Based Similarity

1. **Pre-defined Descriptions**: Each sector has 5 reference descriptions containing relevant keywords and phrases
2. **Embedding Generation**: Text is converted to dense vector representation using sentence-transformers
3. **Similarity Computation**: Cosine similarity is computed between input text and all sector descriptions
4. **Score Aggregation**: Similarities are averaged across all descriptions for each sector
5. **Sector Selection**: Sector with highest average similarity is selected

### Confidence Scoring

Confidence scores are normalized cosine similarities:
- **0.0 - 0.3**: Low confidence (may indicate ambiguous text)
- **0.3 - 0.6**: Medium confidence
- **0.6 - 1.0**: High confidence

### Performance Optimization

- **Caching**: Sector embeddings are computed once and cached
- **Batch Processing**: Multiple texts processed efficiently in batches
- **GPU Acceleration**: Automatic GPU detection and usage if available

## Examples

### Example 1: Energy Sector

```python
text = "ExxonMobil announced plans to expand oil drilling operations"
prediction = classify_sector(text)

assert prediction.sector == "Energy"
assert prediction.confidence > 0.5
```

### Example 2: Technology Sector

```python
text = "Microsoft Azure cloud services revenue increased 30%"
prediction = classify_sector(text)

assert prediction.sector == "Information Technology"

# Get top 3 sectors
top_3 = prediction.get_top_k_sectors(3)
print(top_3)
# [('Information Technology', 0.82), ('Communication Services', 0.65), ...]
```

### Example 3: Analyzing All Scores

```python
text = "Pharmaceutical company develops new cancer treatment"
prediction = classify_sector(text)

# Print all sector scores
for sector, score in sorted(prediction.all_scores.items(),
                            key=lambda x: x[1],
                            reverse=True):
    print(f"{sector:30s} {score:.3f}")
```

### Example 4: Custom Threshold

```python
config = SectorClassifierConfig(similarity_threshold=0.6)
classifier = get_sector_classifier(config)

prediction = classifier.classify("Ambiguous text...")

# Check if threshold was met
if prediction.metadata["threshold_met"]:
    print(f"High confidence: {prediction.sector}")
else:
    print(f"Low confidence: {prediction.sector} ({prediction.confidence:.2f})")
```

## Testing

The module includes comprehensive tests:

```bash
# Run unit tests
pytest tests/test_sector_classifier.py -v

# Run with coverage
pytest tests/test_sector_classifier.py --cov=signalforge.nlp.sector_classifier

# Run integration tests (requires sentence-transformers)
pytest tests/test_sector_classifier.py -m integration
```

## Performance Considerations

### Model Selection

Different models offer trade-offs between speed and accuracy:

- **all-MiniLM-L6-v2** (default): Fast, 384 dimensions, good accuracy
- **all-mpnet-base-v2**: Slower, 768 dimensions, higher accuracy
- **paraphrase-multilingual-MiniLM-L12-v2**: Multilingual support

### GPU Acceleration

GPU significantly improves performance for batch processing:

```python
config = SectorClassifierConfig(device="cuda")  # Force GPU
classifier = get_sector_classifier(config)
```

### Batch Size Tuning

For large batches, adjust batch size:

```python
config = SectorClassifierConfig(batch_size=64)  # Increase batch size
classifier = get_sector_classifier(config)
```

## Limitations

1. **English-focused**: Default model works best with English text
2. **Pre-defined Sectors**: Limited to 11 GICS sectors
3. **Ambiguous Text**: May have low confidence for multi-sector documents
4. **Context-dependent**: Classification based on keywords and semantic similarity

## Integration with SignalForge

The sector classifier integrates seamlessly with other SignalForge NLP modules:

```python
from signalforge.nlp import (
    classify_sector,
    analyze_financial_text,
    extract_entities,
)

text = "Apple reported strong earnings with 20% revenue growth"

# Classify sector
sector_pred = classify_sector(text)
print(f"Sector: {sector_pred.sector}")

# Analyze sentiment
sentiment = analyze_financial_text(text)
print(f"Sentiment: {sentiment.label}")

# Extract entities
entities = extract_entities(text)
print(f"Entities: {[e.text for e in entities.entities]}")
```

## Troubleshooting

### ImportError: No module named 'sentence_transformers'

Install required dependencies:
```bash
pip install sentence-transformers torch
```

### RuntimeError: Model loading failed

Check available models:
```python
from sentence_transformers import SentenceTransformer

# Test model loading
model = SentenceTransformer("all-MiniLM-L6-v2")
```

### Low Confidence Scores

Try these approaches:
1. Use a higher quality model (all-mpnet-base-v2)
2. Lower the similarity threshold
3. Provide more context in the input text
4. Check if text is actually sector-ambiguous

## Contributing

To extend the sector classifier:

1. **Add Sector Descriptions**: Update `SECTOR_DESCRIPTIONS` dictionary
2. **Custom Classifier**: Implement `BaseSectorClassifier` interface
3. **New Models**: Test different sentence-transformer models
4. **Fine-tuning**: Fine-tune models on financial sector data

## References

- [GICS Methodology](https://www.msci.com/gics)
- [Sentence-Transformers Documentation](https://www.sbert.net/)
- [SignalForge NLP Module](/docs/nlp.md)

# SignalForge NLP Embeddings Module

## Overview

The embeddings module provides text embedding capabilities for converting financial text into dense vector representations using sentence-transformers. These embeddings enable semantic similarity computation, semantic search, and other downstream NLP tasks.

## Features

- Multiple pre-trained sentence-transformer models
- Lazy model loading for efficient resource usage
- GPU/CPU/MPS automatic device selection
- Batch processing with configurable batch sizes
- L2 normalization support for similarity tasks
- Cosine similarity computation
- Thread-safe singleton pattern for default configuration
- Strict type checking with mypy
- Comprehensive test coverage

## Architecture

### Core Components

#### 1. Data Classes

**EmbeddingResult**
```python
@dataclass
class EmbeddingResult:
    text: str                # Original input text
    embedding: list[float]   # Dense vector representation
    model_name: str          # Model used to generate embedding
    dimension: int           # Dimensionality of the embedding
```

**EmbeddingsConfig**
```python
@dataclass
class EmbeddingsConfig:
    model_name: str = "all-MiniLM-L6-v2"  # sentence-transformers model
    device: str = "auto"                   # auto, cpu, cuda, mps
    normalize: bool = True                 # L2 normalize embeddings
    batch_size: int = 32                   # Batch size for processing
    max_length: int = 512                  # Maximum sequence length
    cache_model: bool = True               # Cache loaded models
```

#### 2. Abstract Base Class

**BaseEmbeddingModel**
- Defines the interface for all embedding models
- Methods:
  - `encode(text: str) -> EmbeddingResult`
  - `encode_batch(texts: list[str]) -> list[EmbeddingResult]`
  - `dimension: int` (property)
  - `model_name: str` (property)

#### 3. Concrete Implementation

**SentenceTransformerEmbedder**
- Implements BaseEmbeddingModel using sentence-transformers
- Features:
  - Lazy model loading
  - Model caching across instances
  - GPU acceleration support
  - Thread-safe operations
  - Automatic device selection
  - Batch processing optimization

#### 4. Helper Functions

- `get_embedder(config: EmbeddingsConfig | None = None) -> SentenceTransformerEmbedder`
  - Factory function with singleton pattern for default config

- `embed_text(text: str) -> EmbeddingResult`
  - Convenience function for single text embedding

- `embed_texts(texts: list[str]) -> list[EmbeddingResult]`
  - Convenience function for batch embedding

- `compute_similarity(emb1: list[float], emb2: list[float]) -> float`
  - Compute cosine similarity between embeddings

## Supported Models

| Model Name | Dimensions | Description |
|-----------|-----------|-------------|
| all-MiniLM-L6-v2 | 384 | Fast and efficient (default) |
| all-mpnet-base-v2 | 768 | Higher quality, more accurate |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | Multilingual support |

## Usage Examples

### Basic Usage

```python
from signalforge.nlp.embeddings import embed_text

# Generate embedding for a single text
text = "Apple reported strong Q4 earnings, beating analyst expectations."
result = embed_text(text)

print(f"Dimension: {result.dimension}")  # 384
print(f"Model: {result.model_name}")     # all-MiniLM-L6-v2
print(f"Embedding: {result.embedding[:5]}")  # First 5 values
```

### Batch Processing

```python
from signalforge.nlp.embeddings import embed_texts

texts = [
    "Tesla stock surged on delivery numbers.",
    "Amazon reported Q3 earnings miss.",
    "Microsoft announces new AI features.",
]

results = embed_texts(texts)
for result in results:
    print(f"{result.text[:30]}... -> {result.dimension} dims")
```

### Computing Similarity

```python
from signalforge.nlp.embeddings import embed_text, compute_similarity

# Embed two related texts
text1 = "Apple stock rose 5% after earnings beat."
text2 = "AAPL shares jumped on strong results."

emb1 = embed_text(text1)
emb2 = embed_text(text2)

similarity = compute_similarity(emb1.embedding, emb2.embedding)
print(f"Similarity: {similarity:.4f}")  # High similarity (e.g., 0.87)
```

### Custom Configuration

```python
from signalforge.nlp.embeddings import get_embedder, EmbeddingsConfig

# Use higher quality model
config = EmbeddingsConfig(
    model_name="all-mpnet-base-v2",  # 768 dimensions
    device="cuda",                    # Use GPU
    normalize=True,                   # L2 normalize
    batch_size=64,                    # Larger batches
)

embedder = get_embedder(config)
result = embedder.encode("Financial text here...")
print(f"Dimension: {result.dimension}")  # 768
```

### Semantic Search

```python
from signalforge.nlp.embeddings import embed_text, embed_texts, compute_similarity

# Document corpus
documents = [
    "Tesla announces new Gigafactory.",
    "Apple unveils latest iPhone.",
    "Federal Reserve raises rates.",
]

# Search query
query = "What are electric vehicle developments?"

# Generate embeddings
query_emb = embed_text(query)
doc_embs = embed_texts(documents)

# Compute similarities and rank
similarities = [
    (doc, compute_similarity(query_emb.embedding, doc_emb.embedding))
    for doc, doc_emb in zip(documents, doc_embs)
]

# Sort by similarity
similarities.sort(key=lambda x: x[1], reverse=True)

# Top result
print(f"Most relevant: {similarities[0][0]}")  # Tesla document
```

### Multilingual Embeddings

```python
from signalforge.nlp.embeddings import get_embedder, EmbeddingsConfig, compute_similarity

# Use multilingual model
config = EmbeddingsConfig(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
embedder = get_embedder(config)

# Same concept in different languages
texts = [
    "The company reported strong results.",  # English
    "La empresa reporto buenos resultados.",  # Spanish
]

results = embedder.encode_batch(texts)
similarity = compute_similarity(results[0].embedding, results[1].embedding)
print(f"Cross-lingual similarity: {similarity:.4f}")  # High despite different languages
```

## Design Patterns

### Singleton Pattern

The default embedder uses a singleton pattern to avoid loading the same model multiple times:

```python
# Same instance returned for default config
embedder1 = get_embedder()
embedder2 = get_embedder()
assert embedder1 is embedder2  # True

# New instance for custom config
custom_embedder = get_embedder(EmbeddingsConfig(model_name="all-mpnet-base-v2"))
assert embedder1 is not custom_embedder  # True
```

### Model Caching

Models are cached across instances with the same configuration:

```python
# First instance loads the model
config = EmbeddingsConfig(model_name="all-MiniLM-L6-v2")
embedder1 = SentenceTransformerEmbedder(config)
embedder1.encode("Test")

# Second instance reuses cached model
embedder2 = SentenceTransformerEmbedder(config)
embedder2.encode("Test 2")  # No model loading delay
```

### Lazy Loading

Models are loaded only when first needed:

```python
# Model not loaded yet
embedder = SentenceTransformerEmbedder()

# Model loads on first encode call
result = embedder.encode("First text")  # Model loads here

# Subsequent calls use loaded model
result2 = embedder.encode("Second text")  # No loading
```

## Device Selection

The embedder automatically selects the best available device:

```python
# Auto mode (default)
config = EmbeddingsConfig(device="auto")
# Selects: CUDA > MPS > CPU

# Force CPU
config = EmbeddingsConfig(device="cpu")

# Force CUDA (falls back to CPU if unavailable)
config = EmbeddingsConfig(device="cuda")

# Force MPS for Apple Silicon (falls back to CPU if unavailable)
config = EmbeddingsConfig(device="mps")
```

## Error Handling

The module includes comprehensive error handling:

```python
from signalforge.nlp.embeddings import embed_text, compute_similarity

# Empty text
try:
    embed_text("")
except ValueError as e:
    print(f"Error: {e}")  # "Text cannot be empty"

# Mismatched dimensions
emb1 = [0.1, 0.2, 0.3]
emb2 = [0.1, 0.2]
try:
    compute_similarity(emb1, emb2)
except ValueError as e:
    print(f"Error: {e}")  # "must have same dimension"

# Invalid configuration
try:
    EmbeddingsConfig(batch_size=0)
except ValueError as e:
    print(f"Error: {e}")  # "batch_size must be positive"
```

## Performance Considerations

### Normalization

For similarity tasks, L2 normalization is recommended:

```python
config = EmbeddingsConfig(normalize=True)  # Recommended for cosine similarity
```

Normalized embeddings allow faster similarity computation (dot product instead of full cosine calculation).

### Batch Processing

Always prefer batch processing for multiple texts:

```python
# Good - efficient batch processing
results = embed_texts(texts)

# Bad - inefficient sequential processing
results = [embed_text(text) for text in texts]
```

### GPU Acceleration

For large-scale processing, use GPU acceleration:

```python
config = EmbeddingsConfig(
    device="cuda",    # or "mps" for Apple Silicon
    batch_size=64,    # Larger batches on GPU
)
```

### Resource Cleanup

Clean up GPU memory when done:

```python
embedder = SentenceTransformerEmbedder()
# ... use embedder ...
embedder.cleanup()  # Frees GPU memory
```

## Testing

The module includes comprehensive tests with mocked dependencies:

```bash
pytest tests/test_nlp_embeddings.py -v
```

Tests cover:
- Data class validation
- Configuration validation
- Embedding generation
- Batch processing
- Similarity computation
- Error handling
- Device selection
- Model caching
- Thread safety

## Type Safety

The module uses strict mypy type checking:

```bash
mypy src/signalforge/nlp/embeddings.py
```

All functions have proper type annotations and are validated with strict mode.

## Integration with Other Modules

### With Preprocessing

```python
from signalforge.nlp.preprocessing import TextPreprocessor
from signalforge.nlp.embeddings import embed_text

preprocessor = TextPreprocessor()
text = "  Apple Inc. reported $1.5M in revenue.  "
cleaned = preprocessor.clean_text(text)
result = embed_text(cleaned)
```

### With Sentiment Analysis

```python
from signalforge.nlp.embeddings import embed_texts
from signalforge.nlp.sentiment import analyze_financial_text

texts = ["Revenue beat expectations.", "Sales declined."]

# Generate embeddings
embeddings = embed_texts(texts)

# Analyze sentiment
sentiments = [analyze_financial_text(text) for text in texts]

# Combine for downstream analysis
for emb, sent in zip(embeddings, sentiments):
    print(f"Text: {emb.text}")
    print(f"Sentiment: {sent.label} ({sent.confidence:.2f})")
    print(f"Embedding: {emb.dimension} dims")
```

## Best Practices

1. **Use Default Config for Quick Tasks**: The default all-MiniLM-L6-v2 model is fast and sufficient for most tasks.

2. **Use Batch Processing**: Always process multiple texts in batches for better performance.

3. **Enable Normalization for Similarity**: Set `normalize=True` when computing similarities.

4. **Cache Models**: Keep `cache_model=True` (default) to avoid reloading models.

5. **Choose Appropriate Model**:
   - Fast tasks: all-MiniLM-L6-v2 (384 dims)
   - High quality: all-mpnet-base-v2 (768 dims)
   - Multilingual: paraphrase-multilingual-MiniLM-L12-v2

6. **Clean Up Resources**: Call `cleanup()` when done with large-scale processing to free GPU memory.

7. **Handle Empty Texts**: Always validate input texts before embedding.

8. **Use Type Hints**: Leverage type hints for better IDE support and error detection.

## Dependencies

- sentence-transformers >= 2.2.0
- torch >= 2.1.0
- numpy (via sentence-transformers)

## License

Apache-2.0

## See Also

- [Preprocessing Module](./nlp_preprocessing.md)
- [Sentiment Analysis Module](./nlp_sentiment.md)
- [Named Entity Recognition Module](./nlp_ner.md)
- [sentence-transformers Documentation](https://www.sbert.net/)

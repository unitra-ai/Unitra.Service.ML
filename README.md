# Unitra Service ML

Modal.com ML services for Unitra translation platform.

## Features

- **MADLAD-400-3B Model**: High-quality translation for 400+ languages
- **A10G GPU**: Optimized for cost/performance balance
- **Persistent Volume**: Model caching for fast cold starts
- **Batch Translation**: Process up to 16 texts per request
- **Concurrent Requests**: Handles 16 concurrent inputs

## Quick Start

```bash
# Install dependencies
poetry install

# Configure Modal
modal token new

# Download model (optional, speeds up first cold start)
modal run scripts/download_model.py

# Deploy to Modal
./scripts/deploy.sh
```

## API Reference

### POST /translate

Translate single text or batch of texts.

**Single Translation Request**:
```json
{
  "text": "Hello, world!",
  "source_lang": "en",
  "target_lang": "zh"
}
```

**Single Translation Response**:
```json
{
  "translation": "你好，世界！",
  "source_lang": "en",
  "target_lang": "zh",
  "tokens_used": 15,
  "latency_ms": 150.5,
  "processing_mode": "cloud"
}
```

**Batch Translation Request**:
```json
{
  "texts": ["Hello", "World"],
  "source_lang": "en",
  "target_lang": "zh"
}
```

**Batch Translation Response**:
```json
{
  "translations": ["你好", "世界"],
  "source_lang": "en",
  "target_lang": "zh",
  "total_tokens": 20,
  "latency_ms": 250.0,
  "processing_mode": "cloud"
}
```

### GET /health

Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "model_id": "google/madlad400-3b-mt",
  "model_loaded": true,
  "gpu_available": true,
  "gpu_memory_used_gb": 2.5,
  "gpu_memory_total_gb": 24.0,
  "warm": true
}
```

## Client Usage

### Python Client

```python
from src.client import MTClient

async with MTClient() as client:
    # Single translation
    result = await client.translate("Hello", "en", "zh")
    print(result.translation)  # 你好

    # Batch translation
    batch = await client.translate_batch(
        ["Hello", "World"],
        "en",
        "zh"
    )
    print(batch.translations)  # ["你好", "世界"]

    # Health check
    health = await client.health_check()
    print(health.status)  # healthy
```

### cURL

```bash
# Single translation
curl -X POST https://unitra--unitra-mt-translate.modal.run \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello", "source_lang": "en", "target_lang": "zh"}'

# Health check
curl https://unitra--unitra-mt-health.modal.run
```

## Supported Languages

### Priority Languages (20)
- English (en), Chinese (zh), Japanese (ja), Korean (ko)
- Spanish (es), French (fr), German (de), Russian (ru)
- Portuguese (pt), Arabic (ar), Vietnamese (vi), Thai (th)
- Indonesian (id), Turkish (tr), Polish (pl), Italian (it)
- Dutch (nl), Swedish (sv), Czech (cs), Ukrainian (uk)

### Extended Languages (55+)
See `src/config/languages.py` for complete list including:
- South Asian: Hindi, Bengali, Tamil, Telugu, etc.
- European: Greek, Hungarian, Romanian, etc.
- Middle Eastern: Hebrew, Persian, Urdu
- African: Swahili, Amharic, Hausa, etc.

## Development

### Run Tests

```bash
# Unit tests
poetry run pytest tests/ -v

# Integration tests (requires deployment)
RUN_INTEGRATION_TESTS=1 poetry run pytest tests/test_integration.py -v
```

### Code Quality

```bash
# Format
poetry run black src/ tests/

# Lint
poetry run ruff check src/ tests/ --fix

# Type check
poetry run mypy src/
```

### Local Testing

```bash
# Run service locally
modal run src/services/mt_service.py

# Test specific functions
modal run scripts/download_model.py --action verify
```

## Configuration

### MTSettings (`src/config/settings.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `max_input_length` | 512 | Max characters per text |
| `max_output_length` | 256 | Max output tokens |
| `max_batch_size` | 16 | Max texts per batch |
| `container_idle_timeout` | 300 | Container keepalive (seconds) |
| `concurrent_inputs` | 16 | Max concurrent requests |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MT_SERVICE_URL` | Custom service URL for testing |
| `RUN_INTEGRATION_TESTS` | Enable integration tests |

## Architecture

```
src/
├── services/
│   └── mt_service.py    # Modal service definition
├── config/
│   ├── settings.py      # Service configuration
│   └── languages.py     # Language validation
└── client/
    └── mt_client.py     # HTTP client for API calls

scripts/
├── download_model.py    # Model download script
└── deploy.sh            # Deployment script
```

## Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Cold start | <30s | With cached model |
| Warm latency | <500ms | Per sentence |
| GPU | A10G | $1.10/hour |
| VRAM | ~8GB | Model loaded |

## Cost Estimation

| Scenario | Est. Cost |
|----------|-----------|
| Low usage (1K/day) | ~$5/month |
| Medium (10K/day) | ~$30/month |
| High (100K/day) | ~$200/month |

*Based on Modal.com pricing and 5-minute container keepalive*

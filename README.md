# Unitra Service ML

Modal.com ML services for Unitra translation platform.

## Setup

```bash
# Install dependencies
poetry install

# Configure Modal
modal token new

# Deploy
modal deploy src/services/mt_service.py
```

## Services

### MTService

Machine translation service using MADLAD-400-3B model.

**Endpoint**: `/translate`

**Request**:
```json
{
  "text": "Hello, world!",
  "source_lang": "en",
  "target_lang": "zh"
}
```

**Response**:
```json
{
  "translated_text": "你好，世界！",
  "source_lang": "en",
  "target_lang": "zh",
  "latency_ms": 150.5
}
```

## Development

```bash
# Run locally (for testing)
modal run src/services/mt_service.py

# Deploy to Modal
modal deploy src/services/mt_service.py

# View logs
modal app logs unitra-ml
```

## Model

- **Model**: google/madlad400-3b-mt
- **Languages**: 400+ languages
- **Optimizations**: FP16, ONNX (planned)

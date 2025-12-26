"""Self-hosted ML Translation Service.

A standalone FastAPI service for machine translation using MADLAD-400-3B.
Can be deployed to any GPU-enabled server (Coolify, Kubernetes, Docker, etc.)

Usage:
    python -m selfhosted.main

Environment Variables:
    MODEL_CACHE_DIR: Model cache directory (default: /models)
    HUGGINGFACE_TOKEN: Optional HuggingFace token
    PORT: Service port (default: 8001)
    HOST: Service host (default: 0.0.0.0)
    WORKERS: Number of workers (default: 1)
    LOG_LEVEL: Logging level (default: info)
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Configuration from environment
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/models")
PORT = int(os.getenv("PORT", "8001"))
HOST = os.getenv("HOST", "0.0.0.0")
WORKERS = int(os.getenv("WORKERS", "1"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

# Model configuration
MODEL_ID = "google/madlad400-3b-mt"
MODEL_DIR = os.path.join(MODEL_CACHE_DIR, "madlad-400-3b-mt")
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_TOKENS = 256
MAX_BATCH_SIZE = 16

# Scaledown configuration (for serverless platforms)
IDLE_TIMEOUT_SECONDS = 30  # Container idle timeout

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Language configuration
PRIORITY_LANGUAGES = {
    "en", "zh", "ja", "ko", "es", "fr", "de", "ru", "pt", "ar",
    "vi", "th", "id", "tr", "pl", "it", "nl", "sv", "cs", "uk",
    "hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "my", "km",
    "lo", "fil", "ms", "el", "hu", "ro", "bg", "sk", "hr", "sl",
    "sr", "lt", "lv", "et", "fi", "da", "no", "he", "fa", "ur",
    "sw", "am", "ha", "yo", "zu",
}

LANGUAGE_ALIASES = {
    "zh-cn": "zh", "zh-tw": "zh", "zh-hans": "zh", "zh-hant": "zh",
    "pt-br": "pt", "pt-pt": "pt", "en-us": "en", "en-gb": "en",
}


def normalize_language(code: str) -> str:
    """Normalize language code to standard format."""
    code = code.lower().strip()
    if code in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[code]
    if "-" in code:
        code = code.split("-")[0]
    return code


def validate_language(code: str, strict: bool = False) -> str:
    """Validate and normalize a language code."""
    if not code:
        raise ValueError("Language code cannot be empty")
    normalized = normalize_language(code)

    if not (2 <= len(normalized) <= 3) or not normalized.isalpha():
        raise ValueError(f"Invalid language code format: {code}")

    if strict and normalized not in PRIORITY_LANGUAGES:
        raise ValueError(f"Unsupported language: {code}")

    return normalized


# Pydantic models
class TranslateRequest(BaseModel):
    """Single translation request."""
    text: str = Field(..., max_length=512, description="Text to translate")
    source_lang: str = Field(default="en", description="Source language code")
    target_lang: str = Field(..., description="Target language code")


class BatchTranslateRequest(BaseModel):
    """Batch translation request."""
    texts: list[str] = Field(..., max_length=16, description="Texts to translate")
    source_lang: str = Field(default="en", description="Source language code")
    target_lang: str = Field(..., description="Target language code")


class TranslateResponse(BaseModel):
    """Translation response."""
    translation: str
    source_lang: str
    target_lang: str
    tokens_used: int
    latency_ms: float
    processing_mode: str = "self-hosted"


class BatchTranslateResponse(BaseModel):
    """Batch translation response."""
    translations: list[str]
    source_lang: str
    target_lang: str
    total_tokens: int
    latency_ms: float
    processing_mode: str = "self-hosted"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_id: str
    model_loaded: bool
    gpu_available: bool
    gpu_memory_used_gb: float | None
    gpu_memory_total_gb: float | None
    warm: bool
    version: str


# Global model instance
@dataclass
class ModelState:
    """Global model state."""
    tokenizer: Any = None
    model: Any = None
    gpu_memory_total: float | None = None
    loaded: bool = False
    last_request_time: float = 0.0


model_state = ModelState()


def load_model() -> None:
    """Load the translation model."""
    global model_state

    start_time = time.perf_counter()
    logger.info(f"Loading model: {MODEL_ID}")

    # Check if model is cached
    if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
        logger.info(f"Loading from cache: {MODEL_DIR}")
        load_path = MODEL_DIR
    else:
        logger.info(f"Downloading from HuggingFace: {MODEL_ID}")
        load_path = MODEL_ID

    # Load tokenizer and model
    model_state.tokenizer = AutoTokenizer.from_pretrained(load_path)
    model_state.model = AutoModelForSeq2SeqLM.from_pretrained(
        load_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model_state.model.eval()

    # Cache model if downloaded
    if load_path == MODEL_ID:
        logger.info(f"Caching model to: {MODEL_DIR}")
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_state.tokenizer.save_pretrained(MODEL_DIR)
        model_state.model.save_pretrained(MODEL_DIR)

    # Warm up
    warm_up()

    # Store GPU info
    if torch.cuda.is_available():
        model_state.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9

    model_state.loaded = True
    load_time = time.perf_counter() - start_time
    logger.info(f"Model loaded in {load_time:.2f}s")


def warm_up() -> None:
    """Warm up the model with a dummy translation."""
    logger.info("Warming up model...")
    dummy_input = "<2en> 测试"
    inputs = model_state.tokenizer(
        dummy_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32,
    ).to(model_state.model.device)

    with torch.no_grad():
        model_state.model.generate(**inputs, max_new_tokens=16)

    logger.info("Model warmed up successfully")


def translate_single(text: str, source_lang: str, target_lang: str) -> dict[str, Any]:
    """Translate a single text."""
    start_time = time.perf_counter()
    model_state.last_request_time = time.time()

    # Validate inputs
    if not text or not text.strip():
        return {
            "translation": "",
            "source_lang": source_lang,
            "target_lang": target_lang,
            "tokens_used": 0,
            "latency_ms": 0.0,
            "processing_mode": "self-hosted",
        }

    if len(text) > MAX_INPUT_LENGTH:
        raise ValueError(f"Text too long: {len(text)} chars (max {MAX_INPUT_LENGTH})")

    source_lang = validate_language(source_lang)
    target_lang = validate_language(target_lang)

    # Prepare input with MADLAD-400 language tag
    input_text = f"<2{target_lang}> {text}"

    inputs = model_state.tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_OUTPUT_TOKENS,
    ).to(model_state.model.device)

    input_tokens = inputs["input_ids"].shape[1]

    # Generate translation
    with torch.no_grad():
        outputs = model_state.model.generate(
            **inputs,
            max_new_tokens=MAX_OUTPUT_TOKENS,
            num_beams=4,
            early_stopping=True,
        )

    output_tokens = outputs.shape[1]
    translation = model_state.tokenizer.decode(outputs[0], skip_special_tokens=True)

    latency_ms = (time.perf_counter() - start_time) * 1000

    return {
        "translation": translation,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "tokens_used": input_tokens + output_tokens,
        "latency_ms": round(latency_ms, 2),
        "processing_mode": "self-hosted",
    }


def translate_batch(texts: list[str], source_lang: str, target_lang: str) -> dict[str, Any]:
    """Translate a batch of texts."""
    start_time = time.perf_counter()
    model_state.last_request_time = time.time()

    # Validate batch size
    if not texts:
        return {
            "translations": [],
            "source_lang": source_lang,
            "target_lang": target_lang,
            "total_tokens": 0,
            "latency_ms": 0.0,
            "processing_mode": "self-hosted",
        }

    if len(texts) > MAX_BATCH_SIZE:
        raise ValueError(f"Batch too large: {len(texts)} texts (max {MAX_BATCH_SIZE})")

    # Validate languages
    source_lang = validate_language(source_lang)
    target_lang = validate_language(target_lang)

    # Validate and prepare inputs
    input_texts = []
    for text in texts:
        if len(text) > MAX_INPUT_LENGTH:
            raise ValueError(f"Text too long: {len(text)} chars (max {MAX_INPUT_LENGTH})")
        if text.strip():
            input_texts.append(f"<2{target_lang}> {text}")
        else:
            input_texts.append(f"<2{target_lang}> ")

    # Tokenize batch
    inputs = model_state.tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_OUTPUT_TOKENS,
    ).to(model_state.model.device)

    input_tokens = inputs["input_ids"].numel()

    # Generate translations
    with torch.no_grad():
        outputs = model_state.model.generate(
            **inputs,
            max_new_tokens=MAX_OUTPUT_TOKENS,
            num_beams=4,
            early_stopping=True,
        )

    output_tokens = outputs.numel()

    # Decode translations
    translations = model_state.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    latency_ms = (time.perf_counter() - start_time) * 1000

    return {
        "translations": translations,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "total_tokens": input_tokens + output_tokens,
        "latency_ms": round(latency_ms, 2),
        "processing_mode": "self-hosted",
    }


# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup: Load model
    logger.info("Starting ML service...")
    load_model()
    yield
    # Shutdown
    logger.info("Shutting down ML service...")


app = FastAPI(
    title="Unitra ML Translation Service",
    description="Self-hosted machine translation using MADLAD-400-3B",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    gpu_memory_used = None
    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.memory_allocated() / 1e9

    return HealthResponse(
        status="healthy" if model_state.loaded else "unhealthy",
        model_id=MODEL_ID,
        model_loaded=model_state.loaded,
        gpu_available=torch.cuda.is_available(),
        gpu_memory_used_gb=round(gpu_memory_used, 2) if gpu_memory_used else None,
        gpu_memory_total_gb=round(model_state.gpu_memory_total, 2) if model_state.gpu_memory_total else None,
        warm=model_state.loaded,
        version="0.1.0",
    )


@app.post("/translate", response_model=TranslateResponse)
async def translate_endpoint(request: TranslateRequest) -> TranslateResponse:
    """Single text translation endpoint."""
    if not model_state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = translate_single(
            text=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
        )
        return TranslateResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Translation error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/", response_model=TranslateResponse | BatchTranslateResponse)
async def root_translate(request: dict[str, Any]) -> TranslateResponse | BatchTranslateResponse:
    """Root translation endpoint (compatible with Modal API).

    Accepts both single and batch requests based on request body.
    """
    if not model_state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        if "texts" in request:
            # Batch request
            result = translate_batch(
                texts=request["texts"],
                source_lang=request.get("source_lang", "en"),
                target_lang=request["target_lang"],
            )
            return BatchTranslateResponse(**result)
        else:
            # Single request
            result = translate_single(
                text=request["text"],
                source_lang=request.get("source_lang", "en"),
                target_lang=request["target_lang"],
            )
            return TranslateResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Translation error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", response_model=BatchTranslateResponse)
async def batch_translate_endpoint(request: BatchTranslateRequest) -> BatchTranslateResponse:
    """Batch translation endpoint."""
    if not model_state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = translate_batch(
            texts=request.texts,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
        )
        return BatchTranslateResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Batch translation error")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the service."""
    logger.info(f"Starting server on {HOST}:{PORT}")
    uvicorn.run(
        "selfhosted.main:app",
        host=HOST,
        port=PORT,
        workers=WORKERS,
        log_level=LOG_LEVEL,
    )


if __name__ == "__main__":
    main()

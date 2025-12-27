"""Machine Translation Service using MADLAD-400-3B on Modal.com.

This service provides high-quality machine translation for 400+ languages
using the MADLAD-400-3B model deployed on Modal's serverless GPU infrastructure.

Features:
- Single and batch translation
- Language validation
- Model caching via persistent volume
- Health monitoring
- Concurrent request handling
- API Key authentication (only API service can access)

Security:
- All endpoints require X-API-Key header
- API key is stored in Modal secrets
- Only the API service should know the key
"""

import os
import time
from dataclasses import dataclass
from typing import Any

import modal
from fastapi import Header, HTTPException

# Modal app configuration
app = modal.App("unitra-mt")

# API Key secret for authentication
api_key_secret = modal.Secret.from_name("unitra-api-key", required_keys=["API_KEY"])

# Persistent volume for model caching
volume = modal.Volume.from_name("unitra-models", create_if_missing=True)

# Container image with ML dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2",
        "torch==2.1.0",
        "transformers==4.36.0",
        "sentencepiece",
        "accelerate",
        "safetensors",
        "optimum",
        "fastapi[standard]",
    )
    .env({"HF_HOME": "/models/huggingface"})
)


@dataclass
class TranslationResult:
    """Result of a translation operation."""

    translation: str
    source_lang: str
    target_lang: str
    tokens_used: int
    latency_ms: float
    processing_mode: str = "cloud"


@dataclass
class BatchTranslationResult:
    """Result of a batch translation operation."""

    translations: list[str]
    source_lang: str
    target_lang: str
    total_tokens: int
    latency_ms: float
    processing_mode: str = "cloud"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    status: str
    model_id: str
    model_loaded: bool
    gpu_available: bool
    gpu_memory_used_gb: float | None
    gpu_memory_total_gb: float | None
    warm: bool


# Priority languages (commonly used, validated)
PRIORITY_LANGUAGES = {
    "en", "zh", "ja", "ko", "es", "fr", "de", "ru", "pt", "ar",
    "vi", "th", "id", "tr", "pl", "it", "nl", "sv", "cs", "uk",
    "hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "my", "km",
    "lo", "fil", "ms", "el", "hu", "ro", "bg", "sk", "hr", "sl",
    "sr", "lt", "lv", "et", "fi", "da", "no", "he", "fa", "ur",
    "sw", "am", "ha", "yo", "zu",
}

# Language code aliases
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
    """Validate and normalize a language code.

    MADLAD-400 supports 400+ languages. By default, we accept any valid
    2-3 letter language code. Set strict=True to only allow priority languages.

    Args:
        code: Language code to validate
        strict: If True, only allow priority languages

    Returns:
        Normalized language code

    Raises:
        ValueError: If code is empty or (in strict mode) unsupported
    """
    if not code:
        raise ValueError("Language code cannot be empty")
    normalized = normalize_language(code)

    # Basic format validation (2-3 letter codes)
    if not (2 <= len(normalized) <= 3) or not normalized.isalpha():
        raise ValueError(f"Invalid language code format: {code}")

    # In strict mode, only allow priority languages
    if strict and normalized not in PRIORITY_LANGUAGES:
        raise ValueError(f"Unsupported language: {code}")

    return normalized


@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/models": volume},
    container_idle_timeout=30,  # 30 seconds - quick shutdown to save costs
    retries=2,
    timeout=120,
    allow_concurrent_inputs=16,
)
class MTService:
    """Machine translation service using MADLAD-400-3B model.

    This class is deployed as a Modal container class with GPU acceleration.
    The model is cached in a persistent volume for fast cold starts.

    Attributes:
        MODEL_ID: HuggingFace model identifier
        MODEL_DIR: Local directory name for cached model
        MAX_INPUT_LENGTH: Maximum input text length (characters)
        MAX_OUTPUT_TOKENS: Maximum output tokens
        MAX_BATCH_SIZE: Maximum texts in a batch
    """

    MODEL_ID = "google/madlad400-3b-mt"
    MODEL_DIR = "/models/madlad-400-3b-mt"
    MAX_INPUT_LENGTH = 512
    MAX_OUTPUT_TOKENS = 256
    MAX_BATCH_SIZE = 16

    @modal.enter()
    def load_model(self) -> None:
        """Load the model when container starts.

        This method is called once when the container is created.
        It loads the model from the persistent volume if available,
        otherwise downloads it from HuggingFace.
        """
        import os

        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        start_time = time.perf_counter()

        # Check if model is cached
        model_path = self.MODEL_DIR
        if os.path.exists(model_path) and os.listdir(model_path):
            print(f"Loading model from cache: {model_path}")
            load_path = model_path
        else:
            print(f"Downloading model: {self.MODEL_ID}")
            load_path = self.MODEL_ID

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            load_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

        # Cache model if downloaded from HuggingFace
        if load_path == self.MODEL_ID:
            print(f"Caching model to: {model_path}")
            os.makedirs(model_path, exist_ok=True)
            self.tokenizer.save_pretrained(model_path)
            self.model.save_pretrained(model_path)
            volume.commit()

        # Warm up with a dummy translation
        self._warm_up()

        load_time = time.perf_counter() - start_time
        print(f"Model loaded in {load_time:.2f}s")

        # Store GPU info
        if torch.cuda.is_available():
            self.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            self.gpu_memory_total = None

    def _warm_up(self) -> None:
        """Warm up the model with a dummy translation."""
        import torch

        dummy_input = "<2en> 测试"
        inputs = self.tokenizer(
            dummy_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32,
        ).to(self.model.device)

        with torch.no_grad():
            self.model.generate(**inputs, max_new_tokens=16)

        print("Model warmed up successfully")

    @modal.method()
    def translate_single(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> dict[str, Any]:
        """Translate a single text from source to target language.

        Args:
            text: Text to translate (max 512 characters)
            source_lang: Source language code (ISO 639-1)
            target_lang: Target language code (ISO 639-1)

        Returns:
            Dictionary containing:
                - translation: Translated text
                - source_lang: Normalized source language code
                - target_lang: Normalized target language code
                - tokens_used: Approximate token count
                - latency_ms: Processing time in milliseconds
                - processing_mode: Always "cloud"

        Raises:
            ValueError: If text is too long or language is unsupported
        """
        import torch

        start_time = time.perf_counter()

        # Validate inputs
        if not text or not text.strip():
            return {
                "translation": "",
                "source_lang": source_lang,
                "target_lang": target_lang,
                "tokens_used": 0,
                "latency_ms": 0.0,
                "processing_mode": "cloud",
            }

        if len(text) > self.MAX_INPUT_LENGTH:
            raise ValueError(
                f"Text too long: {len(text)} chars (max {self.MAX_INPUT_LENGTH})"
            )

        source_lang = validate_language(source_lang)
        target_lang = validate_language(target_lang)

        # Prepare input with MADLAD-400 language tag
        input_text = f"<2{target_lang}> {text}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.MAX_OUTPUT_TOKENS,
        ).to(self.model.device)

        input_tokens = inputs["input_ids"].shape[1]

        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.MAX_OUTPUT_TOKENS,
                num_beams=4,
                early_stopping=True,
            )

        output_tokens = outputs.shape[1]
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "translation": translation,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "tokens_used": input_tokens + output_tokens,
            "latency_ms": round(latency_ms, 2),
            "processing_mode": "cloud",
        }

    @modal.method()
    def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> dict[str, Any]:
        """Translate a batch of texts from source to target language.

        Args:
            texts: List of texts to translate (max 16 texts, each max 512 chars)
            source_lang: Source language code (ISO 639-1)
            target_lang: Target language code (ISO 639-1)

        Returns:
            Dictionary containing:
                - translations: List of translated texts
                - source_lang: Normalized source language code
                - target_lang: Normalized target language code
                - total_tokens: Total tokens used
                - latency_ms: Processing time in milliseconds
                - processing_mode: Always "cloud"

        Raises:
            ValueError: If batch is too large or text is too long
        """
        import torch

        start_time = time.perf_counter()

        # Validate batch size
        if not texts:
            return {
                "translations": [],
                "source_lang": source_lang,
                "target_lang": target_lang,
                "total_tokens": 0,
                "latency_ms": 0.0,
                "processing_mode": "cloud",
            }

        if len(texts) > self.MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch too large: {len(texts)} texts (max {self.MAX_BATCH_SIZE})"
            )

        # Validate and normalize languages
        source_lang = validate_language(source_lang)
        target_lang = validate_language(target_lang)

        # Validate and prepare inputs
        input_texts = []
        for text in texts:
            if len(text) > self.MAX_INPUT_LENGTH:
                raise ValueError(
                    f"Text too long: {len(text)} chars (max {self.MAX_INPUT_LENGTH})"
                )
            if text.strip():
                input_texts.append(f"<2{target_lang}> {text}")
            else:
                input_texts.append(f"<2{target_lang}> ")

        # Tokenize batch
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.MAX_OUTPUT_TOKENS,
        ).to(self.model.device)

        input_tokens = inputs["input_ids"].numel()

        # Generate translations
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.MAX_OUTPUT_TOKENS,
                num_beams=4,
                early_stopping=True,
            )

        output_tokens = outputs.numel()

        # Decode translations
        translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Handle empty inputs
        result_translations = []
        text_idx = 0
        for text in texts:
            if text.strip():
                result_translations.append(translations[text_idx])
                text_idx += 1
            else:
                result_translations.append("")

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "translations": result_translations,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "total_tokens": input_tokens + output_tokens,
            "latency_ms": round(latency_ms, 2),
            "processing_mode": "cloud",
        }

    @modal.method()
    def health_check(self) -> dict[str, Any]:
        """Perform a health check on the service.

        Returns:
            Dictionary containing:
                - status: "healthy" or "unhealthy"
                - model_id: Model identifier
                - model_loaded: Whether model is loaded
                - gpu_available: Whether GPU is available
                - gpu_memory_used_gb: GPU memory used (if available)
                - gpu_memory_total_gb: Total GPU memory (if available)
                - warm: Whether model has been warmed up
        """
        import torch

        try:
            # Check if model is loaded
            model_loaded = hasattr(self, "model") and self.model is not None

            # Check GPU
            gpu_available = torch.cuda.is_available()
            gpu_memory_used = None
            gpu_memory_total = None

            if gpu_available:
                gpu_memory_used = torch.cuda.memory_allocated() / 1e9
                gpu_memory_total = getattr(self, "gpu_memory_total", None)

            # Test with a quick translation
            warm = False
            if model_loaded:
                try:
                    self.translate_single("test", "en", "zh")
                    warm = True
                except Exception:
                    pass

            status = "healthy" if model_loaded and gpu_available else "unhealthy"

            return {
                "status": status,
                "model_id": self.MODEL_ID,
                "model_loaded": model_loaded,
                "gpu_available": gpu_available,
                "gpu_memory_used_gb": round(gpu_memory_used, 2) if gpu_memory_used else None,
                "gpu_memory_total_gb": round(gpu_memory_total, 2) if gpu_memory_total else None,
                "warm": warm,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "model_id": self.MODEL_ID,
                "model_loaded": False,
                "gpu_available": False,
                "gpu_memory_used_gb": None,
                "gpu_memory_total_gb": None,
                "warm": False,
                "error": str(e),
            }


# Authentication helper
def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")) -> str:
    """Verify API key from header.

    Args:
        x_api_key: API key from X-API-Key header

    Returns:
        The verified API key

    Raises:
        HTTPException: If API key is missing or invalid
    """
    expected_key = os.environ.get("API_KEY")
    if not expected_key:
        # If no API key is configured, allow access (development mode)
        return "development"

    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    if x_api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return x_api_key


# HTTP Web Endpoint for external access (authenticated)
@app.function(image=image, secrets=[api_key_secret])
@modal.fastapi_endpoint(method="POST", docs=True)
def translate(
    request: dict[str, Any],
    x_api_key: str = Header(None, alias="X-API-Key"),
) -> dict[str, Any]:
    """HTTP endpoint for translation.

    Requires X-API-Key header for authentication.

    Request headers:
        X-API-Key: Your API key

    Request body:
        {
            "text": "Hello, world!",
            "source_lang": "en",
            "target_lang": "zh"
        }

    Or for batch:
        {
            "texts": ["Hello", "World"],
            "source_lang": "en",
            "target_lang": "zh"
        }

    Returns:
        Translation result with metadata
    """
    # Verify API key
    verify_api_key(x_api_key)

    service = MTService()

    # Check if batch request
    if "texts" in request:
        return service.translate_batch.remote(
            texts=request["texts"],
            source_lang=request.get("source_lang", "en"),
            target_lang=request["target_lang"],
        )
    else:
        return service.translate_single.remote(
            text=request["text"],
            source_lang=request.get("source_lang", "en"),
            target_lang=request["target_lang"],
        )


@app.function(image=image, secrets=[api_key_secret])
@modal.fastapi_endpoint(method="GET", docs=True)
def health(x_api_key: str = Header(None, alias="X-API-Key")) -> dict[str, Any]:
    """HTTP endpoint for health check.

    Requires X-API-Key header for authentication.

    Request headers:
        X-API-Key: Your API key

    Returns:
        Health status of the service
    """
    # Verify API key
    verify_api_key(x_api_key)

    service = MTService()
    return service.health_check.remote()


# Local entry point for testing
@app.local_entrypoint()
def main():
    """Local testing entry point."""
    print("Testing MT Service...")

    service = MTService()

    # Test single translation
    print("\n--- Single Translation Test ---")
    result = service.translate_single.remote(
        text="Hello, how are you today?",
        source_lang="en",
        target_lang="zh",
    )
    print(f"Input: Hello, how are you today?")
    print(f"Output: {result['translation']}")
    print(f"Latency: {result['latency_ms']}ms")
    print(f"Tokens: {result['tokens_used']}")

    # Test batch translation
    print("\n--- Batch Translation Test ---")
    batch_result = service.translate_batch.remote(
        texts=["Good morning", "Good afternoon", "Good evening"],
        source_lang="en",
        target_lang="zh",
    )
    for i, (src, tgt) in enumerate(zip(
        ["Good morning", "Good afternoon", "Good evening"],
        batch_result["translations"],
    )):
        print(f"  {src} -> {tgt}")
    print(f"Total latency: {batch_result['latency_ms']}ms")
    print(f"Total tokens: {batch_result['total_tokens']}")

    # Test health check
    print("\n--- Health Check ---")
    health_result = service.health_check.remote()
    print(f"Status: {health_result['status']}")
    print(f"Model: {health_result['model_id']}")
    print(f"GPU Memory: {health_result.get('gpu_memory_used_gb', 'N/A')} GB used")

    print("\nAll tests completed!")

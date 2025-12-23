"""HTTP client for calling the Modal MT service.

This module provides a clean interface for FastAPI applications
to call the Modal-hosted machine translation service.

Usage:
    from src.client import MTClient

    async with MTClient() as client:
        result = await client.translate("Hello", "en", "zh")
        print(result.translation)
"""

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx


class MTClientError(Exception):
    """Error from MT client operations."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response: dict[str, Any] | None = None,
    ):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(message)


@dataclass
class TranslationResponse:
    """Response from single translation."""

    translation: str
    source_lang: str
    target_lang: str
    tokens_used: int
    latency_ms: float
    processing_mode: str


@dataclass
class BatchTranslationResponse:
    """Response from batch translation."""

    translations: list[str]
    source_lang: str
    target_lang: str
    total_tokens: int
    latency_ms: float
    processing_mode: str


@dataclass
class HealthResponse:
    """Response from health check."""

    status: str
    model_id: str
    model_loaded: bool
    gpu_available: bool
    gpu_memory_used_gb: float | None
    gpu_memory_total_gb: float | None
    warm: bool


class MTClient:
    """HTTP client for Modal MT service.

    This client provides async methods for calling the Modal-hosted
    machine translation service.

    Attributes:
        base_url: Base URL of the Modal web endpoint
        timeout: Request timeout in seconds

    Example:
        async with MTClient() as client:
            # Single translation
            result = await client.translate("Hello", "en", "zh")
            print(result.translation)

            # Batch translation
            batch = await client.translate_batch(
                ["Hello", "World"],
                "en",
                "zh"
            )
            print(batch.translations)

            # Health check
            health = await client.health_check()
            print(health.status)
    """

    # Default Modal web endpoint URL (update after deployment)
    DEFAULT_BASE_URL = "https://unitra--unitra-mt.modal.run"

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize the MT client.

        Args:
            base_url: Base URL of the Modal web endpoint.
                      If None, uses DEFAULT_BASE_URL.
            timeout: Request timeout in seconds.
        """
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "MTClient":
        """Enter async context manager."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get the HTTP client, ensuring it's initialized."""
        if self._client is None:
            raise RuntimeError(
                "MTClient must be used as an async context manager: "
                "async with MTClient() as client: ..."
            )
        return self._client

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResponse:
        """Translate a single text.

        Args:
            text: Text to translate (max 512 characters)
            source_lang: Source language code (e.g., "en")
            target_lang: Target language code (e.g., "zh")

        Returns:
            TranslationResponse with translation and metadata

        Raises:
            MTClientError: If translation fails
        """
        try:
            response = await self.client.post(
                "/translate",
                json={
                    "text": text,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                },
            )
            response.raise_for_status()
            data = response.json()

            return TranslationResponse(
                translation=data["translation"],
                source_lang=data["source_lang"],
                target_lang=data["target_lang"],
                tokens_used=data["tokens_used"],
                latency_ms=data["latency_ms"],
                processing_mode=data["processing_mode"],
            )

        except httpx.HTTPStatusError as e:
            raise MTClientError(
                f"Translation failed: {e.response.status_code}",
                status_code=e.response.status_code,
                response=e.response.json() if e.response.content else None,
            ) from e
        except httpx.RequestError as e:
            raise MTClientError(f"Request failed: {e}") from e

    async def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> BatchTranslationResponse:
        """Translate a batch of texts.

        Args:
            texts: List of texts to translate (max 16 texts, each max 512 chars)
            source_lang: Source language code (e.g., "en")
            target_lang: Target language code (e.g., "zh")

        Returns:
            BatchTranslationResponse with translations and metadata

        Raises:
            MTClientError: If translation fails
        """
        try:
            response = await self.client.post(
                "/translate",
                json={
                    "texts": texts,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                },
            )
            response.raise_for_status()
            data = response.json()

            return BatchTranslationResponse(
                translations=data["translations"],
                source_lang=data["source_lang"],
                target_lang=data["target_lang"],
                total_tokens=data["total_tokens"],
                latency_ms=data["latency_ms"],
                processing_mode=data["processing_mode"],
            )

        except httpx.HTTPStatusError as e:
            raise MTClientError(
                f"Batch translation failed: {e.response.status_code}",
                status_code=e.response.status_code,
                response=e.response.json() if e.response.content else None,
            ) from e
        except httpx.RequestError as e:
            raise MTClientError(f"Request failed: {e}") from e

    async def health_check(self) -> HealthResponse:
        """Check health of the MT service.

        Returns:
            HealthResponse with service status

        Raises:
            MTClientError: If health check fails
        """
        try:
            response = await self.client.get("/health")
            response.raise_for_status()
            data = response.json()

            return HealthResponse(
                status=data["status"],
                model_id=data["model_id"],
                model_loaded=data["model_loaded"],
                gpu_available=data["gpu_available"],
                gpu_memory_used_gb=data.get("gpu_memory_used_gb"),
                gpu_memory_total_gb=data.get("gpu_memory_total_gb"),
                warm=data["warm"],
            )

        except httpx.HTTPStatusError as e:
            raise MTClientError(
                f"Health check failed: {e.response.status_code}",
                status_code=e.response.status_code,
                response=e.response.json() if e.response.content else None,
            ) from e
        except httpx.RequestError as e:
            raise MTClientError(f"Request failed: {e}") from e


# Convenience function for one-off translations
async def translate(
    text: str,
    source_lang: str,
    target_lang: str,
    base_url: str | None = None,
) -> TranslationResponse:
    """Convenience function for single translation.

    Args:
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code
        base_url: Optional custom base URL

    Returns:
        TranslationResponse with translation

    Example:
        result = await translate("Hello", "en", "zh")
        print(result.translation)
    """
    async with MTClient(base_url=base_url) as client:
        return await client.translate(text, source_lang, target_lang)


# Synchronous wrapper for non-async contexts
def translate_sync(
    text: str,
    source_lang: str,
    target_lang: str,
    base_url: str | None = None,
) -> TranslationResponse:
    """Synchronous wrapper for translate.

    Args:
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code
        base_url: Optional custom base URL

    Returns:
        TranslationResponse with translation

    Example:
        result = translate_sync("Hello", "en", "zh")
        print(result.translation)
    """
    return asyncio.run(translate(text, source_lang, target_lang, base_url))

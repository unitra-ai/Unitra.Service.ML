"""Tests for MT client module."""

import pytest
import httpx

from src.client.mt_client import (
    MTClient,
    MTClientError,
    TranslationResponse,
    BatchTranslationResponse,
    HealthResponse,
    translate,
)


class TestMTClientInit:
    """Tests for MTClient initialization."""

    def test_default_base_url(self):
        """Should use default base URL."""
        client = MTClient()
        assert "modal.run" in client.base_url

    def test_custom_base_url(self):
        """Should accept custom base URL."""
        client = MTClient(base_url="https://custom.example.com")
        assert client.base_url == "https://custom.example.com"

    def test_strips_trailing_slash(self):
        """Should strip trailing slash from base URL."""
        client = MTClient(base_url="https://example.com/")
        assert client.base_url == "https://example.com"

    def test_custom_timeout(self):
        """Should accept custom timeout."""
        client = MTClient(timeout=60.0)
        assert client.timeout == 60.0


class TestMTClientContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_enter_creates_client(self):
        """Should create HTTP client on enter."""
        client = MTClient()
        assert client._client is None

        async with client:
            assert client._client is not None
            assert isinstance(client._client, httpx.AsyncClient)

    @pytest.mark.asyncio
    async def test_exit_closes_client(self):
        """Should close HTTP client on exit."""
        client = MTClient()

        async with client:
            internal_client = client._client

        assert client._client is None

    def test_client_property_without_context_raises(self):
        """Should raise error when accessing client outside context."""
        client = MTClient()

        with pytest.raises(RuntimeError) as exc_info:
            _ = client.client

        assert "context manager" in str(exc_info.value)


class TestTranslationResponse:
    """Tests for TranslationResponse dataclass."""

    def test_creation(self):
        """Should create TranslationResponse correctly."""
        response = TranslationResponse(
            translation="你好",
            source_lang="en",
            target_lang="zh",
            tokens_used=10,
            latency_ms=150.5,
            processing_mode="cloud",
        )
        assert response.translation == "你好"
        assert response.tokens_used == 10
        assert response.latency_ms == 150.5


class TestBatchTranslationResponse:
    """Tests for BatchTranslationResponse dataclass."""

    def test_creation(self):
        """Should create BatchTranslationResponse correctly."""
        response = BatchTranslationResponse(
            translations=["你好", "世界"],
            source_lang="en",
            target_lang="zh",
            total_tokens=20,
            latency_ms=250.0,
            processing_mode="cloud",
        )
        assert len(response.translations) == 2
        assert response.total_tokens == 20


class TestHealthResponse:
    """Tests for HealthResponse dataclass."""

    def test_creation(self):
        """Should create HealthResponse correctly."""
        response = HealthResponse(
            status="healthy",
            model_id="google/madlad400-3b-mt",
            model_loaded=True,
            gpu_available=True,
            gpu_memory_used_gb=2.5,
            gpu_memory_total_gb=24.0,
            warm=True,
        )
        assert response.status == "healthy"
        assert response.gpu_memory_used_gb == 2.5

    def test_nullable_fields(self):
        """Should accept None for GPU fields."""
        response = HealthResponse(
            status="unhealthy",
            model_id="google/madlad400-3b-mt",
            model_loaded=False,
            gpu_available=False,
            gpu_memory_used_gb=None,
            gpu_memory_total_gb=None,
            warm=False,
        )
        assert response.gpu_memory_used_gb is None


class TestMTClientError:
    """Tests for MTClientError exception."""

    def test_basic_error(self):
        """Should create basic error."""
        error = MTClientError("Something went wrong")
        assert error.message == "Something went wrong"
        assert error.status_code is None
        assert error.response is None

    def test_error_with_status(self):
        """Should create error with status code."""
        error = MTClientError("Not found", status_code=404)
        assert error.status_code == 404

    def test_error_with_response(self):
        """Should create error with response data."""
        error = MTClientError(
            "Bad request",
            status_code=400,
            response={"error": "Invalid language"},
        )
        assert error.response == {"error": "Invalid language"}


class TestMTClientTranslate:
    """Tests for translate method with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_translate_success(self, httpx_mock):
        """Should translate text successfully."""
        httpx_mock.add_response(
            method="POST",
            url="https://test.example.com/translate",
            json={
                "translation": "你好",
                "source_lang": "en",
                "target_lang": "zh",
                "tokens_used": 10,
                "latency_ms": 150.0,
                "processing_mode": "cloud",
            },
        )

        async with MTClient(base_url="https://test.example.com") as client:
            result = await client.translate("Hello", "en", "zh")

        assert result.translation == "你好"
        assert result.source_lang == "en"
        assert result.target_lang == "zh"
        assert result.tokens_used == 10

    @pytest.mark.asyncio
    async def test_translate_http_error(self, httpx_mock):
        """Should raise MTClientError on HTTP error."""
        httpx_mock.add_response(
            method="POST",
            url="https://test.example.com/translate",
            status_code=400,
            json={"error": "Invalid language"},
        )

        async with MTClient(base_url="https://test.example.com") as client:
            with pytest.raises(MTClientError) as exc_info:
                await client.translate("Hello", "en", "invalid")

        assert exc_info.value.status_code == 400


class TestMTClientTranslateBatch:
    """Tests for translate_batch method with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_batch_success(self, httpx_mock):
        """Should translate batch successfully."""
        httpx_mock.add_response(
            method="POST",
            url="https://test.example.com/translate",
            json={
                "translations": ["你好", "世界"],
                "source_lang": "en",
                "target_lang": "zh",
                "total_tokens": 20,
                "latency_ms": 250.0,
                "processing_mode": "cloud",
            },
        )

        async with MTClient(base_url="https://test.example.com") as client:
            result = await client.translate_batch(["Hello", "World"], "en", "zh")

        assert len(result.translations) == 2
        assert result.translations[0] == "你好"
        assert result.total_tokens == 20


class TestMTClientHealthCheck:
    """Tests for health_check method with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, httpx_mock):
        """Should return healthy status."""
        httpx_mock.add_response(
            method="GET",
            url="https://test.example.com/health",
            json={
                "status": "healthy",
                "model_id": "google/madlad400-3b-mt",
                "model_loaded": True,
                "gpu_available": True,
                "gpu_memory_used_gb": 2.5,
                "gpu_memory_total_gb": 24.0,
                "warm": True,
            },
        )

        async with MTClient(base_url="https://test.example.com") as client:
            result = await client.health_check()

        assert result.status == "healthy"
        assert result.model_loaded is True
        assert result.warm is True


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_translate_function(self, httpx_mock):
        """Should work as standalone function."""
        httpx_mock.add_response(
            method="POST",
            url="https://test.example.com/translate",
            json={
                "translation": "你好",
                "source_lang": "en",
                "target_lang": "zh",
                "tokens_used": 10,
                "latency_ms": 150.0,
                "processing_mode": "cloud",
            },
        )

        result = await translate(
            "Hello",
            "en",
            "zh",
            base_url="https://test.example.com",
        )

        assert result.translation == "你好"


@pytest.fixture
def httpx_mock(monkeypatch):
    """Mock httpx.AsyncClient for testing."""
    import httpx

    class MockResponse:
        def __init__(self, status_code, json_data):
            self.status_code = status_code
            self._json = json_data
            self.content = b"mock"

        def json(self):
            return self._json

        def raise_for_status(self):
            if self.status_code >= 400:
                response = httpx.Response(self.status_code, content=b"mock")
                response._json = self._json
                # Create a proper mock response for HTTPStatusError
                raise httpx.HTTPStatusError(
                    "Mock error",
                    request=httpx.Request("POST", "http://test"),
                    response=MockHTTPResponse(self.status_code, self._json),
                )

    class MockHTTPResponse:
        def __init__(self, status_code, json_data):
            self.status_code = status_code
            self._json = json_data
            self.content = b"mock"

        def json(self):
            return self._json

    class MockAsyncClient:
        def __init__(self, **kwargs):
            self.responses = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def aclose(self):
            pass

        def add_response(self, method, url, json=None, status_code=200):
            self.responses.append({
                "method": method,
                "url": url,
                "json": json,
                "status_code": status_code,
            })

        async def post(self, url, **kwargs):
            for resp in self.responses:
                if resp["method"] == "POST" and url in resp["url"]:
                    return MockResponse(resp["status_code"], resp["json"])
            raise RuntimeError(f"No mock for POST {url}")

        async def get(self, url, **kwargs):
            for resp in self.responses:
                if resp["method"] == "GET" and url in resp["url"]:
                    return MockResponse(resp["status_code"], resp["json"])
            raise RuntimeError(f"No mock for GET {url}")

    mock_client = MockAsyncClient()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: mock_client)

    return mock_client

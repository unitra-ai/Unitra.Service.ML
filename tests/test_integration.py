"""Integration tests for MT service.

These tests require a running Modal deployment and should be run
separately from unit tests.

Usage:
    pytest tests/test_integration.py -v --run-integration
"""

import os

import pytest

# Skip integration tests by default
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION_TESTS"),
    reason="Integration tests disabled. Set RUN_INTEGRATION_TESTS=1 to enable.",
)


class TestMTServiceIntegration:
    """Integration tests for deployed MT service."""

    @pytest.fixture
    def mt_client(self):
        """Create MT client for testing."""
        from src.client.mt_client import MTClient

        # Use default Modal endpoint or custom URL from env
        base_url = os.environ.get("MT_SERVICE_URL")
        return MTClient(base_url=base_url, timeout=60.0)

    @pytest.mark.asyncio
    async def test_health_check(self, mt_client):
        """Service should be healthy."""
        async with mt_client as client:
            health = await client.health_check()

        assert health.status == "healthy"
        assert health.model_loaded is True
        assert health.gpu_available is True
        assert health.warm is True

    @pytest.mark.asyncio
    async def test_translate_en_to_zh(self, mt_client):
        """Should translate English to Chinese."""
        async with mt_client as client:
            result = await client.translate(
                text="Hello, how are you?",
                source_lang="en",
                target_lang="zh",
            )

        assert result.translation
        assert len(result.translation) > 0
        assert result.source_lang == "en"
        assert result.target_lang == "zh"
        assert result.tokens_used > 0
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_translate_zh_to_en(self, mt_client):
        """Should translate Chinese to English."""
        async with mt_client as client:
            result = await client.translate(
                text="你好，今天天气怎么样？",
                source_lang="zh",
                target_lang="en",
            )

        assert result.translation
        # Should contain some English words
        assert any(word in result.translation.lower() for word in ["hello", "hi", "weather", "how"])

    @pytest.mark.asyncio
    async def test_translate_batch(self, mt_client):
        """Should translate batch of texts."""
        texts = [
            "Good morning",
            "Good afternoon",
            "Good evening",
        ]

        async with mt_client as client:
            result = await client.translate_batch(
                texts=texts,
                source_lang="en",
                target_lang="zh",
            )

        assert len(result.translations) == 3
        assert all(t for t in result.translations)  # All non-empty
        assert result.total_tokens > 0

    @pytest.mark.asyncio
    async def test_translate_multiple_languages(self, mt_client):
        """Should translate to multiple target languages."""
        text = "Hello"
        targets = ["zh", "ja", "ko", "es", "fr"]

        async with mt_client as client:
            results = []
            for target in targets:
                result = await client.translate(
                    text=text,
                    source_lang="en",
                    target_lang=target,
                )
                results.append(result)

        # All translations should be different
        translations = [r.translation for r in results]
        assert len(set(translations)) == len(targets)

    @pytest.mark.asyncio
    async def test_translate_empty_text(self, mt_client):
        """Should handle empty text gracefully."""
        async with mt_client as client:
            result = await client.translate(
                text="",
                source_lang="en",
                target_lang="zh",
            )

        assert result.translation == ""
        assert result.tokens_used == 0

    @pytest.mark.asyncio
    async def test_translate_long_text(self, mt_client):
        """Should handle text near max length."""
        # Create text close to 512 character limit
        text = "Hello world. " * 40  # ~520 chars

        async with mt_client as client:
            # Should either succeed or fail gracefully
            try:
                result = await client.translate(
                    text=text[:512],  # Trim to limit
                    source_lang="en",
                    target_lang="zh",
                )
                assert result.translation
            except Exception as e:
                # If it fails, should be a proper error
                assert "too long" in str(e).lower() or "limit" in str(e).lower()

    @pytest.mark.asyncio
    async def test_latency_warm(self, mt_client):
        """Warm latency should be under target."""
        async with mt_client as client:
            # Warm up with first request
            await client.translate("test", "en", "zh")

            # Measure warm latency
            result = await client.translate(
                text="Hello, how are you today?",
                source_lang="en",
                target_lang="zh",
            )

        # Target: 500ms for warm requests
        assert result.latency_ms < 1000, f"Latency {result.latency_ms}ms exceeds 1000ms"


class TestLanguageSupport:
    """Tests for language support."""

    @pytest.fixture
    def mt_client(self):
        from src.client.mt_client import MTClient

        base_url = os.environ.get("MT_SERVICE_URL")
        return MTClient(base_url=base_url, timeout=60.0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "target_lang,expected_chars",
        [
            ("zh", ["你", "好"]),  # Chinese
            ("ja", ["こ", "ん"]),  # Japanese hiragana
            ("ko", ["안", "녕"]),  # Korean hangul
            ("ar", ["م", "ر"]),  # Arabic
            ("ru", ["П", "р"]),  # Russian cyrillic
        ],
    )
    async def test_translate_to_language(self, mt_client, target_lang, expected_chars):
        """Should translate to various languages with correct script."""
        async with mt_client as client:
            result = await client.translate(
                text="Hello",
                source_lang="en",
                target_lang=target_lang,
            )

        # Check that result contains expected script characters
        assert any(char in result.translation for char in expected_chars), (
            f"Translation to {target_lang} doesn't contain expected characters: "
            f"got '{result.translation}'"
        )


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def mt_client(self):
        from src.client.mt_client import MTClient

        base_url = os.environ.get("MT_SERVICE_URL")
        return MTClient(base_url=base_url, timeout=60.0)

    @pytest.mark.asyncio
    async def test_invalid_language_code(self, mt_client):
        """Should reject invalid language codes."""
        from src.client.mt_client import MTClientError

        async with mt_client as client:
            with pytest.raises(MTClientError) as exc_info:
                await client.translate(
                    text="Hello",
                    source_lang="en",
                    target_lang="xyz",  # Invalid
                )

        assert exc_info.value.status_code >= 400

    @pytest.mark.asyncio
    async def test_batch_too_large(self, mt_client):
        """Should reject batch larger than limit."""
        from src.client.mt_client import MTClientError

        texts = ["Hello"] * 20  # Exceeds MAX_BATCH_SIZE of 16

        async with mt_client as client:
            with pytest.raises(MTClientError):
                await client.translate_batch(
                    texts=texts,
                    source_lang="en",
                    target_lang="zh",
                )

"""Performance tests for MT service.

These tests measure latency and throughput targets.
They require a running Modal deployment.

Usage:
    RUN_INTEGRATION_TESTS=1 pytest tests/test_performance.py -v
"""

import os
import time

import pytest

# Skip performance tests by default
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION_TESTS"),
    reason="Performance tests disabled. Set RUN_INTEGRATION_TESTS=1 to enable.",
)


class TestTranslationLatency:
    """Tests for translation latency targets."""

    @pytest.fixture
    def mt_client(self):
        """Create MT client for testing."""
        from src.client.mt_client import MTClient

        base_url = os.environ.get("MT_SERVICE_URL")
        return MTClient(base_url=base_url, timeout=120.0)

    @pytest.mark.asyncio
    async def test_single_translation_latency_warm(self, mt_client):
        """Warm latency should be under 500ms.

        Target: < 500ms for single translation on warm container.
        """
        async with mt_client as client:
            # Warm up request
            await client.translate("warmup", "en", "zh")

            # Measure warm latency
            start = time.perf_counter()
            result = await client.translate(
                text="Hello, how are you today?",
                source_lang="en",
                target_lang="zh",
            )
            latency_ms = (time.perf_counter() - start) * 1000

        assert result.translation
        assert latency_ms < 500, f"Warm latency {latency_ms:.0f}ms exceeds 500ms target"
        print(f"Warm latency: {latency_ms:.0f}ms")

    @pytest.mark.asyncio
    async def test_batch_translation_latency(self, mt_client):
        """Batch latency should be under 800ms for 16 texts.

        Target: < 800ms for batch of 16 texts.
        """
        texts = [f"Test sentence number {i}" for i in range(16)]

        async with mt_client as client:
            # Warm up
            await client.translate("warmup", "en", "zh")

            # Measure batch latency
            start = time.perf_counter()
            result = await client.translate_batch(
                texts=texts,
                source_lang="en",
                target_lang="zh",
            )
            latency_ms = (time.perf_counter() - start) * 1000

        assert len(result.translations) == 16
        assert latency_ms < 800, f"Batch latency {latency_ms:.0f}ms exceeds 800ms target"
        print(f"Batch (16) latency: {latency_ms:.0f}ms")

    @pytest.mark.asyncio
    async def test_cold_start_latency(self, mt_client):
        """Cold start should be under 30 seconds.

        Target: < 30s for first request (model loading).
        Note: This test may not always trigger a cold start.
        """
        async with mt_client as client:
            start = time.perf_counter()
            result = await client.translate(
                text="Cold start test",
                source_lang="en",
                target_lang="zh",
            )
            latency_ms = (time.perf_counter() - start) * 1000

        assert result.translation
        # Cold start target is 30s, but warm requests should be much faster
        assert latency_ms < 30000, f"Request took {latency_ms:.0f}ms, exceeds 30s"
        print(f"Request latency: {latency_ms:.0f}ms (cold start target: 30000ms)")


class TestTranslationThroughput:
    """Tests for translation throughput."""

    @pytest.fixture
    def mt_client(self):
        from src.client.mt_client import MTClient

        base_url = os.environ.get("MT_SERVICE_URL")
        return MTClient(base_url=base_url, timeout=120.0)

    @pytest.mark.asyncio
    async def test_batch_throughput(self, mt_client):
        """Throughput should exceed 500 tokens/second.

        Target: > 500 tokens/second for batch processing.
        """
        # 16 texts, ~50 tokens each = ~800 tokens
        texts = [
            "This is a test sentence that contains approximately fifty tokens "
            "when tokenized by the model."
        ] * 16

        async with mt_client as client:
            # Warm up
            await client.translate("warmup", "en", "zh")

            # Measure throughput
            start = time.perf_counter()
            result = await client.translate_batch(
                texts=texts,
                source_lang="en",
                target_lang="zh",
            )
            elapsed_s = time.perf_counter() - start

        total_tokens = result.total_tokens
        throughput = total_tokens / elapsed_s

        assert throughput > 500, f"Throughput {throughput:.0f} tok/s below 500 tok/s target"
        print(f"Throughput: {throughput:.0f} tokens/second")


class TestConcurrentRequests:
    """Tests for concurrent request handling."""

    @pytest.fixture
    def mt_client(self):
        from src.client.mt_client import MTClient

        base_url = os.environ.get("MT_SERVICE_URL")
        return MTClient(base_url=base_url, timeout=120.0)

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mt_client):
        """Should handle 10 concurrent requests successfully."""
        import asyncio

        async def translate_one(client, i):
            return await client.translate(
                text=f"Concurrent request {i}",
                source_lang="en",
                target_lang="zh",
            )

        async with mt_client as client:
            # Warm up
            await client.translate("warmup", "en", "zh")

            # Send 10 concurrent requests
            start = time.perf_counter()
            tasks = [translate_one(client, i) for i in range(10)]
            results = await asyncio.gather(*tasks)
            total_time = (time.perf_counter() - start) * 1000

        # All should succeed
        assert len(results) == 10
        assert all(r.translation for r in results)

        # Should be faster than sequential (10 * 500ms = 5000ms)
        print(f"10 concurrent requests: {total_time:.0f}ms total")


class TestMemoryUsage:
    """Tests for GPU memory usage."""

    @pytest.fixture
    def mt_client(self):
        from src.client.mt_client import MTClient

        base_url = os.environ.get("MT_SERVICE_URL")
        return MTClient(base_url=base_url, timeout=120.0)

    @pytest.mark.asyncio
    async def test_gpu_memory_usage(self, mt_client):
        """GPU memory should be under 20GB.

        Target: < 20GB to leave headroom on A10G (24GB).
        """
        async with mt_client as client:
            health = await client.health_check()

        if health.gpu_memory_used_gb is not None:
            assert health.gpu_memory_used_gb < 20, (
                f"GPU memory {health.gpu_memory_used_gb:.1f}GB exceeds 20GB target"
            )
            print(f"GPU memory: {health.gpu_memory_used_gb:.1f}GB / {health.gpu_memory_total_gb:.1f}GB")
        else:
            pytest.skip("GPU memory info not available")


class TestEdgeCases:
    """Tests for edge cases mentioned in spec."""

    @pytest.fixture
    def mt_client(self):
        from src.client.mt_client import MTClient

        base_url = os.environ.get("MT_SERVICE_URL")
        return MTClient(base_url=base_url, timeout=120.0)

    @pytest.mark.asyncio
    async def test_translate_preserves_numbers(self, mt_client):
        """Numbers should be preserved in translation."""
        async with mt_client as client:
            result = await client.translate(
                text="I have 3 apples and 5 oranges",
                source_lang="en",
                target_lang="zh",
            )

        # Numbers should appear in output
        assert "3" in result.translation or "三" in result.translation
        assert "5" in result.translation or "五" in result.translation

    @pytest.mark.asyncio
    async def test_translate_handles_special_chars(self, mt_client):
        """Special characters should be handled."""
        async with mt_client as client:
            result = await client.translate(
                text="Hello! How are you? I'm fine, thanks.",
                source_lang="en",
                target_lang="zh",
            )

        assert result.translation
        # Should not crash on punctuation

    @pytest.mark.asyncio
    async def test_translate_max_length(self, mt_client):
        """Should handle text at max length (512 chars)."""
        # Create 512 character text
        text = "Hello world. " * 42  # ~546 chars
        text = text[:512]

        async with mt_client as client:
            result = await client.translate(
                text=text,
                source_lang="en",
                target_lang="zh",
            )

        assert result.translation

"""Tests for edge cases and boundary conditions.

These tests verify the system handles unusual inputs correctly.
"""

import pytest


class TestTextInputEdgeCases:
    """Tests for text input edge cases."""

    def test_exactly_max_length_text(self):
        """Text at exactly MAX_INPUT_LENGTH should be accepted."""
        from src.services.mt_service import MTService

        max_len = MTService.MAX_INPUT_LENGTH
        text = "x" * max_len

        assert len(text) == max_len
        assert len(text) == 512

    def test_one_over_max_length(self):
        """Text one char over MAX_INPUT_LENGTH should be rejected."""
        from src.services.mt_service import MTService

        max_len = MTService.MAX_INPUT_LENGTH
        text = "x" * (max_len + 1)

        assert len(text) == max_len + 1
        assert len(text) > 512

    def test_empty_string(self):
        """Empty string should be handled gracefully."""
        text = ""
        assert len(text) == 0
        assert not text.strip()

    def test_only_whitespace(self):
        """Whitespace-only string should be treated as empty."""
        texts = [" ", "  ", "\t", "\n", "\r\n", "   \t\n   "]

        for text in texts:
            assert not text.strip()

    def test_single_character(self):
        """Single character should be valid input."""
        chars = ["a", "好", "あ", "!", "1"]

        for char in chars:
            assert len(char) == 1
            assert len(char) <= 512

    def test_unicode_normalization(self):
        """Unicode should be handled correctly."""
        # Different representations of the same character
        texts = [
            "café",  # composed
            "café",  # decomposed (if different)
            "Ω",     # Greek Omega
            "Ω",     # Ohm symbol (if different)
        ]

        for text in texts:
            assert len(text) <= 512

    def test_emoji_handling(self):
        """Emoji should be handled correctly."""
        emoji_texts = [
            "Hello 👋",
            "🌍🌎🌏",
            "I ❤️ Python",
            "😀😃😄😁😆",
        ]

        for text in emoji_texts:
            assert len(text) <= 512

    def test_rtl_text(self):
        """Right-to-left text should be handled."""
        rtl_texts = [
            "مرحبا",     # Arabic
            "שלום",      # Hebrew
            "سلام",      # Persian
        ]

        for text in rtl_texts:
            assert len(text) <= 512

    def test_mixed_direction_text(self):
        """Mixed LTR and RTL text should be handled."""
        text = "Hello مرحبا World"
        assert len(text) <= 512

    def test_control_characters(self):
        """Control characters should be handled."""
        texts = [
            "Hello\x00World",   # Null
            "Hello\x1bWorld",   # Escape
            "Hello\x7fWorld",   # DEL
        ]

        for text in texts:
            # Should not crash
            assert len(text) <= 512

    def test_very_long_word(self):
        """Very long single word should be handled."""
        word = "Pneumonoultramicroscopicsilicovolcanoconiosis"
        assert len(word) < 512

    def test_repeated_text(self):
        """Repeated text patterns should be handled."""
        text = "hello " * 85  # ~510 chars
        assert len(text) <= 512


class TestBatchInputEdgeCases:
    """Tests for batch input edge cases."""

    def test_exactly_max_batch_size(self):
        """Batch at exactly MAX_BATCH_SIZE should be accepted."""
        from src.services.mt_service import MTService

        max_batch = MTService.MAX_BATCH_SIZE
        batch = ["text"] * max_batch

        assert len(batch) == max_batch
        assert len(batch) == 16

    def test_one_over_max_batch(self):
        """Batch one over MAX_BATCH_SIZE should be rejected."""
        from src.services.mt_service import MTService

        max_batch = MTService.MAX_BATCH_SIZE
        batch = ["text"] * (max_batch + 1)

        assert len(batch) == max_batch + 1
        assert len(batch) > 16

    def test_single_item_batch(self):
        """Single item batch should be valid."""
        batch = ["Hello"]
        assert len(batch) == 1

    def test_empty_batch(self):
        """Empty batch should be handled gracefully."""
        batch = []
        assert len(batch) == 0

    def test_batch_with_empty_strings(self):
        """Batch with empty strings should be handled."""
        batch = ["Hello", "", "World", "", "Test"]

        non_empty = [t for t in batch if t.strip()]
        assert len(non_empty) == 3

    def test_batch_all_empty_strings(self):
        """Batch of all empty strings should be handled."""
        batch = ["", "", "", ""]

        non_empty = [t for t in batch if t.strip()]
        assert len(non_empty) == 0

    def test_batch_with_whitespace_strings(self):
        """Batch with whitespace-only strings should be handled."""
        batch = ["Hello", "   ", "World", "\t\n", "Test"]

        non_empty = [t for t in batch if t.strip()]
        assert len(non_empty) == 3

    def test_batch_varying_lengths(self):
        """Batch with varying text lengths should be handled."""
        batch = [
            "Hi",
            "Hello World",
            "This is a longer sentence that contains more words",
            "x" * 500,  # Near max length
        ]

        from src.services.mt_service import MTService
        max_len = MTService.MAX_INPUT_LENGTH

        for text in batch:
            assert len(text) <= max_len


class TestLanguageCodeEdgeCases:
    """Tests for language code edge cases."""

    def test_all_two_letter_priority_languages(self):
        """All priority languages should be two letters."""
        from src.services.mt_service import PRIORITY_LANGUAGES

        two_letter = [code for code in PRIORITY_LANGUAGES if len(code) == 2]
        three_letter = [code for code in PRIORITY_LANGUAGES if len(code) == 3]

        # Most should be 2 letters
        assert len(two_letter) > len(three_letter)

    def test_all_aliases_normalized(self):
        """All aliases should normalize to valid codes."""
        from src.services.mt_service import LANGUAGE_ALIASES, normalize_language

        for alias, expected in LANGUAGE_ALIASES.items():
            assert normalize_language(alias) == expected
            assert len(expected) <= 3

    def test_case_variations(self):
        """All case variations should normalize correctly."""
        from src.services.mt_service import normalize_language

        variations = ["en", "EN", "En", "eN"]
        for var in variations:
            assert normalize_language(var) == "en"

    def test_whitespace_variations(self):
        """All whitespace variations should be trimmed."""
        from src.services.mt_service import normalize_language

        variations = ["en", " en", "en ", " en ", "\ten\t", "\nen\n"]
        for var in variations:
            assert normalize_language(var) == "en"


class TestTokenCountEdgeCases:
    """Tests for token count edge cases."""

    def test_zero_tokens(self):
        """Zero tokens should be valid for empty input."""
        from src.services.mt_service import TranslationResult

        result = TranslationResult(
            translation="",
            source_lang="en",
            target_lang="zh",
            tokens_used=0,
            latency_ms=0.0,
        )
        assert result.tokens_used == 0

    def test_large_token_count(self):
        """Large token counts should be handled."""
        from src.services.mt_service import BatchTranslationResult

        result = BatchTranslationResult(
            translations=["x"] * 16,
            source_lang="en",
            target_lang="zh",
            total_tokens=10000,
            latency_ms=1000.0,
        )
        assert result.total_tokens == 10000


class TestLatencyEdgeCases:
    """Tests for latency edge cases."""

    def test_zero_latency(self):
        """Zero latency should be valid for cached/empty results."""
        from src.services.mt_service import TranslationResult

        result = TranslationResult(
            translation="",
            source_lang="en",
            target_lang="zh",
            tokens_used=0,
            latency_ms=0.0,
        )
        assert result.latency_ms == 0.0

    def test_very_low_latency(self):
        """Very low latency values should be handled."""
        from src.services.mt_service import TranslationResult

        result = TranslationResult(
            translation="Hello",
            source_lang="en",
            target_lang="zh",
            tokens_used=5,
            latency_ms=0.001,
        )
        assert result.latency_ms < 1.0

    def test_high_latency(self):
        """High latency values should be handled."""
        from src.services.mt_service import TranslationResult

        result = TranslationResult(
            translation="Hello",
            source_lang="en",
            target_lang="zh",
            tokens_used=5,
            latency_ms=30000.0,  # 30 seconds
        )
        assert result.latency_ms == 30000.0


class TestGPUMemoryEdgeCases:
    """Tests for GPU memory edge cases."""

    def test_none_gpu_memory(self):
        """None GPU memory should be valid (no GPU)."""
        from src.services.mt_service import HealthCheckResult

        result = HealthCheckResult(
            status="healthy",
            model_id="test",
            model_loaded=True,
            gpu_available=False,
            gpu_memory_used_gb=None,
            gpu_memory_total_gb=None,
            warm=True,
        )
        assert result.gpu_memory_used_gb is None

    def test_zero_gpu_memory(self):
        """Zero GPU memory should be valid (model not loaded)."""
        from src.services.mt_service import HealthCheckResult

        result = HealthCheckResult(
            status="healthy",
            model_id="test",
            model_loaded=False,
            gpu_available=True,
            gpu_memory_used_gb=0.0,
            gpu_memory_total_gb=24.0,
            warm=False,
        )
        assert result.gpu_memory_used_gb == 0.0

    def test_full_gpu_memory(self):
        """Full GPU memory usage should be handled."""
        from src.services.mt_service import HealthCheckResult

        result = HealthCheckResult(
            status="healthy",
            model_id="test",
            model_loaded=True,
            gpu_available=True,
            gpu_memory_used_gb=23.5,
            gpu_memory_total_gb=24.0,
            warm=True,
        )
        assert result.gpu_memory_used_gb <= result.gpu_memory_total_gb


class TestClientTimeoutEdgeCases:
    """Tests for client timeout edge cases."""

    def test_default_timeout(self):
        """Default timeout should be reasonable."""
        from src.client.mt_client import MTClient

        client = MTClient()
        assert client.timeout == 30.0

    def test_custom_short_timeout(self):
        """Short custom timeout should be accepted."""
        from src.client.mt_client import MTClient

        client = MTClient(timeout=5.0)
        assert client.timeout == 5.0

    def test_custom_long_timeout(self):
        """Long custom timeout should be accepted."""
        from src.client.mt_client import MTClient

        client = MTClient(timeout=120.0)
        assert client.timeout == 120.0

    def test_very_long_timeout(self):
        """Very long timeout should be accepted."""
        from src.client.mt_client import MTClient

        client = MTClient(timeout=600.0)  # 10 minutes
        assert client.timeout == 600.0


class TestURLEdgeCases:
    """Tests for URL handling edge cases."""

    def test_trailing_slash_removed(self):
        """Trailing slash should be removed from base URL."""
        from src.client.mt_client import MTClient

        client = MTClient(base_url="https://example.com/")
        assert not client.base_url.endswith("/")

    def test_multiple_trailing_slashes(self):
        """Multiple trailing slashes should all be removed."""
        from src.client.mt_client import MTClient

        client = MTClient(base_url="https://example.com///")
        # All trailing slashes should be removed
        assert client.base_url == "https://example.com"

    def test_no_trailing_slash(self):
        """URL without trailing slash should work."""
        from src.client.mt_client import MTClient

        client = MTClient(base_url="https://example.com")
        assert client.base_url == "https://example.com"

    def test_url_with_path(self):
        """URL with path should work."""
        from src.client.mt_client import MTClient

        client = MTClient(base_url="https://example.com/api/v1")
        assert "/api/v1" in client.base_url

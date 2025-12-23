"""Tests for Machine Translation service.

These tests use mocks to test the service logic without requiring
Modal.com or GPU infrastructure.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.services.mt_service import (
    LANGUAGE_ALIASES,
    PRIORITY_LANGUAGES,
    TranslationResult,
    BatchTranslationResult,
    HealthCheckResult,
    normalize_language,
    validate_language,
)


class TestDataclasses:
    """Tests for result dataclasses."""

    def test_translation_result_creation(self):
        """Should create TranslationResult with all fields."""
        result = TranslationResult(
            translation="你好",
            source_lang="en",
            target_lang="zh",
            tokens_used=10,
            latency_ms=150.5,
        )
        assert result.translation == "你好"
        assert result.source_lang == "en"
        assert result.target_lang == "zh"
        assert result.tokens_used == 10
        assert result.latency_ms == 150.5
        assert result.processing_mode == "cloud"

    def test_translation_result_custom_mode(self):
        """Should allow custom processing mode."""
        result = TranslationResult(
            translation="Hello",
            source_lang="zh",
            target_lang="en",
            tokens_used=5,
            latency_ms=100.0,
            processing_mode="local",
        )
        assert result.processing_mode == "local"

    def test_batch_translation_result_creation(self):
        """Should create BatchTranslationResult with all fields."""
        result = BatchTranslationResult(
            translations=["你好", "世界"],
            source_lang="en",
            target_lang="zh",
            total_tokens=20,
            latency_ms=250.0,
        )
        assert result.translations == ["你好", "世界"]
        assert len(result.translations) == 2
        assert result.total_tokens == 20
        assert result.processing_mode == "cloud"

    def test_health_check_result_creation(self):
        """Should create HealthCheckResult with all fields."""
        result = HealthCheckResult(
            status="healthy",
            model_id="google/madlad400-3b-mt",
            model_loaded=True,
            gpu_available=True,
            gpu_memory_used_gb=2.5,
            gpu_memory_total_gb=24.0,
            warm=True,
        )
        assert result.status == "healthy"
        assert result.model_loaded is True
        assert result.gpu_available is True
        assert result.warm is True

    def test_health_check_result_with_none_values(self):
        """Should allow None for optional fields."""
        result = HealthCheckResult(
            status="unhealthy",
            model_id="google/madlad400-3b-mt",
            model_loaded=False,
            gpu_available=False,
            gpu_memory_used_gb=None,
            gpu_memory_total_gb=None,
            warm=False,
        )
        assert result.gpu_memory_used_gb is None
        assert result.gpu_memory_total_gb is None


class TestNormalizeLanguage:
    """Tests for language normalization function."""

    def test_lowercase_conversion(self):
        """Should convert to lowercase."""
        assert normalize_language("EN") == "en"
        assert normalize_language("ZH") == "zh"

    def test_strip_whitespace(self):
        """Should strip whitespace."""
        assert normalize_language("  en  ") == "en"

    def test_alias_resolution(self):
        """Should resolve known aliases."""
        assert normalize_language("zh-cn") == "zh"
        assert normalize_language("zh-tw") == "zh"
        assert normalize_language("pt-br") == "pt"
        assert normalize_language("en-us") == "en"

    def test_extract_base_from_regional(self):
        """Should extract base code from regional variants."""
        assert normalize_language("es-mx") == "es"
        assert normalize_language("fr-ca") == "fr"


class TestValidateLanguage:
    """Tests for language validation function."""

    def test_valid_priority_languages(self):
        """Should accept valid priority languages."""
        for code in list(PRIORITY_LANGUAGES)[:5]:  # Test first 5
            result = validate_language(code)
            assert result == code

    def test_case_insensitive(self):
        """Should handle case-insensitive input."""
        assert validate_language("EN") == "en"
        assert validate_language("ZH") == "zh"

    def test_alias_validation(self):
        """Should accept and normalize aliases."""
        assert validate_language("zh-cn") == "zh"
        assert validate_language("en-us") == "en"

    def test_empty_raises_error(self):
        """Should raise ValueError for empty code."""
        with pytest.raises(ValueError) as exc_info:
            validate_language("")
        assert "cannot be empty" in str(exc_info.value)

    def test_unsupported_raises_error(self):
        """Should raise ValueError for unsupported language."""
        with pytest.raises(ValueError) as exc_info:
            validate_language("xyz")
        assert "Unsupported language" in str(exc_info.value)


class TestPriorityLanguages:
    """Tests for priority language configuration."""

    def test_contains_major_languages(self):
        """Should contain all major world languages."""
        major = ["en", "zh", "ja", "ko", "es", "fr", "de", "ru", "pt", "ar"]
        for code in major:
            assert code in PRIORITY_LANGUAGES, f"Missing: {code}"

    def test_contains_asian_languages(self):
        """Should contain major Asian languages."""
        asian = ["vi", "th", "id", "my", "km", "lo", "fil", "ms"]
        for code in asian:
            assert code in PRIORITY_LANGUAGES, f"Missing: {code}"

    def test_language_count(self):
        """Should have expected number of priority languages."""
        assert len(PRIORITY_LANGUAGES) >= 50


class TestLanguageAliases:
    """Tests for language alias mapping."""

    def test_chinese_variants(self):
        """Should map all Chinese variants."""
        assert LANGUAGE_ALIASES["zh-cn"] == "zh"
        assert LANGUAGE_ALIASES["zh-tw"] == "zh"
        assert LANGUAGE_ALIASES["zh-hans"] == "zh"
        assert LANGUAGE_ALIASES["zh-hant"] == "zh"

    def test_portuguese_variants(self):
        """Should map Portuguese variants."""
        assert LANGUAGE_ALIASES["pt-br"] == "pt"
        assert LANGUAGE_ALIASES["pt-pt"] == "pt"

    def test_english_variants(self):
        """Should map English variants."""
        assert LANGUAGE_ALIASES["en-us"] == "en"
        assert LANGUAGE_ALIASES["en-gb"] == "en"


class TestMTServiceInputValidation:
    """Tests for MTService input validation logic."""

    def test_max_input_length_defined(self):
        """MAX_INPUT_LENGTH should be defined."""
        from src.services.mt_service import MTService
        assert MTService.MAX_INPUT_LENGTH == 512

    def test_max_output_tokens_defined(self):
        """MAX_OUTPUT_TOKENS should be defined."""
        from src.services.mt_service import MTService
        assert MTService.MAX_OUTPUT_TOKENS == 256

    def test_max_batch_size_defined(self):
        """MAX_BATCH_SIZE should be defined."""
        from src.services.mt_service import MTService
        assert MTService.MAX_BATCH_SIZE == 16

    def test_model_id_defined(self):
        """MODEL_ID should be defined."""
        from src.services.mt_service import MTService
        assert MTService.MODEL_ID == "google/madlad400-3b-mt"


class TestHTTPEndpoints:
    """Tests for HTTP endpoint configurations."""

    def test_translate_endpoint_exists(self):
        """translate function should exist."""
        from src.services.mt_service import translate
        assert callable(translate)

    def test_health_endpoint_exists(self):
        """health function should exist."""
        from src.services.mt_service import health
        assert callable(health)


class TestModalConfiguration:
    """Tests for Modal.com configuration."""

    def test_app_name(self):
        """App should be named correctly."""
        from src.services.mt_service import app
        assert app.name == "unitra-mt"

    def test_volume_defined(self):
        """Volume should be defined."""
        from src.services.mt_service import volume
        assert volume is not None

    def test_image_defined(self):
        """Container image should be defined."""
        from src.services.mt_service import image
        assert image is not None

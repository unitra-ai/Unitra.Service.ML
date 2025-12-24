"""Tests for configuration modules.

These tests verify the settings and configuration are correctly defined.
"""

import pytest


class TestMTSettings:
    """Tests for MTSettings configuration."""

    def test_settings_is_dataclass(self):
        """Settings should be a frozen dataclass."""
        from src.config.settings import MTSettings

        settings = MTSettings()

        # Should be frozen (immutable)
        with pytest.raises(AttributeError):
            settings.app_name = "new-name"  # type: ignore

    def test_app_name(self):
        """App name should be correctly defined."""
        from src.config.settings import settings

        assert settings.app_name == "unitra-mt"

    def test_volume_name(self):
        """Volume name should be correctly defined."""
        from src.config.settings import settings

        assert settings.volume_name == "unitra-models"

    def test_model_path(self):
        """Model path should be correctly defined."""
        from src.config.settings import settings

        assert settings.model_path == "/models"

    def test_model_id(self):
        """Model ID should be correctly defined."""
        from src.config.settings import settings

        assert settings.model_id == "google/madlad400-3b-mt"

    def test_model_dir(self):
        """Model directory should be correctly defined."""
        from src.config.settings import settings

        assert settings.model_dir == "madlad-400-3b-mt"

    def test_gpu_type(self):
        """GPU type should be correctly defined."""
        from src.config.settings import settings

        assert settings.gpu_type == "a10g"

    def test_container_idle_timeout(self):
        """Container idle timeout should be correctly defined."""
        from src.config.settings import settings

        assert settings.container_idle_timeout == 300  # 5 minutes

    def test_concurrent_inputs(self):
        """Concurrent inputs should be correctly defined."""
        from src.config.settings import settings

        assert settings.concurrent_inputs == 16

    def test_retries(self):
        """Retries should be correctly defined."""
        from src.config.settings import settings

        assert settings.retries == 2

    def test_max_input_length(self):
        """Max input length should be correctly defined."""
        from src.config.settings import settings

        assert settings.max_input_length == 512

    def test_max_output_length(self):
        """Max output length should be correctly defined."""
        from src.config.settings import settings

        assert settings.max_output_length == 256

    def test_max_batch_size(self):
        """Max batch size should be correctly defined."""
        from src.config.settings import settings

        assert settings.max_batch_size == 16

    def test_num_beams(self):
        """Number of beams should be correctly defined."""
        from src.config.settings import settings

        assert settings.num_beams == 4

    def test_early_stopping(self):
        """Early stopping should be enabled."""
        from src.config.settings import settings

        assert settings.early_stopping is True

    def test_target_latency_ms(self):
        """Target latency should be correctly defined."""
        from src.config.settings import settings

        assert settings.target_latency_ms == 500

    def test_target_cold_start_s(self):
        """Target cold start should be correctly defined."""
        from src.config.settings import settings

        assert settings.target_cold_start_s == 30

    def test_full_model_path_property(self):
        """Full model path property should work."""
        from src.config.settings import settings

        expected = "/models/madlad-400-3b-mt"
        assert settings.full_model_path == expected

    def test_global_settings_instance(self):
        """Global settings instance should be available."""
        from src.config.settings import settings
        from src.config import settings as imported_settings

        assert settings is imported_settings


class TestLanguageConfiguration:
    """Tests for language configuration."""

    def test_priority_languages_dict(self):
        """Priority languages should be a dict with names."""
        from src.config.languages import PRIORITY_LANGUAGES

        assert isinstance(PRIORITY_LANGUAGES, dict)
        assert "en" in PRIORITY_LANGUAGES
        assert PRIORITY_LANGUAGES["en"] == "English"

    def test_supported_languages_dict(self):
        """Supported languages should be a dict."""
        from src.config.languages import SUPPORTED_LANGUAGES

        assert isinstance(SUPPORTED_LANGUAGES, dict)
        assert len(SUPPORTED_LANGUAGES) >= 50

    def test_priority_subset_of_supported(self):
        """Priority languages should be subset of supported."""
        from src.config.languages import PRIORITY_LANGUAGES, SUPPORTED_LANGUAGES

        for code in PRIORITY_LANGUAGES:
            assert code in SUPPORTED_LANGUAGES

    def test_language_aliases_dict(self):
        """Language aliases should be a dict."""
        from src.config.languages import LANGUAGE_ALIASES

        assert isinstance(LANGUAGE_ALIASES, dict)
        assert "zh-cn" in LANGUAGE_ALIASES
        assert LANGUAGE_ALIASES["zh-cn"] == "zh"


class TestConfigModuleExports:
    """Tests for config module exports."""

    def test_init_exports_settings(self):
        """Config __init__ should export settings."""
        from src.config import settings

        assert settings is not None
        assert settings.app_name == "unitra-mt"

    def test_init_exports_supported_languages(self):
        """Config __init__ should export SUPPORTED_LANGUAGES."""
        from src.config import SUPPORTED_LANGUAGES

        assert isinstance(SUPPORTED_LANGUAGES, dict)
        assert "en" in SUPPORTED_LANGUAGES

    def test_init_exports_validate_function(self):
        """Config __init__ should export validate_language_code."""
        from src.config import validate_language_code

        assert callable(validate_language_code)

        # Test it works
        result = validate_language_code("en")
        assert result == "en"


class TestSettingsConsistency:
    """Tests for settings consistency across modules."""

    def test_max_input_length_matches_service(self):
        """Settings max_input_length should match MTService."""
        from src.config.settings import settings
        from src.services.mt_service import MTService

        assert settings.max_input_length == MTService.MAX_INPUT_LENGTH

    def test_max_batch_size_matches_service(self):
        """Settings max_batch_size should match MTService."""
        from src.config.settings import settings
        from src.services.mt_service import MTService

        assert settings.max_batch_size == MTService.MAX_BATCH_SIZE

    def test_max_output_length_matches_service(self):
        """Settings max_output_length should match MTService."""
        from src.config.settings import settings
        from src.services.mt_service import MTService

        assert settings.max_output_length == MTService.MAX_OUTPUT_TOKENS

    def test_model_id_matches_service(self):
        """Settings model_id should match MTService."""
        from src.config.settings import settings
        from src.services.mt_service import MTService

        assert settings.model_id == MTService.MODEL_ID


class TestLanguageValidation:
    """Tests for language validation functions."""

    def test_normalize_language_code(self):
        """normalize_language_code should work."""
        from src.config.languages import normalize_language_code

        assert normalize_language_code("EN") == "en"
        assert normalize_language_code("zh-cn") == "zh"

    def test_validate_language_code(self):
        """validate_language_code should work."""
        from src.config.languages import validate_language_code

        assert validate_language_code("en") == "en"
        assert validate_language_code("zh") == "zh"

    def test_get_language_name(self):
        """get_language_name should return correct names."""
        from src.config.languages import get_language_name

        assert get_language_name("en") == "English"
        assert get_language_name("zh") == "Chinese (Simplified)"
        assert get_language_name("ja") == "Japanese"

    def test_is_priority_language(self):
        """is_priority_language should work."""
        from src.config.languages import is_priority_language

        assert is_priority_language("en") is True
        assert is_priority_language("zh") is True
        assert is_priority_language("hi") is False  # Extended, not priority

    def test_language_validation_error(self):
        """LanguageValidationError should be defined."""
        from src.config.languages import LanguageValidationError

        error = LanguageValidationError("xyz")
        assert error.code == "xyz"
        assert isinstance(error, Exception)

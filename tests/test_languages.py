"""Tests for language configuration and validation."""

import pytest

from src.config.languages import (
    LANGUAGE_ALIASES,
    PRIORITY_LANGUAGES,
    SUPPORTED_LANGUAGES,
    LanguageValidationError,
    get_language_name,
    is_priority_language,
    normalize_language_code,
    validate_language_code,
)


class TestLanguageConfiguration:
    """Tests for language configuration constants."""

    def test_priority_languages_subset_of_supported(self):
        """Priority languages should be a subset of supported languages."""
        for code in PRIORITY_LANGUAGES:
            assert code in SUPPORTED_LANGUAGES

    def test_priority_languages_count(self):
        """Should have expected number of priority languages."""
        assert len(PRIORITY_LANGUAGES) == 20

    def test_supported_languages_count(self):
        """Should have expected number of supported languages."""
        assert len(SUPPORTED_LANGUAGES) >= 50

    def test_required_priority_languages(self):
        """Critical languages should be in priority list."""
        required = ["en", "zh", "ja", "ko", "es", "fr", "de", "ru"]
        for code in required:
            assert code in PRIORITY_LANGUAGES, f"Missing priority language: {code}"


class TestNormalizeLanguageCode:
    """Tests for language code normalization."""

    def test_lowercase_conversion(self):
        """Should convert to lowercase."""
        assert normalize_language_code("EN") == "en"
        assert normalize_language_code("ZH") == "zh"
        assert normalize_language_code("Ja") == "ja"

    def test_strip_whitespace(self):
        """Should strip whitespace."""
        assert normalize_language_code("  en  ") == "en"
        assert normalize_language_code("\tzh\n") == "zh"

    def test_alias_resolution(self):
        """Should resolve language aliases."""
        assert normalize_language_code("zh-cn") == "zh"
        assert normalize_language_code("zh-tw") == "zh"
        assert normalize_language_code("pt-br") == "pt"
        assert normalize_language_code("en-us") == "en"
        assert normalize_language_code("en-gb") == "en"

    def test_regional_variant_extraction(self):
        """Should extract base code from regional variants."""
        assert normalize_language_code("es-mx") == "es"
        assert normalize_language_code("fr-ca") == "fr"
        assert normalize_language_code("de-at") == "de"

    def test_two_letter_codes_unchanged(self):
        """Two-letter codes should remain unchanged."""
        assert normalize_language_code("en") == "en"
        assert normalize_language_code("zh") == "zh"
        assert normalize_language_code("ja") == "ja"


class TestValidateLanguageCode:
    """Tests for language code validation."""

    def test_valid_priority_languages(self):
        """Should accept valid priority languages."""
        for code in PRIORITY_LANGUAGES:
            result = validate_language_code(code)
            assert result == code

    def test_valid_supported_languages(self):
        """Should accept all supported languages."""
        for code in SUPPORTED_LANGUAGES:
            result = validate_language_code(code)
            assert result == code

    def test_valid_aliases(self):
        """Should accept and normalize valid aliases."""
        for alias, normalized in LANGUAGE_ALIASES.items():
            result = validate_language_code(alias)
            assert result == normalized

    def test_case_insensitive(self):
        """Should handle case-insensitive input."""
        assert validate_language_code("EN") == "en"
        assert validate_language_code("ZH") == "zh"
        assert validate_language_code("Ja") == "ja"

    def test_empty_code_raises_error(self):
        """Should raise error for empty code."""
        with pytest.raises(LanguageValidationError) as exc_info:
            validate_language_code("")
        assert "cannot be empty" in str(exc_info.value)

    def test_none_raises_error(self):
        """Should raise error for None."""
        with pytest.raises(LanguageValidationError):
            validate_language_code(None)  # type: ignore

    def test_unsupported_language_raises_error(self):
        """Should raise error for unsupported language."""
        with pytest.raises(LanguageValidationError) as exc_info:
            validate_language_code("xyz")
        assert "xyz" in str(exc_info.value)
        assert "Unsupported" in str(exc_info.value)

    def test_error_includes_code(self):
        """Error should include the invalid code."""
        try:
            validate_language_code("invalid")
        except LanguageValidationError as e:
            assert e.code == "invalid"


class TestGetLanguageName:
    """Tests for language name lookup."""

    def test_priority_language_names(self):
        """Should return correct names for priority languages."""
        assert get_language_name("en") == "English"
        assert get_language_name("zh") == "Chinese (Simplified)"
        assert get_language_name("ja") == "Japanese"
        assert get_language_name("ko") == "Korean"
        assert get_language_name("es") == "Spanish"
        assert get_language_name("fr") == "French"

    def test_supported_language_names(self):
        """Should return correct names for supported languages."""
        assert get_language_name("hi") == "Hindi"
        assert get_language_name("sw") == "Swahili"
        assert get_language_name("el") == "Greek"

    def test_unknown_language_fallback(self):
        """Should return fallback for unknown languages."""
        result = get_language_name("xyz")
        assert "Unknown" in result
        assert "xyz" in result

    def test_normalized_before_lookup(self):
        """Should normalize code before lookup."""
        assert get_language_name("EN") == "English"
        assert get_language_name("zh-cn") == "Chinese (Simplified)"


class TestIsPriorityLanguage:
    """Tests for priority language checking."""

    def test_priority_languages_return_true(self):
        """Priority languages should return True."""
        for code in PRIORITY_LANGUAGES:
            assert is_priority_language(code) is True

    def test_non_priority_languages_return_false(self):
        """Non-priority languages should return False."""
        non_priority = ["hi", "bn", "sw", "am"]
        for code in non_priority:
            assert is_priority_language(code) is False

    def test_normalized_before_check(self):
        """Should normalize code before checking."""
        assert is_priority_language("EN") is True
        assert is_priority_language("zh-cn") is True


class TestLanguageValidationError:
    """Tests for LanguageValidationError exception."""

    def test_error_with_code_only(self):
        """Should create error with just code."""
        error = LanguageValidationError("xyz")
        assert error.code == "xyz"
        assert "xyz" in str(error)

    def test_error_with_custom_message(self):
        """Should create error with custom message."""
        error = LanguageValidationError("xyz", "Custom error message")
        assert error.code == "xyz"
        assert "Custom error message" in str(error)

    def test_error_inherits_from_exception(self):
        """Error should be an Exception."""
        error = LanguageValidationError("xyz")
        assert isinstance(error, Exception)

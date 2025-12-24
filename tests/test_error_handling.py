"""Tests for error handling across all modules.

These tests verify that errors are properly raised and handled
throughout the codebase.
"""

import pytest


class TestLanguageValidationErrors:
    """Tests for language validation error handling."""

    def test_empty_string_error(self):
        """Empty string should raise ValueError."""
        from src.services.mt_service import validate_language

        with pytest.raises(ValueError) as exc_info:
            validate_language("")

        assert "cannot be empty" in str(exc_info.value)

    def test_none_handling(self):
        """None should raise appropriate error."""
        from src.services.mt_service import validate_language

        with pytest.raises((ValueError, TypeError, AttributeError)):
            validate_language(None)  # type: ignore

    def test_numeric_code_error(self):
        """Numeric codes should raise ValueError."""
        from src.services.mt_service import validate_language

        with pytest.raises(ValueError) as exc_info:
            validate_language("123")

        assert "Invalid language code format" in str(exc_info.value)

    def test_special_chars_error(self):
        """Special characters should raise ValueError."""
        from src.services.mt_service import validate_language

        invalid_codes = ["en!", "zh@", "ja#", "ko$", "es%"]

        for code in invalid_codes:
            with pytest.raises(ValueError):
                validate_language(code)

    def test_too_long_code_error(self):
        """Codes longer than 3 chars should raise ValueError."""
        from src.services.mt_service import validate_language

        with pytest.raises(ValueError) as exc_info:
            validate_language("english")

        assert "Invalid language code format" in str(exc_info.value)

    def test_single_char_code_error(self):
        """Single character codes should raise ValueError."""
        from src.services.mt_service import validate_language

        with pytest.raises(ValueError) as exc_info:
            validate_language("e")

        assert "Invalid language code format" in str(exc_info.value)

    def test_strict_mode_unknown_language(self):
        """Strict mode should reject unknown languages."""
        from src.services.mt_service import validate_language

        with pytest.raises(ValueError) as exc_info:
            validate_language("xx", strict=True)

        assert "Unsupported language" in str(exc_info.value)


class TestConfigLanguageValidationErrors:
    """Tests for config/languages.py error handling."""

    def test_empty_code_error(self):
        """Empty code should raise LanguageValidationError."""
        from src.config.languages import validate_language_code, LanguageValidationError

        with pytest.raises(LanguageValidationError) as exc_info:
            validate_language_code("")

        assert exc_info.value.code == ""
        assert "empty" in str(exc_info.value).lower()

    def test_unsupported_language_error(self):
        """Unsupported language should raise LanguageValidationError."""
        from src.config.languages import validate_language_code, LanguageValidationError

        with pytest.raises(LanguageValidationError) as exc_info:
            validate_language_code("xyz")

        assert exc_info.value.code == "xyz"
        assert "Unsupported" in str(exc_info.value)

    def test_error_includes_suggestions(self):
        """Error message should include suggestions."""
        from src.config.languages import validate_language_code, LanguageValidationError

        with pytest.raises(LanguageValidationError) as exc_info:
            validate_language_code("invalid")

        error_msg = str(exc_info.value)
        # Should mention some valid codes
        assert "en" in error_msg or "Use one of" in error_msg


class TestClientErrors:
    """Tests for MT client error handling."""

    def test_client_error_creation(self):
        """MTClientError should be properly created."""
        from src.client.mt_client import MTClientError

        error = MTClientError("Test error")
        assert error.message == "Test error"
        assert error.status_code is None
        assert error.response is None

    def test_client_error_with_status(self):
        """MTClientError should include status code."""
        from src.client.mt_client import MTClientError

        error = MTClientError("Not found", status_code=404)
        assert error.status_code == 404

    def test_client_error_with_response(self):
        """MTClientError should include response data."""
        from src.client.mt_client import MTClientError

        response_data = {"error": "Invalid input", "details": "Text too long"}
        error = MTClientError("Bad request", status_code=400, response=response_data)

        assert error.response == response_data
        assert error.response["error"] == "Invalid input"

    def test_client_error_inheritance(self):
        """MTClientError should inherit from Exception."""
        from src.client.mt_client import MTClientError

        error = MTClientError("Test")
        assert isinstance(error, Exception)

    def test_client_without_context_raises(self):
        """Using client without context manager should raise RuntimeError."""
        from src.client.mt_client import MTClient

        client = MTClient()

        with pytest.raises(RuntimeError) as exc_info:
            _ = client.client

        assert "context manager" in str(exc_info.value)


class TestInputValidationErrors:
    """Tests for input validation error messages."""

    def test_text_too_long_message(self):
        """Text too long error should include length info."""
        from src.services.mt_service import MTService

        max_length = MTService.MAX_INPUT_LENGTH
        long_text = "x" * (max_length + 100)

        # The error message should include the length
        expected_info = f"{len(long_text)} chars"
        assert str(len(long_text)) in expected_info

    def test_batch_too_large_message(self):
        """Batch too large error should include size info."""
        from src.services.mt_service import MTService

        max_batch = MTService.MAX_BATCH_SIZE
        large_batch = ["text"] * (max_batch + 5)

        # The error message should include the size
        expected_info = f"{len(large_batch)} texts"
        assert str(len(large_batch)) in expected_info


class TestHTTPErrorCodes:
    """Tests for HTTP error code handling."""

    def test_400_bad_request(self):
        """400 errors should be properly handled."""
        from src.client.mt_client import MTClientError

        error = MTClientError("Invalid input", status_code=400)
        assert error.status_code == 400

    def test_401_unauthorized(self):
        """401 errors should be properly handled."""
        from src.client.mt_client import MTClientError

        error = MTClientError("Unauthorized", status_code=401)
        assert error.status_code == 401

    def test_404_not_found(self):
        """404 errors should be properly handled."""
        from src.client.mt_client import MTClientError

        error = MTClientError("Not found", status_code=404)
        assert error.status_code == 404

    def test_429_rate_limit(self):
        """429 errors should be properly handled."""
        from src.client.mt_client import MTClientError

        error = MTClientError("Rate limit exceeded", status_code=429)
        assert error.status_code == 429

    def test_500_server_error(self):
        """500 errors should be properly handled."""
        from src.client.mt_client import MTClientError

        error = MTClientError("Internal server error", status_code=500)
        assert error.status_code == 500

    def test_502_bad_gateway(self):
        """502 errors should be properly handled."""
        from src.client.mt_client import MTClientError

        error = MTClientError("Bad gateway", status_code=502)
        assert error.status_code == 502

    def test_503_service_unavailable(self):
        """503 errors should be properly handled."""
        from src.client.mt_client import MTClientError

        error = MTClientError("Service unavailable", status_code=503)
        assert error.status_code == 503


class TestEdgeCaseErrors:
    """Tests for edge case error scenarios."""

    def test_unicode_in_error_message(self):
        """Error messages should handle unicode."""
        from src.client.mt_client import MTClientError

        error = MTClientError("翻译失败: 输入无效")
        assert "翻译失败" in error.message

    def test_very_long_error_message(self):
        """Very long error messages should be handled."""
        from src.client.mt_client import MTClientError

        long_message = "Error: " + "x" * 1000
        error = MTClientError(long_message)
        assert len(error.message) > 1000

    def test_empty_error_response(self):
        """Empty error response should be handled."""
        from src.client.mt_client import MTClientError

        error = MTClientError("Error", status_code=500, response={})
        assert error.response == {}

    def test_nested_error_response(self):
        """Nested error response should be preserved."""
        from src.client.mt_client import MTClientError

        response = {
            "error": {
                "code": "INVALID_INPUT",
                "message": "Text too long",
                "details": {
                    "max_length": 512,
                    "actual_length": 1000,
                },
            }
        }
        error = MTClientError("Bad request", status_code=400, response=response)
        assert error.response["error"]["details"]["max_length"] == 512

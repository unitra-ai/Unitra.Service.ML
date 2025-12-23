"""Configuration module for Unitra ML services."""

from src.config.languages import SUPPORTED_LANGUAGES, validate_language_code
from src.config.settings import settings

__all__ = ["settings", "SUPPORTED_LANGUAGES", "validate_language_code"]

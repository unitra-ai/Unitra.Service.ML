"""Language configuration and validation for MADLAD-400."""

# Priority languages for Unitra (gaming community focus)
PRIORITY_LANGUAGES = {
    "en": "English",
    "zh": "Chinese (Simplified)",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ru": "Russian",
    "pt": "Portuguese",
    "ar": "Arabic",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "tr": "Turkish",
    "pl": "Polish",
    "it": "Italian",
    "nl": "Dutch",
    "sv": "Swedish",
    "cs": "Czech",
    "uk": "Ukrainian",
}

# Extended language support (MADLAD-400 supports 400+ languages)
# This is a subset of commonly used languages
SUPPORTED_LANGUAGES = {
    **PRIORITY_LANGUAGES,
    # Additional Asian languages
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "my": "Myanmar (Burmese)",
    "km": "Khmer",
    "lo": "Lao",
    "fil": "Filipino",
    "ms": "Malay",
    # European languages
    "el": "Greek",
    "hu": "Hungarian",
    "ro": "Romanian",
    "bg": "Bulgarian",
    "sk": "Slovak",
    "hr": "Croatian",
    "sl": "Slovenian",
    "sr": "Serbian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "et": "Estonian",
    "fi": "Finnish",
    "da": "Danish",
    "no": "Norwegian",
    # Middle Eastern languages
    "he": "Hebrew",
    "fa": "Persian",
    "ur": "Urdu",
    # African languages
    "sw": "Swahili",
    "am": "Amharic",
    "ha": "Hausa",
    "yo": "Yoruba",
    "zu": "Zulu",
}

# Language code aliases (map variants to standard codes)
LANGUAGE_ALIASES = {
    "zh-cn": "zh",
    "zh-tw": "zh",  # MADLAD uses zh for both simplified and traditional
    "zh-hans": "zh",
    "zh-hant": "zh",
    "pt-br": "pt",
    "pt-pt": "pt",
    "en-us": "en",
    "en-gb": "en",
    "es-es": "es",
    "es-mx": "es",
    "fr-fr": "fr",
    "fr-ca": "fr",
    "de-de": "de",
    "de-at": "de",
    "de-ch": "de",
}


class LanguageValidationError(Exception):
    """Raised when language code is invalid or unsupported."""

    def __init__(self, code: str, message: str | None = None):
        self.code = code
        self.message = message or f"Unsupported language code: {code}"
        super().__init__(self.message)


def normalize_language_code(code: str) -> str:
    """Normalize a language code to standard ISO 639-1 format.

    Args:
        code: Language code (e.g., "zh-CN", "en-US", "ja")

    Returns:
        Normalized language code (e.g., "zh", "en", "ja")
    """
    # Convert to lowercase and strip whitespace
    code = code.lower().strip()

    # Check for aliases first
    if code in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[code]

    # Return as-is if it's a valid 2-letter code
    if len(code) == 2:
        return code

    # Try to extract base language from longer codes (e.g., "en-US" -> "en")
    if "-" in code:
        base = code.split("-")[0]
        if len(base) == 2:
            return base

    return code


def validate_language_code(code: str) -> str:
    """Validate and normalize a language code.

    Args:
        code: Language code to validate

    Returns:
        Normalized language code

    Raises:
        LanguageValidationError: If language code is invalid or unsupported
    """
    if not code or not isinstance(code, str):
        raise LanguageValidationError(str(code), "Language code cannot be empty")

    normalized = normalize_language_code(code)

    # Check if it's a supported language
    if normalized not in SUPPORTED_LANGUAGES:
        raise LanguageValidationError(
            code,
            f"Unsupported language code: '{code}'. "
            f"Use one of: {', '.join(sorted(PRIORITY_LANGUAGES.keys()))}",
        )

    return normalized


def get_language_name(code: str) -> str:
    """Get the human-readable name for a language code.

    Args:
        code: Language code (e.g., "zh", "en")

    Returns:
        Language name (e.g., "Chinese (Simplified)", "English")
    """
    normalized = normalize_language_code(code)
    return SUPPORTED_LANGUAGES.get(normalized, f"Unknown ({code})")


def is_priority_language(code: str) -> bool:
    """Check if a language is a priority language.

    Args:
        code: Language code to check

    Returns:
        True if the language is in the priority list
    """
    normalized = normalize_language_code(code)
    return normalized in PRIORITY_LANGUAGES

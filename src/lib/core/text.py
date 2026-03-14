"""
Text utility functions for processing and analyzing text data.

This module provides utility functions for working with text data, including:
- Language detection
- Text truncation and formatting
- Text sanitization and cleaning
- Text statistics (word counts, character counts)
- Text preprocessing for NLP and sentiment analysis
"""

import hashlib
import html
import re
import string
import unicodedata
from collections import Counter
from functools import lru_cache
from typing import Any

from lib.core.logging_config import get_logger

# Try to import language detection libraries
# with fallback options
try:
    import langid  # noqa: F401

    LANGID_AVAILABLE = True
except ImportError:
    LANGID_AVAILABLE = False

try:
    import langdetect  # noqa: F401
    from langdetect import DetectorFactory

    # Set seed for consistent results
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    import pycld2 as cld2  # noqa: F401

    CLD2_AVAILABLE = True
except ImportError:
    CLD2_AVAILABLE = False

# Configure logging
logger = get_logger(__name__)
# Constants
MAX_TEXT_LENGTH = 50000
MIN_TEXT_LENGTH_FOR_LANGDETECT = 10
DEFAULT_LANGUAGE = "en"
DEFAULT_TRUNCATE_SUFFIX = "..."

# Language code mappings
# ISO 639-1 code to language name
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "bn": "Bengali",
    "pa": "Punjabi",
    "te": "Telugu",
    "mr": "Marathi",
    "ta": "Tamil",
    "ur": "Urdu",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "tr": "Turkish",
    "pl": "Polish",
    "uk": "Ukrainian",
    "cs": "Czech",
    "sk": "Slovak",
    "ro": "Romanian",
    "hu": "Hungarian",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sr": "Serbian",
    "bs": "Bosnian",
    "sl": "Slovenian",
    "mk": "Macedonian",
    "el": "Greek",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "et": "Estonian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "he": "Hebrew",
    "fa": "Persian",
    "sw": "Swahili",
    "am": "Amharic",
    "af": "Afrikaans",
}

# Common stopwords for supported languages
STOPWORDS = {
    "en": {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "from",
        "up",
        "down",
        "of",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "don't",
        "should",
        "now",
        "d",
        "ll",
        "m",
        "o",
        "re",
        "ve",
        "y",
    },
    "es": {
        "a",
        "al",
        "algo",
        "algunas",
        "algunos",
        "ante",
        "antes",
        "como",
        "con",
        "contra",
        "cual",
        "cuando",
        "de",
        "del",
        "desde",
        "donde",
        "durante",
        "e",
        "el",
        "ella",
        "ellas",
        "ellos",
        "en",
        "entre",
        "era",
        "erais",
        "eran",
        "eras",
        "eres",
        "es",
        "esa",
        "esas",
        "ese",
        "eso",
        "esos",
        "esta",
        "estaba",
        "estabais",
        "estaban",
        "estabas",
        "estad",
        "estada",
        "estadas",
        "estado",
        "estados",
        "estamos",
        "estando",
        "estar",
        "estaremos",
        "estará",
        "estarán",
        "estarás",
        "estaré",
        "estaréis",
        "estaría",
        "estaríais",
        "estaríamos",
        "estarían",
        "estarías",
        "estas",
        "este",
        "estemos",
        "esto",
        "estos",
        "estoy",
        "estuve",
        "estuviera",
        "estuvierais",
        "estuvieran",
        "estuvieras",
        "estuvieron",
        "estuviese",
        "estuvieseis",
        "estuviesen",
        "estuvieses",
        "estuvimos",
        "estuviste",
        "estuvisteis",
        "estuviéramos",
        "estuviésemos",
        "estuvo",
        "está",
        "estábamos",
        "estáis",
        "están",
        "estás",
        "esté",
        "estéis",
        "estén",
        "estés",
        "fue",
        "fuera",
        "fuerais",
        "fueran",
        "fueras",
        "fueron",
        "fuese",
        "fueseis",
        "fuesen",
        "fueses",
        "fui",
        "fuimos",
        "fuiste",
        "fuisteis",
        "fuéramos",
        "fuésemos",
        "ha",
        "habida",
        "habidas",
        "habido",
        "habidos",
        "habiendo",
        "habremos",
        "habrá",
        "habrán",
        "habrás",
        "habré",
        "habréis",
        "habría",
        "habríais",
        "habríamos",
        "habrían",
        "habrías",
        "habéis",
        "había",
        "habíais",
        "habíamos",
        "habían",
        "habías",
        "han",
        "has",
        "hasta",
        "hay",
        "haya",
        "hayamos",
        "hayan",
        "hayas",
        "hayáis",
        "he",
        "hemos",
        "hube",
        "hubiera",
        "hubierais",
        "hubieran",
        "hubieras",
        "hubieron",
        "hubiese",
        "hubieseis",
        "hubiesen",
        "hubieses",
        "hubimos",
        "hubiste",
        "hubisteis",
        "hubiéramos",
        "hubiésemos",
        "hubo",
        "la",
        "las",
        "le",
        "les",
        "lo",
        "los",
        "me",
        "mi",
        "mis",
        "mucho",
        "muchos",
        "muy",
        "más",
        "mí",
        "mía",
        "mías",
        "mío",
        "míos",
        "nada",
        "ni",
        "no",
        "nos",
        "nosotras",
        "nosotros",
        "nuestra",
        "nuestras",
        "nuestro",
        "nuestros",
        "o",
        "os",
        "otra",
        "otras",
        "otro",
        "otros",
        "para",
        "pero",
        "poco",
        "por",
        "porque",
        "que",
        "quien",
        "quienes",
        "qué",
        "se",
        "sea",
        "seamos",
        "sean",
        "seas",
        "seremos",
        "será",
        "serán",
        "serás",
        "seré",
        "seréis",
        "sería",
        "seríais",
        "seríamos",
        "serían",
        "serías",
        "seáis",
        "si",
        "sido",
        "siendo",
        "sin",
        "sobre",
        "sois",
        "somos",
        "son",
        "soy",
        "su",
        "sus",
        "suya",
        "suyas",
        "suyo",
        "suyos",
        "sí",
        "también",
        "tanto",
        "te",
        "tendremos",
        "tendrá",
        "tendrán",
        "tendrás",
        "tendré",
        "tendréis",
        "tendría",
        "tendríais",
        "tendríamos",
        "tendrían",
        "tendrías",
        "tened",
        "tenemos",
        "tenga",
        "tengamos",
        "tengan",
        "tengas",
        "tengo",
        "tengáis",
        "tenida",
        "tenidas",
        "tenido",
        "tenidos",
        "teniendo",
        "tenéis",
        "tenía",
        "teníais",
        "teníamos",
        "tenían",
        "tenías",
        "ti",
        "tiene",
        "tienen",
        "tienes",
        "todo",
        "todos",
        "tu",
        "tus",
        "tuve",
        "tuviera",
        "tuvierais",
        "tuvieran",
        "tuvieras",
        "tuvieron",
        "tuviese",
        "tuvieseis",
        "tuviesen",
        "tuvieses",
        "tuvimos",
        "tuviste",
        "tuvisteis",
        "tuviéramos",
        "tuviésemos",
        "tuvo",
        "tuya",
        "tuyas",
        "tuyo",
        "tuyos",
        "tú",
        "un",
        "una",
        "uno",
        "unos",
        "vosotras",
        "vosotros",
        "vuestra",
        "vuestras",
        "vuestro",
        "vuestros",
        "y",
        "ya",
        "yo",
        "él",
        "éramos",
    },
    "fr": {
        "a",
        "au",
        "aux",
        "avec",
        "ce",
        "ces",
        "dans",
        "de",
        "des",
        "du",
        "elle",
        "en",
        "et",
        "eux",
        "il",
        "ils",
        "je",
        "la",
        "le",
        "les",
        "leur",
        "lui",
        "ma",
        "mais",
        "me",
        "même",
        "mes",
        "moi",
        "mon",
        "ni",
        "notre",
        "nous",
        "ou",
        "par",
        "pas",
        "pour",
        "qu",
        "que",
        "qui",
        "s",
        "sa",
        "se",
        "si",
        "son",
        "sur",
        "ta",
        "te",
        "tes",
        "toi",
        "ton",
        "tu",
        "un",
        "une",
        "votre",
        "vous",
        "c",
        "d",
        "j",
        "l",
        "à",
        "m",
        "n",
        "t",
        "y",
        "est",
        "été",
        "étée",
        "étées",
        "étés",
        "étant",
        "suis",
        "es",
        "sommes",
        "êtes",
        "sont",
        "serai",
        "seras",
        "sera",
        "serons",
        "serez",
        "seront",
        "serais",
        "serait",
        "serions",
        "seriez",
        "seraient",
        "étais",
        "était",
        "étions",
        "étiez",
        "étaient",
        "fus",
        "fut",
        "fûmes",
        "fûtes",
        "furent",
        "sois",
        "soit",
        "soyons",
        "soyez",
        "soient",
        "fusse",
        "fusses",
        "fût",
        "fussions",
        "fussiez",
        "fussent",
        "avoir",
        "ayant",
        "eu",
        "eue",
        "eues",
        "eus",
        "ai",
        "as",
        "avons",
        "avez",
        "ont",
        "aurai",
        "auras",
        "aura",
        "aurons",
        "aurez",
        "auront",
        "aurais",
        "aurait",
        "aurions",
        "auriez",
        "auraient",
        "avais",
        "avait",
        "avions",
        "aviez",
        "avaient",
        "eut",
        "eûmes",
        "eûtes",
        "eurent",
        "aie",
        "aies",
        "ait",
        "ayons",
        "ayez",
        "aient",
        "eusse",
        "eusses",
        "eût",
        "eussions",
        "eussiez",
        "eussent",
    },
}

# Regular expressions for text cleaning
URL_REGEX = re.compile(
    r"(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)",
    re.IGNORECASE,
)
EMAIL_REGEX = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+", re.IGNORECASE)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f700-\U0001f77f"  # alchemical symbols
    "\U0001f780-\U0001f7ff"  # Geometric Shapes
    "\U0001f800-\U0001f8ff"  # Supplemental Arrows-C
    "\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
    "\U0001fa00-\U0001fa6f"  # Chess Symbols
    "\U0001fa70-\U0001faff"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027b0"  # Dingbats
    "\U000024c2-\U0001f251"
    "]+",
    flags=re.UNICODE,
)
HASHTAG_REGEX = re.compile(r"#\w+")
MENTION_REGEX = re.compile(r"@\w+")
EXTRA_WHITESPACE_REGEX = re.compile(r"\s+")
HTML_TAGS_REGEX = re.compile(r"<.*?>")
NUMBERS_REGEX = re.compile(r"\d+")
PUNCTUATION_REGEX = re.compile(r"[^\w\s]")
SPECIAL_CHARS_REGEX = re.compile(r"[^a-zA-Z0-9\s]")


# Language detection functions
@lru_cache(maxsize=1024)
def detect_language(text: str) -> str:
    """
    Detect the language of the given text.

    Args:
        text: Text to analyze

    Returns:
        ISO 639-1 language code (2 characters)
    """
    if not text or len(text.strip()) < MIN_TEXT_LENGTH_FOR_LANGDETECT:
        return DEFAULT_LANGUAGE

    # Normalize and take just a sample for better performance
    sample_text = text[: min(len(text), 1000)]

    # Try multiple detection methods for better accuracy
    detected_langs = []

    # Method 1: langid
    if LANGID_AVAILABLE:
        try:
            import langid as langid_lib

            lang_id, _ = langid_lib.classify(sample_text)
            detected_langs.append(lang_id)
        except Exception as e:
            logger.debug("langid detection failed", error=str(e))

    # Method 2: langdetect
    if LANGDETECT_AVAILABLE:
        try:
            import langdetect as langdetect_lib

            lang_detect = langdetect_lib.detect(sample_text)
            detected_langs.append(lang_detect)
        except Exception as e:
            logger.debug("langdetect detection failed", error=str(e))

    # Method 3: cld2
    if CLD2_AVAILABLE:
        try:
            import pycld2 as cld2_lib

            is_reliable, _, details = cld2_lib.detect(sample_text)
            if is_reliable:
                cld2_lang = details[0][1]
                detected_langs.append(cld2_lang)
        except Exception as e:
            logger.debug("cld2 detection failed", error=str(e))

    # If no languages detected, fall back to default
    if not detected_langs:
        return DEFAULT_LANGUAGE

    # Get most common detected language
    counter = Counter(detected_langs)
    most_common_lang = counter.most_common(1)[0][0]

    # Validate language code
    if len(most_common_lang) != 2 or most_common_lang not in LANGUAGE_NAMES:
        return DEFAULT_LANGUAGE

    return most_common_lang


def get_language_name(language_code: str) -> str:
    """
    Get the full language name from a language code.

    Args:
        language_code: ISO 639-1 language code

    Returns:
        Full language name
    """
    return LANGUAGE_NAMES.get(language_code.lower(), "Unknown")


# Text truncation and formatting functions
def truncate_text(text: str, max_length: int = 100, suffix: str = DEFAULT_TRUNCATE_SUFFIX) -> str:
    """
    Truncate text to a specified length, adding a suffix if truncated.

    Args:
        text: Text to truncate
        max_length: Maximum length of the truncated text
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if not text:
        return ""

    # Return text as is if it's shorter than max_length
    if len(text) <= max_length:
        return text

    # Truncate text
    truncated = text[: max_length - len(suffix)].strip()

    # Try to truncate at word boundary if possible
    last_space = truncated.rfind(" ")
    if last_space > max_length * 0.8:  # Only truncate at word boundary if we're not losing too much text
        truncated = truncated[:last_space]

    return truncated + suffix


def format_as_paragraph(text: str, width: int = 80) -> str:
    """
    Format text as a paragraph with a specified width.

    Args:
        text: Text to format
        width: Maximum width of each line

    Returns:
        Formatted text
    """
    if not text:
        return ""

    # Clean up text first
    text = re.sub(EXTRA_WHITESPACE_REGEX, " ", text).strip()

    # Format as paragraph
    lines = []
    words = text.split()
    current_line: list[str] = []
    current_length = 0

    for word in words:
        # Check if adding this word would exceed the line width
        if current_length + len(word) + (1 if current_length > 0 else 0) > width:
            # Add the current line to the result and start a new line
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            # Add the word to the current line
            if current_length > 0:
                current_length += 1  # Account for space
            current_length += len(word)
            current_line.append(word)

    # Add the last line if there's anything left
    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)


def text_to_sentences(text: str) -> list[str]:
    """
    Split text into sentences.

    Args:
        text: Text to split

    Returns:
        List of sentences
    """
    if not text:
        return []

    # Basic sentence splitting
    # This is a simple implementation - for production use, consider using a more sophisticated NLP library
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Filter out empty sentences
    return [s.strip() for s in sentences if s.strip()]


# Text cleaning and sanitization functions
def remove_urls(text: str, replacement: str = " ") -> str:
    """
    Remove URLs from text.

    Args:
        text: Text to process
        replacement: String to replace URLs with

    Returns:
        Text with URLs removed
    """
    if not text:
        return ""

    return re.sub(URL_REGEX, replacement, text)


def remove_emails(text: str, replacement: str = " ") -> str:
    """
    Remove email addresses from text.

    Args:
        text: Text to process
        replacement: String to replace email addresses with

    Returns:
        Text with email addresses removed
    """
    if not text:
        return ""

    return re.sub(EMAIL_REGEX, replacement, text)


def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags from text.

    Args:
        text: Text to process

    Returns:
        Text with HTML tags removed
    """
    if not text:
        return ""

    # First unescape HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    return re.sub(HTML_TAGS_REGEX, "", text)


def remove_emojis(text: str, replacement: str = "") -> str:
    """
    Remove emojis from text.

    Args:
        text: Text to process
        replacement: String to replace emojis with

    Returns:
        Text with emojis removed
    """
    if not text:
        return ""

    return re.sub(EMOJI_PATTERN, replacement, text)


def remove_hashtags_and_mentions(text: str, replacement: str = "") -> str:
    """
    Remove hashtags and @mentions from text.

    Args:
        text: Text to process
        replacement: String to replace hashtags and mentions with

    Returns:
        Text with hashtags and mentions removed
    """
    if not text:
        return ""

    text = re.sub(HASHTAG_REGEX, replacement, text)
    text = re.sub(MENTION_REGEX, replacement, text)
    return text


def remove_special_characters(text: str, keep_spaces: bool = True) -> str:
    """
    Remove special characters from text.

    Args:
        text: Text to process
        keep_spaces: Whether to keep spaces

    Returns:
        Text with special characters removed
    """
    if not text:
        return ""

    if keep_spaces:
        return re.sub(SPECIAL_CHARS_REGEX, "", text)
    else:
        return "".join(c for c in text if c.isalnum())


def remove_extra_whitespace(text: str) -> str:
    """
    Remove extra whitespace from text.

    Args:
        text: Text to process

    Returns:
        Text with extra whitespace removed
    """
    if not text:
        return ""

    return re.sub(EXTRA_WHITESPACE_REGEX, " ", text).strip()


def normalize_text(text: str) -> str:
    """
    Normalize Unicode text by converting to ASCII where possible.

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Normalize Unicode
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")


def clean_text(text: str, options: dict[str, bool] | None = None) -> str:
    """
    Clean text with configurable options.

    Args:
        text: Text to clean
        options: Dictionary of cleaning options
            - remove_urls: Remove URLs
            - remove_emails: Remove email addresses
            - remove_html: Remove HTML tags
            - remove_emojis: Remove emojis
            - remove_hashtags_mentions: Remove hashtags and mentions
            - remove_special_chars: Remove special characters
            - remove_extra_spaces: Remove extra whitespace
            - normalize_unicode: Normalize Unicode
            - lowercase: Convert to lowercase

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    if options is None:
        options = {}

    # Default options
    default_options = {
        "remove_urls": True,
        "remove_emails": True,
        "remove_html": True,
        "remove_emojis": False,
        "remove_hashtags_mentions": False,
        "remove_special_chars": False,
        "remove_extra_spaces": True,
        "normalize_unicode": False,
        "lowercase": True,
    }

    # Update with user-provided options
    if options:
        default_options.update(options)

    # Apply cleaning operations
    cleaned_text = text

    if default_options["remove_html"]:
        cleaned_text = remove_html_tags(cleaned_text)

    if default_options["remove_urls"]:
        cleaned_text = remove_urls(cleaned_text)

    if default_options["remove_emails"]:
        cleaned_text = remove_emails(cleaned_text)

    if default_options["remove_emojis"]:
        cleaned_text = remove_emojis(cleaned_text)

    if default_options["remove_hashtags_mentions"]:
        cleaned_text = remove_hashtags_and_mentions(cleaned_text)

    if default_options["normalize_unicode"]:
        cleaned_text = normalize_text(cleaned_text)

    if default_options["remove_special_chars"]:
        cleaned_text = remove_special_characters(cleaned_text)

    if default_options["remove_extra_spaces"]:
        cleaned_text = remove_extra_whitespace(cleaned_text)

    if default_options["lowercase"]:
        cleaned_text = cleaned_text.lower()

    return cleaned_text


# Text statistics functions
def get_text_stats(text: str) -> dict[str, Any]:
    """
    Get statistics about text.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with text statistics
    """
    if not text:
        return {
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "paragraph_count": 0,
            "avg_word_length": 0,
            "avg_sentence_length": 0,
        }

    # Clean the text for analysis
    clean = remove_extra_whitespace(text)

    # Count characters (excluding whitespace)
    char_count = len(clean.replace(" ", ""))

    # Count words
    words = clean.split()
    word_count = len(words)

    # Count sentences
    sentences = text_to_sentences(clean)
    sentence_count = len(sentences)

    # Count paragraphs
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    paragraph_count = len(paragraphs)

    # Calculate averages
    avg_word_length = char_count / word_count if word_count > 0 else 0
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "avg_word_length": round(avg_word_length, 2),
        "avg_sentence_length": round(avg_sentence_length, 2),
    }


def get_word_frequency(text: str, remove_stopwords: bool = True, language: str = "en") -> dict[str, int]:
    """
    Get word frequency in text.

    Args:
        text: Text to analyze
        remove_stopwords: Whether to remove stopwords
        language: Language code for stopwords

    Returns:
        Dictionary with word frequencies
    """
    if not text:
        return {}

    # Clean the text
    clean = clean_text(
        text,
        {
            "lowercase": True,
            "remove_special_chars": True,
            "remove_urls": True,
            "remove_emails": True,
            "remove_html": True,
        },
    )

    # Split into words
    words = clean.split()

    # Remove stopwords if requested
    if remove_stopwords:
        stopwords = STOPWORDS.get(language.lower(), STOPWORDS["en"])
        words = [w for w in words if w.lower() not in stopwords]

    # Count word frequencies
    return dict(Counter(words))


# Text preprocessing for NLP
def tokenize_text(text: str) -> list[str]:
    """
    Tokenize text into words.

    Args:
        text: Text to tokenize

    Returns:
        List of tokens
    """
    if not text:
        return []

    # Clean the text
    clean = clean_text(
        text,
        {
            "lowercase": True,
            "remove_special_chars": False,
            "remove_urls": True,
            "remove_emails": True,
            "remove_html": True,
        },
    )

    # Tokenize
    # This is a simple implementation - for production use, consider using a more sophisticated NLP library
    tokens = []
    for word in clean.split():
        # Remove punctuation attached to words
        word = word.strip(string.punctuation)
        if word:
            tokens.append(word)

    return tokens


def remove_stopwords(tokens: list[str], language: str = "en") -> list[str]:
    """
    Remove stopwords from a list of tokens.

    Args:
        tokens: List of tokens
        language: Language code for stopwords

    Returns:
        List of tokens with stopwords removed
    """
    if not tokens:
        return []

    # Get stopwords for the specified language
    stopwords = STOPWORDS.get(language.lower(), STOPWORDS["en"])

    # Remove stopwords
    return [token for token in tokens if token.lower() not in stopwords]


def get_text_hash(text: str) -> str:
    """
    Generate a hash of the text for caching or identification.

    Args:
        text: Text to hash

    Returns:
        Hash of the text
    """
    if not text:
        return ""

    # Normalize the text
    normalized = remove_extra_whitespace(text.lower())

    # Generate hash
    return hashlib.md5(normalized.encode()).hexdigest()


# Text similarity and comparison functions
def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between two texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Jaccard similarity (0-1)
    """
    if not text1 or not text2:
        return 0.0

    # Tokenize texts
    tokens1 = set(tokenize_text(text1.lower()))
    tokens2 = set(tokenize_text(text2.lower()))

    # Calculate Jaccard similarity
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))

    return intersection / union if union > 0 else 0.0


def text_to_sentences_with_tokens(text: str, language: str = "en") -> list[dict[str, Any]]:
    """
    Split text into sentences with tokens.

    Args:
        text: Text to process
        language: Language code

    Returns:
        List of dictionaries, each containing a sentence and its tokens
    """
    if not text:
        return []

    # Split into sentences
    sentences = text_to_sentences(text)

    # Process each sentence
    result = []
    for sentence in sentences:
        # Tokenize
        tokens = tokenize_text(sentence)

        # Remove stopwords
        filtered_tokens = remove_stopwords(tokens, language)

        result.append(
            {
                "sentence": sentence,
                "tokens": tokens,
                "filtered_tokens": filtered_tokens,
                "token_count": len(tokens),
                "filtered_token_count": len(filtered_tokens),
            }
        )

    return result


def extract_key_phrases(text: str, language: str = "en", max_phrases: int = 5) -> list[str]:
    """
    Extract key phrases from text using a simple approach.

    Args:
        text: Text to analyze
        language: Language code
        max_phrases: Maximum number of phrases to extract

    Returns:
        List of key phrases
    """
    if not text or len(text) < 10:
        return []

    # Process text into sentences with tokens
    sentences_data = text_to_sentences_with_tokens(text, language)

    # Extract candidate phrases
    candidates = []
    for sentence_data in sentences_data:
        tokens = sentence_data["tokens"]
        if len(tokens) < 2:
            continue

        # Create phrases from adjacent tokens
        for i in range(len(tokens) - 1):
            phrase = f"{tokens[i]} {tokens[i + 1]}"
            candidates.append(phrase.lower())

    # Count phrase frequencies
    phrase_counts = Counter(candidates)

    # Get most common phrases
    most_common = phrase_counts.most_common(max_phrases)

    return [phrase for phrase, count in most_common if count > 1]


def is_positive_text(text: str, positive_words: set[str] | None = None) -> bool:
    """
    Simple heuristic to check if text has a positive sentiment.

    This is a very simplistic approach and should be replaced with a proper
    sentiment analysis model in production.

    Args:
        text: Text to analyze
        positive_words: Set of positive words

    Returns:
        True if the text seems positive, False otherwise
    """
    if not text:
        return False

    # Default positive words
    if positive_words is None:
        positive_words = {
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
            "fantastic",
            "terrific",
            "outstanding",
            "superb",
            "brilliant",
            "awesome",
            "fabulous",
            "incredible",
            "marvelous",
            "perfect",
            "happy",
            "enjoy",
            "like",
            "love",
            "best",
            "better",
            "positive",
            "beautiful",
            "recommend",
            "pleased",
            "glad",
            "delighted",
            "satisfied",
            "impressive",
        }

    # Tokenize and clean text
    tokens = [token.lower() for token in tokenize_text(text)]

    # Count positive words
    positive_count = sum(1 for token in tokens if token in positive_words)

    # Simple heuristic - if more than 10% of words are positive, consider it positive
    return positive_count > len(tokens) * 0.1


def is_negative_text(text: str, negative_words: set[str] | None = None) -> bool:
    """
    Simple heuristic to check if text has a negative sentiment.

    This is a very simplistic approach and should be replaced with a proper
    sentiment analysis model in production.

    Args:
        text: Text to analyze
        negative_words: Set of negative words

    Returns:
        True if the text seems negative, False otherwise
    """
    if not text:
        return False

    # Default negative words
    if negative_words is None:
        negative_words = {
            "bad",
            "poor",
            "terrible",
            "awful",
            "horrible",
            "dreadful",
            "appalling",
            "atrocious",
            "abysmal",
            "unacceptable",
            "unsatisfactory",
            "disappointing",
            "dissatisfied",
            "dislike",
            "hate",
            "worst",
            "worse",
            "negative",
            "ugly",
            "annoying",
            "frustrating",
            "useless",
            "waste",
            "regret",
            "sad",
            "angry",
            "annoyed",
            "disappointed",
            "upset",
            "unhappy",
            "problem",
            "issue",
            "complaint",
        }

    # Tokenize and clean text
    tokens = [token.lower() for token in tokenize_text(text)]

    # Count negative words
    negative_count = sum(1 for token in tokens if token in negative_words)

    # Simple heuristic - if more than 10% of words are negative, consider it negative
    return negative_count > len(tokens) * 0.1


def extract_entities(text: str) -> dict[str, list[str]]:
    """
    Extract basic entities from text using simple patterns.
    This is a very basic implementation - for production use,
    consider using a proper NER (Named Entity Recognition) model.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with entity types and values
    """
    if not text:
        return {"persons": [], "organizations": [], "locations": [], "dates": [], "emails": [], "urls": []}

    entities: dict[str, list[str]] = {
        "emails": [],
        "urls": [],
        "dates": [],
        "persons": [],
        "organizations": [],
        "locations": [],
    }

    # Extract emails
    emails = re.findall(EMAIL_REGEX, text)
    entities["emails"] = emails

    # Extract URLs
    urls = re.findall(URL_REGEX, text)
    entities["urls"] = [url[0] for url in urls if url[0]]

    # Extract dates (very simple pattern)
    date_patterns = [
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",  # DD/MM/YYYY or MM/DD/YYYY
        r"\d{4}[/-]\d{1,2}[/-]\d{1,2}",  # YYYY/MM/DD
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b",  # Month DD, YYYY
    ]

    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text))
    entities["dates"] = dates

    # For proper entity extraction, use a NER model
    # This is a placeholder for a more sophisticated implementation

    return entities


def get_text_readability(text: str) -> dict[str, float]:
    """
    Calculate readability scores for text.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with readability metrics
    """
    if not text or len(text) < 100:
        return {"flesch_reading_ease": 0, "flesch_kincaid_grade": 0, "syllables_per_word": 0}

    # Clean text
    clean = clean_text(text)

    # Split into sentences and words
    sentences = text_to_sentences(clean)
    words = tokenize_text(clean)

    # Count syllables (very rough approximation)
    def count_syllables(word):
        # This is a very basic syllable counter - not accurate for all words
        word = word.lower()
        if len(word) <= 3:
            return 1

        # Count vowel groups
        count = 0
        vowels = "aeiouy"
        prev_is_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel

        # Adjust for silent e at the end
        if word.endswith("e") and len(word) > 2 and word[-2] not in vowels:
            count -= 1

        # Ensure at least one syllable
        return max(1, count)

    # Count total syllables
    total_syllables = sum(count_syllables(word) for word in words)

    # Calculate metrics
    num_sentences = len(sentences)
    num_words = len(words)

    if num_sentences == 0 or num_words == 0:
        return {"flesch_reading_ease": 0, "flesch_kincaid_grade": 0, "syllables_per_word": 0}

    # Flesch Reading Ease score
    flesch_reading_ease = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (total_syllables / num_words)

    # Flesch-Kincaid Grade Level
    flesch_kincaid_grade = 0.39 * (num_words / num_sentences) + 11.8 * (total_syllables / num_words) - 15.59

    # Ensure scores are within reasonable ranges
    flesch_reading_ease = max(0, min(100, flesch_reading_ease))
    flesch_kincaid_grade = max(0, min(18, flesch_kincaid_grade))

    return {
        "flesch_reading_ease": round(flesch_reading_ease, 2),
        "flesch_kincaid_grade": round(flesch_kincaid_grade, 2),
        "syllables_per_word": round(total_syllables / num_words, 2) if num_words > 0 else 0,
    }

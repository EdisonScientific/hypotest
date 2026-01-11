import logging
import re

logger = logging.getLogger(__name__)


def extract_xml_content(text: str, tag_name: str) -> str | None:
    """
    Extract content between XML-like tags from text.

    Args:
        text (str): The text containing XML-like tags
        tag_name (str): The name of the tag to extract content from

    Returns:
        str or None: The content between the tags, or None if not found
    """
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()
    return None


def extract_code_from_markdown(text: str) -> str:
    r"""
    Extract code from markdown triple backticks, stripping backticks and language identifiers.

    Handles these formats:
    - ```code``` (single line or multiline without language)
    - ```\ncode\n``` (multiline without language)
    - ```python\ncode\n``` (with any language identifier)

    Args:
        text: Input text that may contain markdown code blocks

    Returns:
        Extracted code if wrapped in backticks, otherwise original text

    Examples:
        >>> extract_code_from_markdown("```python\nprint('hello')\n```")
        "print('hello')"
        >>> extract_code_from_markdown("```\nprint('hello')\n```")
        "print('hello')"
        >>> extract_code_from_markdown("```print('hello')```")
        "print('hello')"
        >>> extract_code_from_markdown("print('hello')")  # No backticks, returns original
        "print('hello')"
    """
    text = text.strip()

    # Known code language identifiers (for detection, not restriction)
    known_languages = {
        "python",
        "py",
        "r",
        "bash",
        "sh",
        "shell",
        "sql",
        "javascript",
        "js",
        "typescript",
        "ts",
        "c",
        "cpp",
        "java",
        "go",
        "rust",
        "php",
        "ruby",
        "swift",
        "kotlin",
    }

    # Check if text starts and ends with triple backticks
    if text.startswith("```") and text.endswith("```"):
        # Remove the opening and closing ```
        content = text[3:-3]

        # Check if first line might be a language identifier
        lines = content.split("\n")
        if len(lines) > 1:  # Multi-line content
            first_line = lines[0].strip().lower()
            # If first line looks like a language identifier, remove it
            if first_line in known_languages or (first_line and first_line.isalnum() and len(first_line) <= 15):
                content = "\n".join(lines[1:])

        return content.strip()

    # If no backticks found, return original text unchanged
    return text

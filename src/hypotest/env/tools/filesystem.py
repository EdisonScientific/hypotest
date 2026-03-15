"""Filesystem tools for reading, writing, and editing files.

This module provides tools for file operations including:
- Reading various file formats (text, images, PDFs, PowerPoint, notebooks)
- Writing new files
- Editing existing files (single and multi-edit)

Public API:
    make_filesystem_tools(work_dir: Path) -> dict[str, Tool]
        Factory function to create filesystem tools for a given working directory.

    FilesystemTool
        Class wrapper with working directory normalization.

    read_tool, write_tool, edit_tool, multi_edit_tool
        Raw tool functions (typically used via FilesystemTool wrapper).
"""

from pathlib import Path
from typing import Any, cast

import nbformat
import numpy as np
from aviary.core import Message, Tool

from hypotest.env.utils import img_utils
from hypotest.env.utils.notebook_utils import view_notebook

# =============================================================================
# CONSTANTS
# =============================================================================

# File size limits
MAX_PDF_SIZE = 10 * 1024 * 1024  # 10MB limit for PDFs
MAX_PPTX_SIZE = 10 * 1024 * 1024  # 10MB limit for PowerPoint files
MAX_TEXT_FILE_SIZE = 256 * 1024  # 256KB limit for text files

# Line formatting
MAX_LINE_LENGTH = 2000  # Truncate lines longer than this
MAX_TOTAL_CHARS = 30000  # Roughly 30KB max output

# Unsupported file type guidance messages - provides helpful instructions when
# users attempt to read unhandled files that should be analyzed with code instead.
UNSUPPORTED_FILE_GUIDANCE: dict[str, dict[str, str | set[str]]] = {
    "excel": {
        "extensions": {".xlsx", ".xls", ".xlsm", ".xlsb"},
        "message": (
            "This is an Excel file ({ext}). Excel files contain binary data "
            "and should be analyzed using Python code.\n\n"
            "Suggested approach:\n"
            "1. Use pandas to read the file: pd.read_excel('{path}')\n"
            "2. Explore the data structure, columns, and contents\n"
            "3. Perform any needed analysis or transformations\n\n"
            "Example code:\n"
            "```python\n"
            "import pandas as pd\n"
            "df = pd.read_excel('{path}')\n"
            "print(df.head())\n"
            "print(df.info())\n"
            "```"
        ),
    },
    "database": {
        "extensions": {".db", ".sqlite", ".sqlite3", ".db3"},
        "message": (
            "This is a database file ({ext}). Database files contain binary data "
            "and should be queried using code.\n\n"
            "Suggested approach:\n"
            "1. Use sqlite3 to connect: conn = sqlite3.connect('{path}')\n"
            "2. List tables and inspect schema\n"
            "3. Query the data you need\n\n"
            "Example code:\n"
            "```python\n"
            "import sqlite3\n"
            "conn = sqlite3.connect('{path}')\n"
            "cursor = conn.cursor()\n"
            "# List tables\n"
            "cursor.execute(\n"
            "    \"SELECT name FROM sqlite_master WHERE type='table'\"\n"
            ")\n"
            "print(cursor.fetchall())\n"
            "```"
        ),
    },
    "pickle": {
        "extensions": {".pkl", ".pickle"},
        "message": (
            "This is a Python pickle file ({ext}). Pickle files contain "
            "serialized Python objects.\n\n"
            "Suggested approach:\n"
            "1. Use pickle to load: obj = pickle.load(open('{path}', 'rb'))\n"
            "2. Inspect the object type and contents\n"
            "3. Analyze the data as needed\n\n"
            "Example code:\n"
            "```python\n"
            "import pickle\n"
            "with open('{path}', 'rb') as f:\n"
            "    obj = pickle.load(f)\n"
            "print(type(obj))\n"
            "print(obj)\n"
            "```"
        ),
    },
    "parquet": {
        "extensions": {".parquet", ".pq"},
        "message": (
            "This is a Parquet file ({ext}). Parquet files contain columnar data "
            "in binary format.\n\n"
            "Suggested approach:\n"
            "1. Use pandas or pyarrow to read: pd.read_parquet('{path}')\n"
            "2. Explore the data structure and contents\n\n"
            "Example code:\n"
            "```python\n"
            "import pandas as pd\n"
            "df = pd.read_parquet('{path}')\n"
            "print(df.head())\n"
            "print(df.info())\n"
            "```"
        ),
    },
    "hdf5": {
        "extensions": {".h5", ".hdf5", ".hdf"},
        "message": (
            "This is an HDF5 file ({ext}). HDF5 files contain hierarchical data "
            "in binary format.\n\n"
            "Suggested approach:\n"
            "1. Use h5py or pandas to read the file\n"
            "2. Explore the hierarchical structure\n"
            "3. Access specific datasets\n\n"
            "Example code:\n"
            "```python\n"
            "import h5py\n"
            "with h5py.File('{path}', 'r') as f:\n"
            "    print(list(f.keys()))  # Show top-level groups/datasets\n"
            "```"
        ),
    },
    "numpy": {
        "extensions": {".npy", ".npz"},
        "message": (
            "This is a NumPy file ({ext}). NumPy files contain array data "
            "in binary format.\n\n"
            "Suggested approach:\n"
            "1. Use numpy to load: arr = np.load('{path}')\n"
            "2. Inspect shape, dtype, and contents\n\n"
            "Example code:\n"
            "```python\n"
            "import numpy as np\n"
            "arr = np.load('{path}')\n"
            "print(f'Shape: {{arr.shape}}')\n"
            "print(f'Dtype: {{arr.dtype}}')\n"
            "print(arr)\n"
            "```"
        ),
    },
    "archive": {
        "extensions": {".zip", ".tar", ".gz", ".bz2", ".7z", ".rar"},
        "message": (
            "This is an archive file ({ext}). Archive files contain "
            "compressed data.\n\n"
            "Suggested approach:\n"
            "1. Use appropriate library to extract and list contents\n"
            "2. Extract specific files as needed\n\n"
            "For .zip files:\n"
            "```python\n"
            "import zipfile\n"
            "with zipfile.ZipFile('{path}', 'r') as z:\n"
            "    print(z.namelist())  # List contents\n"
            "    # z.extractall('destination/')  # Extract all\n"
            "```"
        ),
    },
}


# =============================================================================
# PUBLIC API - Main Tool Functions
# =============================================================================


def read_file(
    file_path: str, limit: int | None = None, offset: int | None = None
) -> str | Message | list[dict[str, Any]]:
    """Read file contents with line numbers.

    This tool is designed for reading human-readable files including source code,
    text files, configuration files, images, PDFs, PowerPoint presentations, and
    Jupyter notebooks.

    Supported File Types:
        - Text files: .txt, .md, .log, etc. (shown with line numbers)
        - Source code: .py, .js, .java, .cpp, .go, .rs, etc.
          (shown with line numbers)
        - Config files: .json, .yaml, .toml, .ini, .xml, etc.
          (shown with line numbers)
        - Images: .png, .jpg, .gif, etc. (rendered as images in Message)
        - PDFs: .pdf (text + images extracted, shown with page markers)
        - PowerPoint: .pptx (text + images extracted, shown with slide markers)
        - Notebooks: .ipynb (cells with code + outputs + images)

    Usage Notes:
        - For large files (>256KB), use offset/limit parameters to read
          specific sections
        - For PDF and PowerPoint files, offset/limit represent page/slide numbers
          (not line numbers)
        - For binary data files (Excel, databases, etc.), use Python code
          to analyze instead
        - For searching within files, use the Grep tool instead

    PDF and PowerPoint Specific Behavior:
        For PDF and PowerPoint files, offset and limit are interpreted as
        page/slide numbers:
        - offset: starting page/slide number (1-indexed)
        - limit: number of pages/slides to read from the starting position
        Example: offset=5, limit=3 reads pages/slides 5, 6, and 7

    Usage examples:
        Read entire file: read(file_path="script.py")
        Read lines 100-200: read(file_path="log.txt", offset=100, limit=100)
        Read PDF pages 1-5: read(file_path="report.pdf", offset=1, limit=5)
        Read PowerPoint slides 1-3: read(file_path="presentation.pptx", offset=1, limit=3)
        View image: read(file_path="chart.png")

    Args:
        file_path: Path to file to read
        limit: Optional limit on number of lines/pages to read
        offset: Optional line/page offset to start reading from

    Returns:
        File contents with line numbers (format: line_num→content) or Message
        for image/PDF/PowerPoint files
    """
    try:
        path, error = _validate_file_path(file_path)
        if error:
            return error
        return _dispatch_file_read(path, file_path, offset, limit)
    except Exception as e:
        return f"Error reading file: {e!s}"


def write_file(file_path: str, content: str) -> str:
    """Writes a file to the local filesystem with the given content.

    Usage:
    - You must pass the file path AND content to this tool as parameters. Do NOT put the content in the message body.
    - This tool will overwrite the existing file if there is one at the provided path.
    - If this is an existing file, you MUST use the read tool first to read the file's contents.
    - ALWAYS prefer editing existing files in the codebase.
    - Only use emojis if the user explicitly requests it.

    Usage Examples:
        write(file_path="new_file.txt", content="Hello, world!")       # Create new file
        write(file_path="existing_file.txt", content="New content")    # Overwrite existing file

    Args:
        file_path: Path to file to write
        content: Content to write to file (string)

    Returns:
        Confirmation message
    """
    try:
        target_path = Path(file_path).resolve()

        # Ensure parent directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        Path(target_path).write_text(content, encoding="utf-8")

        return f"File created successfully at: {file_path}"  # noqa: TRY300

    except Exception as e:
        return f"Error writing file: {e!s}"


def edit_file(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Edit file by replacing old_string with new_string.

    Usage:
    - You must read the file at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file.
    - When editing text from read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix.
    - The line number format is: `line_number→content` (e.g., `42→def foo():`). Everything after the → arrow is the actual file content to match.
    - Never include any part of the line number prefix in the old_string or new_string.
    - ALWAYS prefer editing existing files in the codebase.
    - Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.
    - The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`.
    - Use `replace_all` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance.

    Usage Examples:
        Edit file: edit(file_path="file.txt", old_string="old_string", new_string="new_string")    # replace the only occurrence of "old_string" with "new_string"
        Edit file with replace_all: edit("file.txt", "old_string", "new_string", replace_all=True)  # replace all occurrences of "old_string" with "new_string"

    Args:
        file_path: Path to file to edit (e.g. './file.txt' or '/home/user/file.txt')
        old_string: Text to replace
        new_string: Replacement text
        replace_all: Opt-in flag to replace all occurrences

    Returns:
        Confirmation message
    """  # noqa: E501
    try:
        target_path = Path(file_path).resolve()
        if not target_path.exists():
            return f"File does not exist: {file_path}"

        # Read file
        content = Path(target_path).read_text(encoding="utf-8")

        # Check if old_string exists
        if old_string not in content:
            return f"Text not found in file: {old_string}"

        # Replace text
        if replace_all:
            new_content = content.replace(old_string, new_string)
            count = content.count(old_string)
        else:
            new_content = content.replace(old_string, new_string, 1)
            count = 1

        # Write back to file
        Path(target_path).write_text(new_content, encoding="utf-8")

        return (  # noqa: TRY300
            f"File updated successfully. {count} replacement(s) made."
        )

    except Exception as e:
        return f"Error editing file: {e!s}"


def list_dir_tool(
    directory: str = ".",
    max_files: int = 20,
    show_hidden: bool = False,
) -> str:
    """List contents of a directory with truncation protection.

    Recursively lists files in a directory, with built-in protection against
    overwhelming the context with too many files. Use this tool instead of
    writing code to list directories to avoid context bloat.

    Usage Examples:
        list_dir()                      # List working directory
        list_dir("data/")               # List specific folder
        list_dir(max_files=50)          # Show more files
        list_dir(show_hidden=True)      # Include hidden files

    Args:
        directory: Directory path to list (default: current working directory)
        max_files: Maximum number of files to display (default: 20)
        show_hidden: Whether to show hidden files starting with '.' (default: False)
    """
    try:
        dir_path = Path(directory).resolve()
        if not dir_path.exists() or not dir_path.is_dir():
            return f"Path is not a directory: {directory}"

        paths = _collect_dir_paths(dir_path, show_hidden=show_hidden)

        if not paths:
            return "Directory is empty."

        if len(paths) > max_files:
            displayed = paths[:max_files]
            truncated = len(paths) - max_files
            return (
                "Files in directory:\n"
                + "\n".join(f"  {p}" for p in displayed)
                + f"\n  ({truncated} more files not shown)"
            )
        return "Files in directory:\n" + "\n".join(f"  {p}" for p in paths)

    except PermissionError:
        return f"Permission denied accessing directory: {directory}"
    except Exception as e:
        return f"Error listing directory: {e!s}"


def _collect_dir_paths(path: Path, prefix: str = "", show_hidden: bool = False) -> list[str]:
    """Recursively collect all file paths relative to the starting directory.

    Args:
        path: Directory path to list
        prefix: Current path prefix for recursion
        show_hidden: Whether to include hidden files

    Returns:
        List of relative file paths
    """
    paths: list[str] = []
    try:
        items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
    except PermissionError:
        rel_path = f"{prefix}{path.name}/" if prefix else f"{path.name}/"
        return [f"# {rel_path} (permission denied)"]

    for item in items:
        if not show_hidden and item.name.startswith("."):
            continue

        rel_path = f"{prefix}{item.name}" if prefix else item.name
        if item.is_dir():
            paths.extend(_collect_dir_paths(item, prefix=f"{rel_path}/", show_hidden=show_hidden))
        else:
            paths.append(rel_path)
    return paths


# =============================================================================
# PUBLIC API - Class Wrapper and Factory
# =============================================================================


class FilesystemTool:
    """Wraps filesystem tools with working directory normalization."""

    def __init__(self, work_dir: Path):
        self.work_dir = work_dir

    def _normalize_file_path(self, file_path: str) -> Path:
        """Normalize the path."""
        if not Path(file_path).is_absolute():
            return Path(self.work_dir) / file_path
        return Path(file_path).resolve()

    def read(
        self, file_path: str, limit: int | None = None, offset: int | None = None
    ) -> str | Message | list[dict[str, Any]]:
        return read_file(str(self._normalize_file_path(file_path)), limit, offset)

    def write(self, file_path: str, content: str) -> str:
        return write_file(str(self._normalize_file_path(file_path)), content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        return edit_file(
            str(self._normalize_file_path(file_path)),
            old_string,
            new_string,
            replace_all,
        )

    def list_dir(
        self,
        directory: str = ".",
        max_files: int = 20,
        show_hidden: bool = False,
    ) -> str:
        return list_dir_tool(str(self._normalize_file_path(directory)), max_files, show_hidden)

    read.__doc__ = read_file.__doc__
    read.__annotations__ = read_file.__annotations__
    write.__doc__ = write_file.__doc__
    write.__annotations__ = write_file.__annotations__
    edit.__doc__ = edit_file.__doc__
    edit.__annotations__ = edit_file.__annotations__
    list_dir.__doc__ = list_dir_tool.__doc__
    list_dir.__annotations__ = list_dir_tool.__annotations__


def make_filesystem_tools(work_dir: Path) -> dict[str, Tool]:
    """Factory function to create filesystem tools for a given working directory.

    Args:
        work_dir: Working directory for resolving relative paths

    Returns:
        Dictionary mapping tool names to Tool instances
    """
    filesystem_tool = FilesystemTool(work_dir)
    return {
        "read": Tool.from_function(filesystem_tool.read),
        "write": Tool.from_function(filesystem_tool.write),
        "edit": Tool.from_function(filesystem_tool.edit),
        "list_dir": Tool.from_function(filesystem_tool.list_dir),
    }


# =============================================================================
# INTERNAL - File Type Readers
# =============================================================================


def read_image_tool(path: Path) -> str | Message:
    """Read image file contents.

    Args:
        path: Path to image file to read

    Returns:
        Message with image content or error string
    """
    try:
        return img_utils.create_image_message(path, role="tool")
    except Exception as e:
        return f"Error reading image file: {e!s}"


def notebook_read_tool(path: Path) -> str | Message:
    """Read Jupyter notebook contents."""
    try:
        with path.open(encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)

        md_notebook, notebook_images = view_notebook(notebook.cells, "python")
        return Message.create_message(text=md_notebook, images=cast(list[np.ndarray | str], notebook_images))

    except Exception as e:
        return f"Error reading notebook file: {e!s}"


def _read_text_file(path: Path, offset: int | None, limit: int | None) -> str:
    """Read and format text file contents.

    Args:
        path: Path to file
        offset: Starting line number
        limit: Number of lines to read

    Returns:
        Formatted text with line numbers or error/guidance message
    """
    file_size = path.stat().st_size

    # Check if file is too large without pagination
    if (
        file_size > MAX_TEXT_FILE_SIZE
        and limit is None  # noqa: FURB124
        and offset is None
    ):
        size_mb = file_size / (1024 * 1024)
        return (
            f"File content ({size_mb:.1f}MB) exceeds maximum allowed size "
            "(256KB). Please use offset and limit parameters to read "
            "specific portions of the file, or use the Grep tool to search "
            "for specific content."
        )

    # Read and format file lines
    selected_lines = _read_file_lines(path, file_size, offset, limit)
    return _format_lines_with_numbers(selected_lines, offset)


def _read_file_lines(
    path: Path,
    file_size: int,
    offset: int | None,
    limit: int | None,
) -> list[str]:
    """Read file lines with optional offset and limit.

    Handles both small files (read all at once) and large files
    (read line by line). Falls back to latin-1 encoding if utf-8 fails.

    Args:
        path: Path to file
        file_size: Size of file in bytes
        offset: Starting line number (0-indexed)
        limit: Maximum number of lines to read

    Returns:
        List of selected lines
    """
    start_line = offset or 0
    max_lines = limit or 2000

    # For large files with offset/limit, read line by line to avoid memory issues
    if file_size > MAX_TEXT_FILE_SIZE and (limit is not None or offset is not None):
        selected_lines: list[str] = []
        try:
            with path.open(encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i < start_line:
                        continue
                    if len(selected_lines) >= max_lines:
                        break
                    selected_lines.append(line)
        except UnicodeDecodeError:
            # Try with different encoding
            with path.open(encoding="latin-1") as f:
                for i, line in enumerate(f):
                    if i < start_line:
                        continue
                    if len(selected_lines) >= max_lines:
                        break
                    selected_lines.append(line)
        return selected_lines

    # Read entire file for smaller files
    try:
        lines = _read_file_with_encoding(path)
    except UnicodeDecodeError:
        lines = _read_file_with_encoding(path, "latin-1")

    # Apply offset and limit
    end_line = start_line + (limit or len(lines))
    return lines[start_line:end_line]


def _read_file_with_encoding(path: Path, encoding: str = "utf-8") -> list[str]:
    """Read file lines with specified encoding.

    Args:
        path: Path to file
        encoding: Encoding to use (default: utf-8)

    Returns:
        List of lines from file
    """
    with path.open(encoding=encoding) as f:
        return f.readlines()


def _format_lines_with_numbers(lines: list[str], offset: int | None) -> str:
    """Format lines with line numbers and apply content protections.

    Args:
        lines: List of lines to format
        offset: Starting line number offset (0-indexed)

    Returns:
        Formatted string with line numbers
    """
    result = []
    actual_start = offset or 0
    total_chars = 0

    for i, line in enumerate(lines, start=actual_start + 1):
        # Remove trailing newline for formatting
        content = line.rstrip("\n\r")

        # Truncate very long lines
        if len(content) > MAX_LINE_LENGTH:
            content = content[:MAX_LINE_LENGTH] + " [line truncated - too long]"

        # Format line with line number
        formatted_line = f"{i}→{content}"

        # Check if adding this line would exceed total character limit
        if total_chars + len(formatted_line) + 1 > MAX_TOTAL_CHARS:
            result.append(
                "[Content truncated - output too large. Use offset/limit parameters to read specific portions.]"
            )
            break

        result.append(formatted_line)
        total_chars += len(formatted_line) + 1  # +1 for newline

    return "\n".join(result)


# =============================================================================
# INTERNAL - Utilities
# =============================================================================


def _validate_file_path(file_path: str) -> tuple[Path, str | None]:
    """Validate and resolve file path.

    Args:
        file_path: Path to validate

    Returns:
        Tuple of (resolved_path, error_message). Error message is None if valid.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        return path, f"File does not exist: {file_path}"
    if not path.is_file():
        return path, f"Path is not a file: {file_path}"
    return path, None


def _dispatch_file_read(path: Path, file_path: str, offset: int | None, limit: int | None) -> str | Message:
    """Dispatch to appropriate file reader based on file type.

    Args:
        path: Resolved path to file
        file_path: Original file path string (for error messages)
        offset: Line/page offset
        limit: Number of lines/pages to read

    Returns:
        File contents or error/guidance message
    """
    file_ext = path.suffix.lower()

    # Check for unsupported file types that need code-based analysis
    if unsupported_guidance := _get_unsupported_file_guidance(file_ext, file_path):
        return unsupported_guidance

    # Dispatch based on file type
    if img_utils.is_image_file(path):
        return read_image_tool(path)
    if file_ext == ".ipynb":
        return notebook_read_tool(path)

    # Default to text file handling
    return _read_text_file(path, offset, limit)


def _get_unsupported_file_guidance(file_ext: str, file_path: str) -> str | None:
    """Get guidance message for unsupported file types.

    Args:
        file_ext: File extension (lowercase, with leading dot)
        file_path: Path to the file

    Returns:
        Guidance message if file type is recognized, None otherwise
    """
    for config in UNSUPPORTED_FILE_GUIDANCE.values():
        if file_ext in config["extensions"]:
            return str(config["message"]).format(ext=file_ext, path=file_path)
    return None

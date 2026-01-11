import base64
import io
import mimetypes
from pathlib import Path

from aviary.core import Message
from PIL import Image

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".svg"}


def is_image_file(file_path: Path) -> bool:
    """Check if a file is likely an image based on its extension."""
    return file_path.suffix.lower() in IMG_EXTENSIONS


def resize_image_if_needed(image_data: bytes, max_dimension: int = 8000) -> bytes:
    """Resize image if any dimension exceeds max_dimension (in pixels)."""
    img = Image.open(io.BytesIO(image_data))
    if max(img.size) <= max_dimension:
        return image_data
    img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    img.save(output, format=img.format or "PNG")
    return output.getvalue()


def compress_image_if_needed(image_data: bytes, max_size_mib: float = 5.0) -> bytes:
    """Compress image if base64 encoded size exceeds max_size_mib.

    Uses a multi-stage approach:
    1. Check if already under limit
    2. Convert to JPEG and try progressive quality reduction
    3. If quality reduction insufficient, progressively scale down dimensions
    """
    max_bytes = int(max_size_mib * 1024 * 1024)
    small_image_size_threshold = 100

    # Check if already under limit
    if len(base64.b64encode(image_data)) <= max_bytes:
        return image_data

    img: Image.Image = Image.open(io.BytesIO(image_data))

    # Convert to RGB if necessary (JPEG doesn't support transparency)
    if img.mode in {"RGBA", "LA", "P"}:
        # Create white background for transparent images
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Try progressive quality reduction (JPEG actually uses quality parameter)
    for quality in (85, 70, 55, 40, 25, 15):
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=quality, optimize=True)
        compressed_data = output.getvalue()
        if len(base64.b64encode(compressed_data)) <= max_bytes:
            return compressed_data

    # Quality reduction wasn't enough, try scaling down dimensions
    current_img = img
    for scale_factor in (0.75, 0.5, 0.35, 0.25, 0.15, 0.1):
        new_size = (
            int(current_img.width * scale_factor),
            int(current_img.height * scale_factor),
        )
        # Avoid tiny images
        if new_size[0] < small_image_size_threshold or new_size[1] < small_image_size_threshold:
            break
        scaled_img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Try a range of qualities on the scaled image
        for quality in (70, 50, 30, 15):
            output = io.BytesIO()
            scaled_img.save(output, format="JPEG", quality=quality, optimize=True)
            compressed_data = output.getvalue()
            if len(base64.b64encode(compressed_data)) <= max_bytes:
                return compressed_data

    # If we still can't get under the limit, raise an error
    raise RuntimeError(f"Image could not be compressed below {max_size_mib} MiB limit.")


def _load_image_as_base64(file_path: Path) -> str:
    """Load an image file and return it as a base64 data URL."""
    try:
        image_data = Path(file_path).read_bytes()

        # Get MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/png"  # Default fallback

        # Apply resize and compression
        processed_image_data = resize_image_if_needed(image_data)
        processed_image_data = compress_image_if_needed(processed_image_data)

        # Encode as base64 data URL
        base64_data = base64.b64encode(processed_image_data).decode("utf-8")
        return f"data:{mime_type};base64,{base64_data}"  # noqa: TRY300

    except Exception as e:
        raise RuntimeError(f"Failed to load image {file_path}: {e}") from e


def encode_image_to_base64(image: str) -> str:
    decoded_image = base64.b64decode(image)
    decoded_image = resize_image_if_needed(decoded_image)
    decoded_image = compress_image_if_needed(decoded_image)
    return base64.b64encode(decoded_image).decode("utf-8")


def create_image_message(file_path: Path, role: str = "user") -> Message | str:
    """Create a message with an image."""
    try:
        image_data_url = _load_image_as_base64(file_path)
        return Message.create_message(
            role=role,
            images=[image_data_url],
        )
    except Exception as e:
        return f"Error loading image: {e!s}"

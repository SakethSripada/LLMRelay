"""Parse image blocks without fetching URLs or reading caller-supplied paths."""

import base64
import binascii
import re
from dataclasses import dataclass


MAX_IMAGES = 4
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 8 * 1024 * 1024

FORMATS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


class ImageError(ValueError):
    pass


@dataclass(frozen=True)
class ImageInput:
    data: bytes
    media_type: str

    @property
    def suffix(self) -> str:
        return FORMATS[self.media_type]


def _matches_format(data: bytes, media_type: str) -> bool:
    if media_type == "image/png":
        return data.startswith(b"\x89PNG\r\n\x1a\n") and data[12:16] == b"IHDR"
    if media_type == "image/jpeg":
        return data.startswith(b"\xff\xd8\xff")
    if media_type == "image/webp":
        return data.startswith(b"RIFF") and data[8:12] == b"WEBP"
    return data.startswith((b"GIF87a", b"GIF89a"))


def decode_image(media_type: str, encoded: str) -> ImageInput:
    if media_type not in FORMATS:
        raise ImageError("Supported image types are PNG, JPEG, WebP, and GIF.")
    if not isinstance(encoded, str) or len(encoded) > (MAX_IMAGE_BYTES * 4 // 3) + 8:
        raise ImageError("Image data is missing or exceeds 5 MiB.")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageError("Image data must be valid base64.") from exc
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ImageError("Image data is empty or exceeds 5 MiB.")
    if not _matches_format(data, media_type):
        raise ImageError("Image bytes do not match the declared media type.")
    return ImageInput(data, media_type)


def from_data_url(value: str) -> ImageInput:
    if not isinstance(value, str):
        raise ImageError("image_url must be a base64 data URL.")
    match = re.fullmatch(r"data:(image/[a-z]+);base64,([A-Za-z0-9+/=]+)", value)
    if not match:
        raise ImageError("Only base64 image data URLs are supported; remote URLs are not fetched.")
    return decode_image(match.group(1), match.group(2))


def check_image_budget(images: list[ImageInput]) -> None:
    if len(images) > MAX_IMAGES:
        raise ImageError(f"At most {MAX_IMAGES} images are allowed per request.")
    if sum(len(image.data) for image in images) > MAX_TOTAL_IMAGE_BYTES:
        raise ImageError("Images exceed the 8 MiB total limit.")

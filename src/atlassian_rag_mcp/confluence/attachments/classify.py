"""Classify attachments: convertible or skipped (video/audio/image/archive/binary)."""

from __future__ import annotations

VIDEO_EXTS = {
    "mp4", "mov", "webm", "avi", "mkv", "m4v", "wmv", "flv",
    "mpg", "mpeg", "3gp", "ts", "m2ts", "ogv",
}

AUDIO_EXTS = {
    "m4a", "mp3", "wav", "ogg", "aac", "flac", "wma",
    "opus", "m4b", "mid", "midi", "aiff", "amr",
}

IMAGE_EXTS = {
    "png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff",
    "webp", "ico", "heic", "heif", "avif", "psd",
}

ARCHIVE_EXTS = {
    "zip", "rar", "7z", "tar", "gz", "tgz", "bz2", "xz", "zst",
}

BINARY_EXTS = {
    "msi", "exe", "dll", "cer", "crt", "p12", "pfx", "der",
    "vsd", "vpp", "tfss", "bin", "dat", "db", "sqlite", "parquet", "avro",
}


def _ext(filename: str) -> str:
    name = (filename or "").rsplit("/", 1)[-1]
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[-1].lower()


def classify(filename: str, media_type: str | None = None) -> str | None:
    """Return ``None`` when the file should be converted,
    or ``"skipped:<reason>"`` when it must be skipped.

    Files without an extension (and unknown ones) return ``None`` — the
    extractor resolves them by magic bytes. ``image/`` MIME types are skipped,
    except svg, which is convertible.
    """
    mt = (media_type or "").lower()
    if mt.startswith("video/"):
        return "skipped:video"
    if mt.startswith("audio/"):
        return "skipped:audio"

    ext = _ext(filename)
    if ext in VIDEO_EXTS:
        return "skipped:video"
    if ext in AUDIO_EXTS:
        return "skipped:audio"
    if ext in IMAGE_EXTS:
        return "skipped:image"
    if ext in ARCHIVE_EXTS:
        return "skipped:archive"
    if ext in BINARY_EXTS:
        return "skipped:binary"

    if mt.startswith("image/"):
        if "svg" not in mt:
            return "skipped:image"
    return None

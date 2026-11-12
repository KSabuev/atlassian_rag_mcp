"""Exceptions for attachment text extraction."""

from __future__ import annotations


class AttachmentError(Exception):
    """Base error for attachment processing."""


class UnsupportedFormat(AttachmentError):
    """File type cannot be converted to text."""


class ConversionError(AttachmentError):
    """File was downloaded, but conversion failed."""


class AttachmentTooLarge(AttachmentError):
    """File exceeds the size limit."""

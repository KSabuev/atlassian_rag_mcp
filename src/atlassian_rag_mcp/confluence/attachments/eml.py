"""EML (email) → text: headers and text/plain body parts."""

from __future__ import annotations

import email
from email import policy

from .registry import converter


@converter(name="eml", exts=("eml",))
def convert(data: bytes, filename: str = "") -> str:
    msg = email.message_from_bytes(data, policy=policy.default)
    parts: list[str] = []
    for header in ("Subject", "From", "To", "Cc", "Date"):
        value = msg.get(header)
        if value:
            parts.append(f"{header}: {value}")
    bodies: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body = part.get_content()
                if body and body.strip():
                    bodies.append(body.strip())
    else:
        body = msg.get_content()
        if body and body.strip():
            bodies.append(body.strip())
    parts.extend(bodies)
    return "\n\n".join(parts)

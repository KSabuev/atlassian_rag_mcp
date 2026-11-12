"""Shared param types and helpers for the query builders.

The Pydantic field annotations (OptionalText / OptionalTextList / RequiredText)
give lenient coercion on input and strict types on output. Also holds the
helpers the CQL/JQL builders share: natural-language date parsing
(:func:`parse_date`) and ORDER BY handling (:func:`validate_order_by`,
:func:`has_top_level_order_by`, :func:`split_top_level_order_by`).
"""

from __future__ import annotations

import re
from calendar import monthrange
from datetime import date, timedelta
from typing import Annotated, Callable

from pydantic import BeforeValidator


def as_text(value: object) -> str | None:
    """Coerce a scalar param to str, or None for values not usable as text.

    Accepts str/int/float; rejects lists, dicts, bools and None so builders
    never crash on a stray cross-typed value from the client.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def as_list(value: object) -> list[str]:
    """Normalize a scalar or list/tuple param to a list of usable text items.

    A bare scalar becomes a single-item list; non-text items are dropped.
    """
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else [value]
    return [v for v in (as_text(i) for i in items) if v is not None]


def to_text(value: object) -> str | None:
    """Coerce a scalar to text, raising on values that cannot be text.

    Accepts str (as-is) and int/float (stringified); None stays None.
    Rejects bool, list, dict with a ``ValueError`` so invalid params surface
    as a validation error instead of being silently dropped.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    raise ValueError(f"expected a text value, got {type(value).__name__}")


def to_text_list(value: object) -> list[str] | None:
    """Normalize a scalar or list/tuple to a list of text values (strict).

    A bare scalar becomes a single-item list; ``None`` stays ``None``.
    Raises ``ValueError`` on non-text items (bool, dict, nested list, ...).
    """
    if value is None:
        return None
    items = value if isinstance(value, (list, tuple)) else [value]
    return [to_text(item) for item in items]


def to_required_text(value: object) -> str:
    """Like :func:`to_text` but rejects ``None`` (required-field guard)."""
    text = to_text(value)
    if text is None:
        raise ValueError("field is required")
    return text


def _clamp_to_month(year: int, month: int, day: int) -> date:
    """Clamp *day* to the length of *year-month* (Jan 31 + 1m -> Feb 28/29)."""
    return date(year, month, min(day, monthrange(year, month)[1]))


def parse_date(value: str) -> str:
    """Convert a natural-language date to ``YYYY-MM-DD``.

    Accepted: today/tomorrow/yesterday (English and Czech), "next week",
    "next month", relative ``+Nd/+Nw/+Nm/+Ny``, European DD.MM.YYYY /
    DD/MM/YYYY and ISO YYYY-MM-DD. Anything else is passed through for the
    server to handle. The day is clamped to the target month's length, so
    month rollovers from long months (Jan 31 + 1m) never produce an invalid
    date.
    """
    trimmed = value.strip().lower()
    today = date.today()

    if trimmed in ("today", "dnes"):
        return today.isoformat()
    if trimmed in ("tomorrow", "zitra", "zítra"):
        return (today + timedelta(days=1)).isoformat()
    if trimmed in ("yesterday", "vcera", "včera"):
        return (today - timedelta(days=1)).isoformat()
    if trimmed in ("next week", "pristi tyden", "příští týden"):
        return (today + timedelta(days=7)).isoformat()
    if trimmed in ("next month", "pristi mesic", "příští měsíc"):
        month = today.month + 1
        year = today.year + (1 if month > 12 else 0)
        if month > 12:
            month = 1
        return _clamp_to_month(year, month, today.day).isoformat()

    m = re.match(r"^\+(\d+)([dwmy])$", trimmed)
    if m:
        amount = int(m.group(1))
        unit = m.group(2)
        d = today
        if unit == "d":
            d += timedelta(days=amount)
        elif unit == "w":
            d += timedelta(weeks=amount)
        elif unit == "m":
            month = d.month + amount
            year = d.year + (month - 1) // 12
            month = (month - 1) % 12 + 1
            d = _clamp_to_month(year, month, d.day)
        elif unit == "y":
            d = _clamp_to_month(d.year + amount, d.month, d.day)
        return d.isoformat()

    m = re.match(r"^(\d{1,2})[./](\d{1,2})[./](\d{4})$", value)
    if m:
        return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"

    m = re.match(r"^(\d{4})[./-](\d{1,2})[./-](\d{1,2})$", value)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    return value


_DATE_FUNC_RE = re.compile(r"^[A-Za-z_]\w*\(")


def is_date_function(value: str) -> bool:
    """Whether *value* is a date-function call (``now()``, ``startOfYear()``, ...).

    Shared by the CQL and JQL builders: date functions must be emitted UNQUOTED
    in both query languages — inside quotes the server reads them as a plain
    string literal and rejects the query.
    """
    return bool(_DATE_FUNC_RE.match(value))


def build_date_condition(field: str, date_str: str, sanitize: Callable[[str], str]) -> str:
    """Build a ``field >= <date>`` condition from a natural-language date or a date function.

    Natural-language/ISO dates are converted via :func:`parse_date` and emitted
    quoted (sanitized); date-function calls (``now()``, ``startOfYear()``, ...)
    are passed through unquoted for the server to evaluate.
    """
    value = parse_date(date_str).strip()
    if is_date_function(value):
        return f"{field} >= {value}"
    return f'{field} >= "{sanitize(value)}"'


_ORDER_BY_SAFE = re.compile(r"^[a-zA-Z0-9\s,._@]+( DESC| ASC)?(, [a-zA-Z0-9\s,._@]+( DESC| ASC)?)*$", re.IGNORECASE)


def validate_order_by(clause: str) -> None:
    """Validate an ``ORDER BY`` clause contains only safe characters.

    Shared by the CQL and JQL builders so both reject unsafe ORDER BY input
    identically.
    """
    if not _ORDER_BY_SAFE.match(clause):
        raise ValueError(
            f"Unsafe characters in ORDER BY clause: '{clause}'. "
            f"Only field names, ASC, DESC, commas, and spaces are allowed."
        )


_ORDER_BY_RE = re.compile(r"\border\s+by\b", re.IGNORECASE)
_QUOTED_RE = re.compile(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"")
_QUOTED_OR_CLAUSE_RE = re.compile(
    r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|\bORDER\s+BY\b", re.IGNORECASE
)


def has_top_level_order_by(query: str) -> bool:
    """Check if *query* contains a real top-level ``ORDER BY`` clause.

    Quoted strings are stripped first so ``text ~ "order by x"`` does not
    count.
    """
    return bool(_ORDER_BY_RE.search(_QUOTED_RE.sub("", query)))


def split_top_level_order_by(clause: str) -> tuple[str, str]:
    """Split *clause* into ``(head, order)`` around a real top-level ``ORDER BY``.

    Returns ``(clause, "")`` when there is no top-level ORDER BY. Quoted strings
    are skipped so ``text ~ "order by x"`` is not split. A trailing ``AND``
    before the clause is dropped.
    """
    for m in _QUOTED_OR_CLAUSE_RE.finditer(clause):
        if not m.group(0).upper().startswith("ORDER BY"):
            continue
        head = clause[: m.start()].rstrip()
        if head.endswith("AND"):
            head = head[:-3].rstrip()
        return head, clause[m.start() :].strip()
    return clause, ""


OptionalText = Annotated[str | None, BeforeValidator(to_text)]
OptionalTextList = Annotated[list[str] | None, BeforeValidator(to_text_list)]
RequiredText = Annotated[str, BeforeValidator(to_required_text)]

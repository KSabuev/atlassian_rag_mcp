"""FTS text normalization — HTML cleanup, Russian stemming, query-side transforms.

The index side (chunk payload -> Tantivy doc) and the query side (FTS builder)
apply the SAME token transform, so inflected forms match: "модели" in a document
matches "модель" in a query and vice versa. Codes, acronyms, Latin and numeric
tokens are left untouched on both sides.

Tokenization invariants:
- Word parts are Unicode word chars MINUS underscore: the Tantivy default
  tokenizer also splits on underscores, so "Модель_данных" is indexed as the
  two tokens "модел"/"данных" — splitting the same way keeps every
  pure-Cyrillic part stem-able on both sides.
- Pattern tokens (code families) are an explicit whitelist of regexes —
  case codes (ABC.DEF.01, C_ABC.DEF_GHI.01, A_B.CD_EF.01), Jira ticket
  IDs (PROJ-1002, AB-15), short domain codes (A-CODE, B-TYPE) and release
  versions (release.2026.3.1.1028). A token is a pattern token ONLY if it
  matches one, so the `pattern` field stays free of noise (OBJECT_IDENTIFIER,
  APP_VERSION, base64 macro payloads and dates like 23.01.2023 are never
  indexed there). The same string drives both the field's Rust regex
  tokenizer (storage/fts/tantivy.py) and the query-side extraction; every
  pattern is LINEAR (bounded quantifiers, no backtracking) and compatible
  with both Python `re` and Rust `regex`, so index-side and query-side
  tokenization agree.
- RU_STOPWORDS (high-frequency Russian function words) are removed from
  plain-text query terms only — the index keeps them (stopword removal at
  index time is ineffective per Trotman et al. 2014, and phrases still need
  them).
"""

from __future__ import annotations

import re

from snowballstemmer import stemmer as _make_stemmer

_ru_stemmer = _make_stemmer("russian")

WORD_PART_RE = re.compile(r"[^\W_]+", re.UNICODE)
_CYRILIC_RUN_RE = re.compile(r"[а-яёА-ЯЁ]{3,}")
_CYR_RE = re.compile(r"^[а-яёА-ЯЁ]+$")

PATTERNS: tuple[str, ...] = (
    r"\b(?:[A-Z]_)?[A-Z]{1,10}(?:_[A-Z]{1,10})*(?:\.[A-Z]{1,10}(?:_[A-Z]{1,10})*){1,3}\.\d{1,4}\b",
    r"\b[A-Z]{2,10}-\d{1,6}\b",
    r"\b[A-Z]{1,3}-[A-Z]{2,10}\b",
    r"\brelease\.\d{4}\.\d{1,2}\.\d{1,2}(?:\.\d+)?\b",
)
PATTERN_REGEX = "|".join(PATTERNS)
PATTERN_RE = re.compile(PATTERN_REGEX)

RU_STOPWORDS = frozenset(
    """
    и в во не что а по он она оно они мы вы ты я с со как то так или но да нет
    же ли бы у на за от из до при над под между через после перед без ради про
    об о обо к ко около среди вокруг рядом вместо помимо согласно
    чтобы чтоб затем потом тогда здесь тут там сейчас теперь уже ещё еще только
    лишь почти совсем вполне именно больше более меньше менее много мало
    сам сама сами самое самой самый самых сама самим самой
    себя свой чей кто что какой который кто
    этот эта это эти этого этого этих этому этим этой этой
    такой такая такое такого такой таких такому таким такой такой
    какой какая какое какие какого какой каких какому каким какой какой
    все вся всё всех всем всей всему всеми всеми всеми
    есть быть был была было были будем будешь будете будет будет было
    имею имеешь имеет имеем имеете имеют
    очень где когда куда откуда куда почему зачем как сколько который
    для из от до по на у в во ей его её нее ему им их их ею ней
    """
    .split()
)


def stem_token(token: str) -> str:
    """Stem a single token (Russian snowball); non-Cyrillic and short tokens pass through.

    Deterministic and idempotent on the output — applied identically at index
    time and query time.
    """
    if len(token) >= 3 and _CYR_RE.match(token):
        return _ru_stemmer.stemWord(token.lower())
    return token


def normalize_index_text(text: str) -> str:
    """Tokenize, stem and rejoin — the transform applied to indexed chunk text/titles.

    Word parts follow the Tantivy default tokenizer (Unicode word chars minus
    underscore), so every part — including the pieces of underscore-joined
    identifiers like ``Модель_данных`` — is stemmed identically to how the
    query side prepares them.
    """
    if not text:
        return ""
    return " ".join(stem_token(t) for t in WORD_PART_RE.findall(text))


def stem_query_string(raw: str) -> str:
    """Stem Cyrillic word tokens inside a RAW Tantivy query, preserving structure.

    Field prefixes, quotes, operators, boosts and codes are untouched; every
    run of 3+ Cyrillic letters is stemmed (inside quoted phrases too) so raw
    queries match the stemmed index.
    """
    if not raw:
        return raw
    return _CYRILIC_RUN_RE.sub(lambda m: stem_token(m.group(0)), raw)


def extract_pattern_tokens(text: str) -> list[str]:
    """All tokens in ``text`` matching PATTERNS (query-side twin of the
    ``pattern`` field tokenizer).

    Duplicates are preserved — term frequency is meaningful for FTS scoring (FTS) on the
    pattern field (a chunk that mentions the code 3 times scores
    higher).
    """
    if not text:
        return []
    return PATTERN_RE.findall(text)

"""Geometry, value, and text normalization.

Two things in this module decide whether a citation is trustworthy:

* ``normalize_bbox`` puts every coordinate system the extractor emits into one
  top-left 0-1 space, so a stored region can be highlighted without knowing
  which artifact produced it.
* ``parse_value`` reads displayed numbers the way a report prints them --
  ``(40.2) %`` is a 40.2 point decrease, not a positive 40.2 -- using
  ``Decimal`` so an exact-value claim is compared exactly.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Iterable

from .models import GeometryPrecision, Region, RegionRole

logger = logging.getLogger(__name__)

# Bump when a change here would give the same source different stored evidence
# -- different normalized text, a different region, a different parsed value.
# It is part of the build fingerprint, so bumping it rebuilds; not bumping it
# after such a change leaves an index whose text no longer matches its rules.
#
# Unit and period canonicalization is deliberately *not* covered by that rule:
# `compare()` re-normalizes both the claim and the stored fact every time it
# runs, so an index built before this understood "MtCO2e" or "FY24" answers
# exactly as one built after it. Re-indexing a 494-page report to relabel a
# column would cost hours and change no verdict.
NORMALIZATION_VERSION = 1

# Dashes, minus signs, and non-breaking hyphens all render as "-" in a PDF but
# are distinct code points, which silently breaks substring quote checks.
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−­"), "-")
_QUOTES = {
    ord("‘"): "'",
    ord("’"): "'",
    ord("“"): '"',
    ord("”"): '"',
    ord(" "): " ",
    ord(" "): " ",
    ord(" "): " ",
}
_TRANSLATION = {**_DASHES, **_QUOTES}

_NUMBER_RE = re.compile(r"[-+]?\d[\d\s,._]*\d|\d")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_SCOPE_RE = re.compile(r"\bscope\s*([123])\b")
_APPROX_WORDS = ("about", "roughly", "approximately", "around", "circa", "~", "approx")
_DECREASE_WORDS = ("reduc", "decreas", "declin", "lower", "cut", "down by", "fell")
_INCREASE_WORDS = ("increas", "grew", "growth", "rose", "up by", "higher", "gain")


def clean_text(value: str) -> str:
    """Unicode-fold punctuation and collapse whitespace, preserving case."""
    folded = unicodedata.normalize("NFKC", value).translate(_TRANSLATION)
    return re.sub(r"\s+", " ", folded).strip()


def normalize_for_match(value: str) -> str:
    """Case-insensitive form used for substring quote verification and search."""
    return clean_text(value).casefold()


def contains_quote(haystack: str, quote: str) -> bool:
    """Whether ``quote`` really appears in the source text.

    This is the gate that rejects a fabricated fact: the model must echo text
    that exists in the evidence unit it cited, not a paraphrase of it.
    """
    needle = normalize_for_match(quote)
    return bool(needle) and needle in normalize_for_match(haystack)


# --- Geometry ---------------------------------------------------------------


def normalize_bbox(
    bbox: Any,
    page_width: float,
    page_height: float,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float], str]:
    """Return ``(normalized, source, origin)`` for one extractor bbox.

    Accepts the ``{"l","t","r","b","origin"}`` dicts the extractor writes and
    already-normalized ``[l, t, r, b]`` sequences. The normalized tuple is
    top-left, 0-1, and ordered so left <= right and top <= bottom.
    """
    if bbox is None:
        raise ValueError("bbox is required")
    if isinstance(bbox, dict):
        left = float(bbox["l"])
        right = float(bbox["r"])
        top = float(bbox["t"])
        bottom = float(bbox["b"])
        origin = str(bbox.get("origin") or "TOPLEFT").upper()
    else:
        left, top, right, bottom = (float(v) for v in bbox)
        origin = "NORMALIZED"

    source = (left, top, right, bottom)

    if origin == "NORMALIZED":
        norm = (left, top, right, bottom)
    else:
        if page_width <= 0 or page_height <= 0:
            raise ValueError("page size is required to normalize a bbox")
        if origin == "BOTTOMLEFT":
            top, bottom = page_height - top, page_height - bottom
        norm = (
            left / page_width,
            top / page_height,
            right / page_width,
            bottom / page_height,
        )

    left, top, right, bottom = norm
    if left > right:
        left, right = right, left
    if top > bottom:
        top, bottom = bottom, top
    clamp = lambda v: min(1.0, max(0.0, round(v, 6)))  # noqa: E731
    return (clamp(left), clamp(top), clamp(right), clamp(bottom)), source, origin


def union_bbox(
    boxes: Iterable[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    boxes = list(boxes)
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def region_from(
    bbox: Any,
    page_width: float,
    page_height: float,
    *,
    role: RegionRole,
    precision: GeometryPrecision,
) -> Region:
    norm, source, origin = normalize_bbox(bbox, page_width, page_height)
    return Region(
        bbox=norm,
        role=role,
        precision=precision,
        source_bbox=source,
        source_origin=origin,
    )


# --- Values -----------------------------------------------------------------


def parse_value(raw: str) -> tuple[Decimal | None, str | None]:
    """Parse one displayed cell/inline value into ``(decimal, unit)``.

    Accounting parentheses mean negative, which matters because a
    "(40.2) %" variation is a *reduction* of 40.2 points.
    """
    text = clean_text(raw)
    if not text:
        logger.debug(
            "value normalization found no value",
            extra={"event": "value_normalized", "raw_chars": len(raw), "normalized_value": None},
        )
        return None, None

    negative = False
    stripped = text.strip()
    # "(40.2) %" and "(40.2%)" both mean a negative variation.
    paren = re.match(r"^\((.*?)\)\s*(.*)$", stripped)
    if paren:
        negative = True
        stripped = f"{paren.group(1)} {paren.group(2)}".strip()

    match = _NUMBER_RE.search(stripped)
    if not match:
        unit = unit_of(text)
        logger.debug(
            "value normalization found no number",
            extra={
                "event": "value_normalized", "raw_chars": len(raw),
                "raw_value": None, "normalized_value": None, "unit": unit,
            },
        )
        return None, unit

    digits = re.sub(r"[\s,_]", "", match.group(0))
    # Trailing separators such as "1.234." are noise, not precision.
    digits = digits.rstrip(".")
    try:
        value = Decimal(digits)
    except InvalidOperation:
        unit = unit_of(text)
        logger.warning(
            "numeric value could not be normalized",
            extra={
                "event": "value_normalization_failed", "raw_chars": len(raw),
                "raw_value": match.group(0), "unit": unit,
            },
        )
        return None, unit
    if negative and value > 0:
        value = -value
    unit = unit_of(text)
    logger.debug(
        "value normalized",
        extra={
            "event": "value_normalized", "raw_chars": len(raw),
            "raw_value": match.group(0), "normalized_value": str(value),
            "unit": unit, "accounting_negative": negative,
        },
    )
    return value, unit


def unit_of(text: str) -> str | None:
    lowered = clean_text(text).casefold()
    if "%" in lowered or "percent" in lowered or "pp" == lowered.strip():
        return "%"
    return None


# One canonical spelling per unit. Only wordings that mean exactly the same
# quantity: "ton" and "tonne" are the same unit spelled two ways, whereas a
# short ton is not, and is deliberately absent.
_UNIT_ALIASES = {
    "percent": "%", "percentage": "%",
    "percentage point": "percentage points", "pp": "percentage points",
    "tonne": "t", "tonnes": "t", "ton": "t", "tons": "t",
    "metric ton": "t", "metric tons": "t",
    "metric tonne": "t", "metric tonnes": "t",
    "kilotonne": "kt", "kilotonnes": "kt", "thousand tonnes": "kt",
    "megatonne": "mt", "megatonnes": "mt", "million tonnes": "mt",
    "gigatonne": "gt", "gigatonnes": "gt", "billion tonnes": "gt",
    "kilogram": "kg", "kilograms": "kg", "gram": "g", "grams": "g",
    "litre": "l", "litres": "l", "liter": "l", "liters": "l",
    "cubic metre": "m3", "cubic metres": "m3",
    "cubic meter": "m3", "cubic meters": "m3", "m³": "m3",
    "kilometre": "km", "kilometres": "km", "kilometer": "km", "kilometers": "km",
}

# "MtCO2e", "Mt CO2-eq", "million tonnes of CO₂ equivalent". The equivalence
# marker is required: a tonne of CO2 and a tonne of CO2-equivalent are written
# differently because they are claimed differently, and a bare "tonnes of CO2"
# is left un-canonicalized so it falls through to cited semantic adjudication
# rather than being silently compared against a CO2e figure.
_CO2E_RE = re.compile(
    r"^(?P<prefix>g|k|m|gt|kt|mt|t|gigatonnes?|megatonnes?|kilotonnes?|tonnes?|"
    r"million tonnes|thousand tonnes|billion tonnes)?\s*(?:of\s+)?"
    r"co2\s*[-\s]?\s*(?:e|eq|equiv|equivalents?)\s*\.?$"
)
_CO2E_PREFIX = {
    "": "t", "t": "t", "tonne": "t", "tonnes": "t",
    "k": "kt", "kt": "kt", "kilotonne": "kt", "kilotonnes": "kt",
    "thousand tonnes": "kt",
    "m": "mt", "mt": "mt", "megatonne": "mt", "megatonnes": "mt",
    "million tonnes": "mt",
    "g": "gt", "gt": "gt", "gigatonne": "gt", "gigatonnes": "gt",
    "billion tonnes": "gt",
}

# Exact Decimal multipliers into one base unit per physical quantity. Only
# same-dimension prefixes are convertible; "t" (mass) and "tco2e" (emissions)
# share a base deliberately never, because a tonne of waste is not a tonne of
# CO2-equivalent. A unit absent from here is simply incomparable.
_UNIT_BASE: dict[str, tuple[str, Decimal]] = {
    "%": ("%", Decimal(1)),
    "percentage points": ("percentage points", Decimal(1)),
    "g": ("t", Decimal("0.000001")), "kg": ("t", Decimal("0.001")),
    "t": ("t", Decimal(1)), "kt": ("t", Decimal(1_000)),
    "mt": ("t", Decimal(1_000_000)), "gt": ("t", Decimal(1_000_000_000)),
    "tco2e": ("tco2e", Decimal(1)), "ktco2e": ("tco2e", Decimal(1_000)),
    "mtco2e": ("tco2e", Decimal(1_000_000)),
    "gtco2e": ("tco2e", Decimal(1_000_000_000)),
    "kwh": ("kwh", Decimal(1)), "mwh": ("kwh", Decimal(1_000)),
    "gwh": ("kwh", Decimal(1_000_000)), "twh": ("kwh", Decimal(1_000_000_000)),
    "mj": ("mj", Decimal(1)), "gj": ("mj", Decimal(1_000)),
    "tj": ("mj", Decimal(1_000_000)),
    "ml": ("l", Decimal("0.001")), "l": ("l", Decimal(1)),
    "hl": ("l", Decimal(100)), "m3": ("l", Decimal(1_000)),
    "km": ("km", Decimal(1)),
}


def normalize_unit(raw: str | None) -> str | None:
    """Collapse the many ways a report spells the same unit."""
    if raw is None:
        return None
    lowered = clean_text(raw).casefold().strip(" .")
    if not lowered:
        return None
    if "%" in lowered or lowered.startswith("percentage") or lowered == "percent":
        canonical, method = "%", "percent"
        logger.debug(
            "unit normalized",
            extra={
                "event": "unit_normalized", "raw_chars": len(raw),
                "normalized_unit": canonical, "normalization_method": method,
            },
        )
        return canonical
    match = _CO2E_RE.match(lowered)
    if match:
        prefix = (match.group("prefix") or "").strip()
        canonical = _CO2E_PREFIX.get(prefix)
        if canonical:
            result = f"{canonical}co2e"
            logger.debug(
                "unit normalized",
                extra={
                    "event": "unit_normalized", "raw_chars": len(raw),
                    "normalized_unit": result, "normalization_method": "co2e",
                },
            )
            return result
    result = _UNIT_ALIASES.get(lowered, lowered)
    logger.debug(
        "unit normalized",
        extra={
            "event": "unit_normalized", "raw_chars": len(raw),
            "normalized_unit": result,
            "normalization_method": "alias" if result != lowered else "unchanged",
        },
    )
    return result


# The spellings that really are a percent unit. `normalize_unit` answers "%" for
# anything *containing* one, which is right for a table header like "Percentage
# of variation" and wrong for a phrase that merely has a "%" further along.
_PERCENT_SPELLINGS = frozenset({
    "%", "percent", "percentage",
    "percentage point", "percentage points", "pp",
})


def known_unit(raw: str | None) -> str | None:
    """The canonical spelling, but only when the text *is* that unit.

    Stricter than :func:`normalize_unit` on purpose. This is what reads the unit
    stated next to a value, and there the surrounding words matter: in "Scope 1
    and 2 emissions by 40.2%" the text after the 1 contains a percent sign, and
    a lenient reading makes the metric name "Scope 1" into the value 1%.
    """
    unit = normalize_unit(raw)
    if unit not in _UNIT_BASE:
        return None
    if unit == "%" and clean_text(raw or "").casefold().strip(" .") not in _PERCENT_SPELLINGS:
        return None
    return unit


def unit_conversion(source: str | None, target: str | None) -> Decimal | None:
    """Exact multiplier from one unit into another, or None if they do not
    measure the same thing.

    Exact on purpose: every factor is a power of ten held as a ``Decimal``, so
    converting 21.3 Mt to tonnes is 21300000 and not 21299999.999999998.
    """
    if not source or not target:
        return None
    if source == target:
        return Decimal(1)
    left, right = _UNIT_BASE.get(source), _UNIT_BASE.get(target)
    if left is None or right is None or left[0] != right[0]:
        return None
    return left[1] / right[1]


def parse_period(text: str) -> str | None:
    """Pick the reporting period out of a header or sentence."""
    years = _YEAR_RE.findall(clean_text(text))
    if not years:
        return None
    matches = _YEAR_RE.finditer(clean_text(text))
    found = [m.group(0) for m in matches]
    return found[-1] if len(found) == 1 else found[0]


def all_years(text: str) -> list[str]:
    return [m.group(0) for m in _YEAR_RE.finditer(clean_text(text))]


# "FY24", "FY 2024", "FY2024". A two-digit year is read as 20xx: these are
# sustainability reports, and 1924 is not a reporting period anyone means.
_FY_RE = re.compile(r"\bFY\s?(\d{4}|\d{2})\b", re.I)
_PERIOD_RE = re.compile(r"\bFY\s?(?:\d{4}|\d{2})\b|\b(?:19|20)\d{2}\b", re.I)


def normalize_period(value: str | None) -> str | None:
    """One canonical label for a period a claim or a fact states.

    ``FY24`` and ``FY2024`` are the same fiscal period and normalize together.
    Neither becomes calendar ``2024``: a fiscal year that ends in June overlaps
    two calendar years, and quietly equating them would compare a figure to a
    period the report never reported.
    """
    if value is None:
        return None
    text = clean_text(value)
    if not text:
        return None
    match = _FY_RE.fullmatch(text) or _FY_RE.search(text)
    if match:
        year = match.group(1)
        result = f"FY{year if len(year) == 4 else '20' + year}"
        logger.debug(
            "period normalized",
            extra={
                "event": "period_normalized", "raw_chars": len(value),
                "normalized_period": result, "normalization_method": "fiscal_year",
            },
        )
        return result
    logger.debug(
        "period normalization left value unchanged",
        extra={
            "event": "period_normalized", "raw_chars": len(value),
            "normalized_period": text, "normalization_method": "unchanged",
        },
    )
    return text


def all_periods(text: str) -> list[str]:
    """Every period the text states, canonicalized, in the order stated."""
    return [
        normalize_period(m.group(0)) or m.group(0)
        for m in _PERIOD_RE.finditer(clean_text(text))
    ]


def detect_direction(text: str) -> str:
    lowered = clean_text(text).casefold()
    if any(word in lowered for word in _DECREASE_WORDS):
        return "decrease"
    if any(word in lowered for word in _INCREASE_WORDS):
        return "increase"
    return "unknown"


def is_approximate(text: str) -> bool:
    lowered = clean_text(text).casefold()
    return any(word in lowered for word in _APPROX_WORDS)


def signed_change(value: Decimal | None, direction: str) -> Decimal | None:
    """Express a magnitude plus a direction word as one signed number.

    A claim says "reduced by 40.2%"; the table prints "(40.2) %". Both mean
    -40.2, and only the signed form can be compared without re-reading prose.
    """
    if value is None:
        return None
    if direction == "decrease":
        return -abs(value)
    if direction == "increase":
        return abs(value)
    return value


def values_agree(
    claimed: Decimal, observed: Decimal, *, approximate: bool
) -> bool:
    """Exact displayed agreement, or agreement at the claim's own precision.

    A hedged claim is compared at the precision it was written to: "about 40%"
    is satisfied by 40.2%, because rounded to whole percent that *is* 40. A
    fixed percentage tolerance is deliberately not used -- 5% of a small figure
    and 5% of a large one are different claims, and neither is what the writer
    said. "about 21.3" therefore still excludes 21.5.
    """
    if claimed == observed:
        return True
    if not approximate:
        return False
    try:
        return observed.quantize(claimed, rounding=ROUND_HALF_UP) == claimed
    except (InvalidOperation, ValueError):
        # A figure too large to express at the claim's precision is not a near
        # miss; refusing to compare is the honest answer.
        return False


def scope_markers(text: str) -> frozenset[str]:
    """Scope tokens that must not conflict between a claim and a fact.

    "Scope 1 & 2" and "Scope 3" are different emissions boundaries; comparing
    a value across them produces a confidently wrong verdict.
    """
    lowered = clean_text(text).casefold()
    markers = {f"scope{n}" for n in _SCOPE_RE.findall(lowered)}
    if re.search(r"\bscope\s*1\s*(&|and|\+|,)\s*2\b", lowered):
        markers |= {"scope1", "scope2"}
    for phrase in ("market-based", "location-based", "gross", "net", "intensity"):
        if phrase in lowered:
            markers.add(phrase.replace("-", ""))
    return frozenset(markers)


def scopes_comparable(claim_text: str, fact_text: str) -> bool:
    """Whether two scope descriptions describe the same boundary.

    Fails closed: an empty claim scope is not treated as "matches anything",
    because a vague claim must land on `insufficient`, not on the nearest
    number that happens to exist.
    """
    claim_markers = scope_markers(claim_text)
    fact_markers = scope_markers(fact_text)
    if not claim_markers or not fact_markers:
        return False
    return claim_markers <= fact_markers or fact_markers <= claim_markers


def content_tokens(text: str) -> frozenset[str]:
    """Lowercase word tokens with stopwords dropped, for overlap scoring."""
    words = re.findall(r"[a-z0-9]+", normalize_for_match(text))
    return frozenset(w for w in words if len(w) > 2 and w not in _STOPWORDS)


_STOPWORDS = frozenset(
    """the and for with from that this than into over under between per was were
    are has have had been being its our their they them then when what which
    while about above below more most less least also such very can will
    would could should may might must not but out off""".split()
)


__all__ = [
    "all_periods",
    "all_years",
    "clean_text",
    "content_tokens",
    "contains_quote",
    "detect_direction",
    "is_approximate",
    "known_unit",
    "normalize_bbox",
    "normalize_for_match",
    "normalize_period",
    "normalize_unit",
    "parse_period",
    "parse_value",
    "region_from",
    "scope_markers",
    "scopes_comparable",
    "signed_change",
    "union_bbox",
    "unit_conversion",
    "unit_of",
    "values_agree",
]

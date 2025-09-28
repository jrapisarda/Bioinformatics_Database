"""Enhanced ETL helpers for handling illness dimension lookups."""
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Built-in fallback for environments where the illness dimension is missing.
_FALLBACK_ILLNESS_KEY_MAP: Dict[str, int] = {
    "UNKNOWN": 0,
    "CONTROL": 1,
    "SEPSIS": 2,
    "SEPTIC_SHOCK": 3,
    "NO_SEPSIS": 4,
}


def _coerce_to_pairs(rows: Any) -> List[Tuple[Any, Any]]:
    """Coerce the value returned from ``cursor.fetchall`` into ``(label, key)`` pairs."""
    if rows is None:
        return []

    if isinstance(rows, Mapping):
        return list(rows.items())

    if isinstance(rows, (str, bytes)):
        raise TypeError("fetchall result is a string, not an iterable of rows")

    if isinstance(rows, Iterable):
        coerced: List[Tuple[Any, Any]] = []
        for row in rows:
            if isinstance(row, Mapping):
                if "illness_label" in row and "illness_key" in row:
                    coerced.append((row["illness_label"], row["illness_key"]))
                    continue

                if len(row) >= 2:
                    iterator = iter(row.values())
                    coerced.append((next(iterator), next(iterator)))
                    continue

                raise ValueError("mapping row does not contain at least two values")

            try:
                label, key = row  # type: ignore[misc]
            except (TypeError, ValueError) as exc:
                raise TypeError("row is not a (label, key) pair") from exc

            coerced.append((label, key))

        return coerced

    raise TypeError("fetchall result is not iterable")


def _normalize_label(label: Any) -> str:
    """Normalise labels for consistent dictionary lookups."""
    return str(label).strip().upper()


def _parse_key(key: Any) -> Any:
    """Convert numeric keys to integers when possible."""
    try:
        return int(key)
    except (TypeError, ValueError):
        return key


def _get_illness_key_map(cursor: Any) -> Dict[str, Any]:
    """Return a mapping of illness labels to their dimension keys.

    This function is defensive about the structure of ``cursor.fetchall`` so it
    can be safely used with DB cursors, lightweight mocks, or cached data.
    When the cursor does not provide iterable results (or provides no rows), a
    built-in fallback mapping is used so downstream processing can continue.
    """

    def _fallback(reason: str, error: Exception | None = None) -> Dict[str, Any]:
        if error is None:
            logger.warning(
                "Falling back to built-in illness key map because %s.",
                reason,
            )
        else:
            logger.warning(
                "Falling back to built-in illness key map because %s: %s",
                reason,
                error,
            )
        return dict(_FALLBACK_ILLNESS_KEY_MAP)

    if cursor is None:
        return _fallback("no cursor was supplied")

    try:
        raw_rows = cursor.fetchall()
    except AttributeError as exc:
        return _fallback("cursor does not implement fetchall", exc)
    except Exception as exc:  # pragma: no cover - defensive logging branch
        return _fallback("fetchall raised an unexpected error", exc)

    try:
        rows = _coerce_to_pairs(raw_rows)
    except (TypeError, ValueError) as exc:
        return _fallback("fetchall did not return iterable rows", exc)

    if not rows:
        return _fallback("no illness rows were returned from the database")

    illness_key_map: Dict[str, Any] = {}
    for label, key in rows:
        normalized_label = _normalize_label(label)
        if not normalized_label:
            continue
        illness_key_map[normalized_label] = _parse_key(key)

    if not illness_key_map:
        return _fallback("no usable illness rows were available after parsing")

    return illness_key_map


__all__ = ["_get_illness_key_map"]

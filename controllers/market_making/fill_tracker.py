"""Cross-controller fill tracker for the bbo_peg buy and sell sides.

The buy and sell controllers are normally independent (no inventory
coupling — see the bbo_peg_sell module docstring). When the buy side
runs with max_fills > 1 it can accumulate base inventory faster than
the sell side can clear it. This module provides a per-trading-pair
shared counter so the buy controller can refuse to keep accumulating
when its lead over sells exceeds a configurable threshold.

State is in-memory only — resets to zero on process restart, matching
the `_fill_count` semantics on the controllers themselves. Any prior
inventory on the account is invisible to this tracker; the safe usage
pattern is to start each process with an inventory snapshot the user
is comfortable with.

Thread/concurrency note: Hummingbot V2 controllers run in the same
asyncio event loop, so no lock is needed. If a future change runs
controllers in worker threads, wrap the counter mutations in a Lock.
"""
from __future__ import annotations

from typing import Dict


# {trading_pair: {"buys": N, "sells": M}}. Keys created lazily on first
# record_*; missing keys treated as zero by the readers.
_counters: Dict[str, Dict[str, int]] = {}


def record_buy_fill(trading_pair: str) -> None:
    """Increment the buy fill counter for the given pair by one. Called by
    the buy controller's _update_fill_latch the first time it counts a
    newly-filled executor (deduped by executor id at the controller layer,
    so this is called exactly once per real fill).
    """
    counters = _counters.setdefault(trading_pair, {"buys": 0, "sells": 0})
    counters["buys"] += 1


def record_sell_fill(trading_pair: str) -> None:
    """Increment the sell fill counter for the given pair by one. Mirror
    of record_buy_fill on the sell side.
    """
    counters = _counters.setdefault(trading_pair, {"buys": 0, "sells": 0})
    counters["sells"] += 1


def buy_lead(trading_pair: str) -> int:
    """Return (buys - sells) for the pair. Positive means the buy side has
    accumulated inventory; zero means balanced; negative means the sell
    side ran ahead (possible if pre-existing inventory was being sold off).
    """
    counters = _counters.get(trading_pair)
    if counters is None:
        return 0
    return counters["buys"] - counters["sells"]


def buys_filled(trading_pair: str) -> int:
    """Read-only accessor for the buy count. Useful for diagnostics."""
    return _counters.get(trading_pair, {}).get("buys", 0)


def sells_filled(trading_pair: str) -> int:
    """Read-only accessor for the sell count. Useful for diagnostics."""
    return _counters.get(trading_pair, {}).get("sells", 0)


def reset(trading_pair: str) -> None:
    """Reset the counters for a single pair. Intended for tests; do not
    call from production code (the in-memory state is meant to live for
    the full process lifetime).
    """
    _counters.pop(trading_pair, None)


def reset_all() -> None:
    """Clear all counters. Intended for tests."""
    _counters.clear()

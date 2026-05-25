"""HTX pair screener for the BBO peg algorithm.

Ranks USDT pairs by suitability for the cancel-debounced peg-inside-BBO strategy.
Suitability = small but meaningful tick-as-percent-of-price, natural spread
wide enough that the 2% anti-spoof gate has signal, daily volume in the
mid-cap retail range (deep enough to fill, thin enough to avoid pro MMs).

No API keys needed — uses HTX's public endpoints. Run as:
    .venv/bin/python scripts/htx_pair_screener.py

Output: ranked candidate list with reasoning per row. Verify any pick against
the live order book and a paper run before deploying capital.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional

import requests


HTX_TICKERS_URL = "https://api.huobi.pro/market/tickers"
HTX_SYMBOLS_URL = "https://api.huobi.pro/v1/common/symbols"
HTX_DEPTH_URL = "https://api.huobi.pro/market/depth"

# Screening thresholds, calibrated for HTX 0.2% maker fee (no discounts):
# round-trip fees = 0.4%, adverse selection drag ~0.2-0.5%, so we need
# natural market spread >= ~1% just to break even and more like 1.5%+ to
# be reliably profitable. If your effective fee tier changes (HT discount,
# VIP, zero-fee promo), drop MIN_SPREAD_PCT accordingly.
MAKER_FEE_PCT = Decimal("0.002")                       # per side
ROUND_TRIP_FEE_PCT = MAKER_FEE_PCT * 2                 # 0.4%
ADVERSE_SELECTION_BUFFER_PCT = Decimal("0.003")        # 0.3% rule of thumb
BREAK_EVEN_SPREAD_PCT = ROUND_TRIP_FEE_PCT + ADVERSE_SELECTION_BUFFER_PCT  # 0.7%

MIN_PRICE = Decimal("0.001")
MAX_PRICE = Decimal("20")
MIN_VOLUME_USDT = Decimal("500000")        # 500k daily — thin enough to avoid pro HFTs
MAX_VOLUME_USDT = Decimal("30000000")      # 30M daily — fat enough to fill
MIN_SPREAD_PCT = BREAK_EVEN_SPREAD_PCT     # 0.7% — must cover fees + adverse selection
MAX_SPREAD_PCT = Decimal("0.05")           # 5% — not totally broken
MIN_TICK_PCT = Decimal("0.0001")           # 0.01% — peg edge is real
MAX_TICK_PCT = Decimal("0.01")             # 1% — tick isn't dominating price


@dataclass
class Candidate:
    symbol: str           # e.g. "xnousdt"
    base: str             # e.g. "xno"
    price: Decimal        # last trade price
    volume_usdt: Decimal  # 24h quote volume
    tick: Optional[Decimal] = None
    spread_pct: Optional[Decimal] = None
    tick_pct: Optional[Decimal] = None
    score: Optional[Decimal] = None


def fetch_tickers() -> List[dict]:
    r = requests.get(HTX_TICKERS_URL, timeout=10)
    r.raise_for_status()
    return r.json()["data"]


def fetch_symbols() -> dict:
    """Returns {symbol: {price_precision, amount_precision, min_order_value, ...}}.
    price_precision tells us tick size: tick = 10**-price_precision.
    """
    r = requests.get(HTX_SYMBOLS_URL, timeout=10)
    r.raise_for_status()
    out = {}
    for s in r.json()["data"]:
        out[s["symbol"]] = s
    return out


def fetch_depth(symbol: str) -> Optional[dict]:
    try:
        r = requests.get(
            HTX_DEPTH_URL, params={"symbol": symbol, "type": "step0"}, timeout=5
        )
        r.raise_for_status()
        return r.json().get("tick")
    except Exception:
        return None


def initial_screen(tickers: List[dict], symbols: dict) -> List[Candidate]:
    """First pass: filter by 24h ticker (price, volume) — no per-pair HTTP."""
    out: List[Candidate] = []
    for t in tickers:
        sym = t["symbol"]
        if not sym.endswith("usdt"):
            continue
        if sym not in symbols:
            continue
        if symbols[sym].get("state") != "online":
            continue
        try:
            price = Decimal(str(t["close"]))
            vol = Decimal(str(t.get("vol", 0)))  # 24h quote (USDT) volume
        except Exception:
            continue
        if not (MIN_PRICE <= price <= MAX_PRICE):
            continue
        if not (MIN_VOLUME_USDT <= vol <= MAX_VOLUME_USDT):
            continue
        base = sym[:-4]
        # Skip leveraged tokens (HTX has 3L/3S/5L/5S suffixed pairs).
        if any(base.endswith(x) for x in ("3l", "3s", "5l", "5s")):
            continue
        out.append(Candidate(symbol=sym, base=base, price=price, volume_usdt=vol))
    return out


def enrich_with_book(candidates: List[Candidate], symbols: dict) -> List[Candidate]:
    """Second pass: per-pair order book fetch to get live spread + tick.
    Rate-limited (HTX public: 100 req / 10s); we sleep 0.1s between calls.
    """
    enriched: List[Candidate] = []
    for c in candidates:
        sym_info = symbols[c.symbol]
        price_precision = int(sym_info.get("price-precision", 8))
        tick = Decimal(10) ** -price_precision
        c.tick = tick
        c.tick_pct = tick / c.price if c.price > 0 else None

        book = fetch_depth(c.symbol)
        time.sleep(0.1)  # rate limit
        if not book:
            continue
        try:
            best_bid = Decimal(str(book["bids"][0][0]))
            best_ask = Decimal(str(book["asks"][0][0]))
        except (KeyError, IndexError):
            continue
        if best_bid <= 0:
            continue
        spread_pct = (best_ask - best_bid) / best_bid
        c.spread_pct = spread_pct
        enriched.append(c)
    return enriched


def passes_book_filters(c: Candidate) -> bool:
    if c.tick_pct is None or c.spread_pct is None:
        return False
    if not (MIN_TICK_PCT <= c.tick_pct <= MAX_TICK_PCT):
        return False
    if not (MIN_SPREAD_PCT <= c.spread_pct <= MAX_SPREAD_PCT):
        return False
    return True


def score(c: Candidate) -> Decimal:
    """Composite score: higher = better candidate.

    Reward (in roughly decreasing order):
      - Expected net profit per round-trip (captured spread - fees -
        adverse selection). Peaks at ~2% market spread under 0.2% maker.
      - Tick small relative to price (our 1-tick edge is meaningful)
      - Volume in the 1-10M range (fill-able but not pro-MM-dominated)
    """
    assert c.spread_pct is not None and c.tick_pct is not None
    # Expected net per round-trip in % of price. Negative means unprofitable.
    expected_net = c.spread_pct - ROUND_TRIP_FEE_PCT - ADVERSE_SELECTION_BUFFER_PCT
    # Quadratic peak around 2% market spread (1.3% net after costs); falls
    # off for very wide spreads (those imply illiquidity / volatility).
    spread_term = max(Decimal("0"), Decimal("1") - ((c.spread_pct - Decimal("0.02")) ** 2) * Decimal("400"))
    # Inverse of tick_pct, normalized: smaller tick % = better.
    tick_term = max(Decimal("0"), Decimal("1") - c.tick_pct * Decimal("100"))
    # Volume in 1-10M sweet spot.
    log_vol = c.volume_usdt.ln() if c.volume_usdt > 0 else Decimal("0")
    log_target = Decimal("3000000").ln()
    vol_term = max(Decimal("0"), Decimal("1") - abs(log_vol - log_target) / Decimal("3"))
    # Hard penalty if net negative — should never rank highly.
    if expected_net <= 0:
        return Decimal("0")
    return spread_term + tick_term + vol_term + expected_net * Decimal("10")


def main():
    print("Fetching HTX tickers + symbol metadata...")
    tickers = fetch_tickers()
    symbols = fetch_symbols()

    candidates = initial_screen(tickers, symbols)
    print(f"  {len(candidates)} pairs survived price/volume screen "
          f"({MIN_VOLUME_USDT}-{MAX_VOLUME_USDT} USDT 24h vol)")

    print(f"Fetching order books for top-volume {min(len(candidates), 80)} candidates...")
    # Hit the top N by volume to keep the rate-limit budget reasonable.
    candidates.sort(key=lambda c: c.volume_usdt, reverse=True)
    candidates = enrich_with_book(candidates[:80], symbols)

    survivors = [c for c in candidates if passes_book_filters(c)]
    for c in survivors:
        c.score = score(c)
    survivors.sort(key=lambda c: c.score or Decimal("0"), reverse=True)

    print()
    print(f"Break-even spread: {float(BREAK_EVEN_SPREAD_PCT) * 100:.2f}% "
          f"({float(ROUND_TRIP_FEE_PCT) * 100:.2f}% fees + "
          f"{float(ADVERSE_SELECTION_BUFFER_PCT) * 100:.2f}% adverse-selection buffer)")
    print()
    print(f"{'Rank':<5} {'Pair':<12} {'Price':>10} {'24h Vol $':>14} "
          f"{'Spread%':>9} {'NetExp%':>9} {'Tick%':>8} {'Score':>7}")
    print("-" * 85)
    for rank, c in enumerate(survivors[:20], start=1):
        net_exp = (c.spread_pct or Decimal("0")) - ROUND_TRIP_FEE_PCT - ADVERSE_SELECTION_BUFFER_PCT
        print(
            f"{rank:<5} {c.base.upper() + '/USDT':<12} "
            f"{float(c.price):>10.6f} "
            f"{float(c.volume_usdt):>14,.0f} "
            f"{float(c.spread_pct or 0) * 100:>8.3f}% "
            f"{float(net_exp) * 100:>8.3f}% "
            f"{float(c.tick_pct or 0) * 100:>7.4f}% "
            f"{float(c.score or 0):>7.3f}"
        )
    print()
    print("NetExp% = expected net per round-trip after fees + adverse-selection buffer.")
    print("Sanity check: verify any pick has the leapfrog-spoof pattern in its")
    print("order book before deploying. Paper-test for 1-2 hours first.")


if __name__ == "__main__":
    main()

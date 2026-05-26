"""Cross-exchange arbitrage screener.

Pulls best bid + best ask + 24h quote volume from HTX, Binance, OKX,
Bybit, MEXC via their public APIs (no auth needed). Normalizes pair
symbols, finds tokens listed on >= 2 exchanges, and reports the cases
where the highest bid (sell into) exceeds the lowest ask (buy at) by
more than the round-trip taker-fee floor (default 0.4%) + slippage
buffer (0.2%).

This is the CORRECT arb test: an opportunity exists only when
    max_bid_across_exchanges > min_ask_across_exchanges
because that's the only way you can simultaneously
    BUY at the lowest ask (taker, paying the offer) and
    SELL into the highest bid (taker, hitting the bid)
and lock in a profit. Using last-trade prices (an earlier version of
this script) overstated apparent opportunities since last-trade often
sits between bid and ask.

Read-only research tool — no trading. Use this to identify candidates,
then evaluate whether the gap is real (vs stale tickers, fake liquidity,
chain mismatches, or symbol collisions like "BLUE" being a different
project on each exchange).

Limitations:
- Symbol collisions: some tokens share tickers across chains. The screener
  can't tell — sanity-check any pick by visiting both exchange listings.
- Best-bid/ask reflects TOP of book only; depth at those prices is not
  measured. A real fill at the top quote may consume only a small amount.
- Deposit/withdraw status checked only for HTX (its currencies endpoint
  is public). Binance, OKX, Bybit, MEXC require API keys to check. Pairs
  with HTX suspensions are dropped; for non-HTX legs you must verify
  manually that the asset can be moved in/out.

Run with:
    .venv/bin/python scripts/cross_exchange_arb_screener.py
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple

import requests


# ---- Tunable thresholds -----------------------------------------------------

# Fee assumption (round-trip): one taker on HTX (0.2%) + one taker on the
# counterparty exchange (typically 0.1%). Total ~0.3% in the best case,
# higher for HTX-only pairs. Plus a 0.2% buffer for slippage on either side.
ROUND_TRIP_FEE_PCT = Decimal("0.004")   # 0.4% pessimistic
SLIPPAGE_BUFFER_PCT = Decimal("0.002")  # 0.2%
BREAK_EVEN_PCT = ROUND_TRIP_FEE_PCT + SLIPPAGE_BUFFER_PCT  # 0.6%

# Don't bother with thin tickers — arbing them is fictional.
MIN_VOLUME_USDT = Decimal("100000")  # 100k 24h on EACH leg

# Cap on max gap to surface as a likely-real arb. Anything above this is
# almost certainly a symbol collision (e.g. "ELON" on Bybit is Dogelon Mars
# while "ELON" on HTX is a different project entirely), a token rebrand,
# wrapped vs native confusion, or a stale ticker. The screener still
# computes these but puts them in a separate "suspicious" bucket so they
# don't bury the real opportunities.
MAX_REALISTIC_GAP_PCT = Decimal("0.10")  # 10%

# Don't surface stablecoin pegs (USDC/USDT, BUSD/USDT, etc.) — the "arb"
# is the depeg risk, not a trading edge.
STABLECOIN_BASES = {"USDC", "USDT", "BUSD", "DAI", "TUSD", "USDD", "FDUSD", "USDE", "PYUSD"}

# Strip leveraged-token suffixes that some exchanges list (BTC3L, BTC3S etc.).
LEVERAGED_SUFFIXES = ("3L", "3S", "5L", "5S", "UP", "DOWN", "BULL", "BEAR")

# Binance applies EU MiCA / account-tier trading restrictions to certain pairs.
# The public /api/v3/exchangeInfo endpoint reports these as status=TRADING
# with SPOT permission — there's NO API field that distinguishes them from
# universally-tradeable pairs. Restrictions are enforced at the trading layer,
# not the data layer. So an EU user sees "redirect to BTC" when opening the
# trade UI, but the screener happily reports the bid/ask as if tradeable.
#
# Workaround: maintain a manual blocklist of Binance pairs known to be
# inaccessible. Verify by opening https://www.binance.com/en/trade/<BASE>_USDT
# in your browser — if it redirects to BTC, append the base symbol below.
#
# The list below is seeded with pairs verified by the user. Likely incomplete;
# expand as new restrictions are encountered.
BINANCE_EXCLUDED_BASES: set = {
    "DGB",   # DigiByte — verified redirect 2026-05-26
    "DCR",   # Decred — same legacy-PoW profile, likely same restriction
    "MMT",   # small-cap, conservative exclusion
    "USTC",  # Terra Classic USD — depegged stablecoin reference
}


# ---- Exchange-specific fetchers --------------------------------------------

@dataclass
class Ticker:
    base: str            # e.g. "BTC"
    bid: Decimal         # best bid (highest someone will buy at)
    ask: Decimal         # best ask (lowest someone will sell at)
    volume_usdt: Decimal # 24h quote volume


def _normalize_base(base: str) -> Optional[str]:
    base = base.upper().strip()
    if not base or base in STABLECOIN_BASES:
        return None
    if any(base.endswith(s) for s in LEVERAGED_SUFFIXES):
        return None
    return base


def _add(out: Dict[str, Ticker], base: str, bid: Decimal, ask: Decimal, vol: Decimal) -> None:
    """Insert a ticker if it passes the basic sanity filters: positive bid/ask,
    non-crossed single-exchange book (bid <= ask), and minimum 24h volume.
    """
    if bid <= 0 or ask <= 0 or bid > ask:
        return
    if vol < MIN_VOLUME_USDT:
        return
    out[base] = Ticker(base, bid, ask, vol)


def _fetch_htx_movable_bases() -> Set[str]:
    """Return the set of base asset symbols where HTX has at least one chain
    with BOTH deposit AND withdraw currently allowed. HTX is unique among
    the five exchanges in exposing this via a public endpoint
    (/v2/reference/currencies); other exchanges require API keys.

    Pairs whose base is NOT in this set should be excluded from any arb
    involving HTX, because even if the cross-exchange price gap exists, you
    cannot move the asset in or out of HTX to capture it (and even with
    pre-positioned inventory, eventual rebalancing breaks). HTX reports
    statuses like "allowed", "prohibited", "delayed" per chain — we require
    "allowed" on both directions for at least one chain.
    """
    try:
        r = requests.get("https://api.huobi.pro/v2/reference/currencies", timeout=15)
        r.raise_for_status()
    except Exception:
        # If the status endpoint is down, return None-like (empty) → fetch_htx
        # treats every base as un-movable and HTX legs vanish from arb output.
        # Safer than wrongly assuming everything is fine.
        return set()
    out: Set[str] = set()
    for c in r.json().get("data", []):
        base = c.get("currency", "").upper()
        if not base:
            continue
        for chain in c.get("chains", []):
            if (chain.get("depositStatus") == "allowed"
                    and chain.get("withdrawStatus") == "allowed"):
                out.add(base)
                break
    return out


def fetch_htx() -> Dict[str, Ticker]:
    """HTX ticker fields: symbol (e.g. "btcusdt"), bid, ask, vol (24h QUOTE vol USDT).

    Additionally filters out bases where HTX has currently suspended deposit
    or withdrawal — those pairs can't participate in real arb even if the
    ticker shows a gap (asset is trapped on HTX). See _fetch_htx_movable_bases.
    """
    movable = _fetch_htx_movable_bases()
    r = requests.get("https://api.huobi.pro/market/tickers", timeout=15)
    r.raise_for_status()
    out: Dict[str, Ticker] = {}
    for t in r.json()["data"]:
        sym = t["symbol"]
        if not sym.endswith("usdt"):
            continue
        base = _normalize_base(sym[:-4])
        if base is None or base not in movable:
            continue
        try:
            _add(
                out,
                base,
                Decimal(str(t.get("bid", 0))),
                Decimal(str(t.get("ask", 0))),
                Decimal(str(t.get("vol", 0))),
            )
        except Exception:
            continue
    return out


def fetch_binance() -> Dict[str, Ticker]:
    """Binance 24hr fields include bidPrice, askPrice, quoteVolume.

    Skips bases in BINANCE_EXCLUDED_BASES (EU MiCA / account-tier restricted
    pairs that the public API can't distinguish from tradeable ones).
    """
    r = requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=15)
    r.raise_for_status()
    out: Dict[str, Ticker] = {}
    for t in r.json():
        sym = t["symbol"]
        if not sym.endswith("USDT"):
            continue
        base = _normalize_base(sym[:-4])
        if base is None or base in BINANCE_EXCLUDED_BASES:
            continue
        try:
            _add(
                out,
                base,
                Decimal(t.get("bidPrice", "0")),
                Decimal(t.get("askPrice", "0")),
                Decimal(t.get("quoteVolume", "0")),
            )
        except Exception:
            continue
    return out


def fetch_okx() -> Dict[str, Ticker]:
    """OKX fields: instId, bidPx, askPx, volCcy24h."""
    r = requests.get(
        "https://www.okx.com/api/v5/market/tickers",
        params={"instType": "SPOT"},
        timeout=15,
    )
    r.raise_for_status()
    out: Dict[str, Ticker] = {}
    for t in r.json().get("data", []):
        inst = t["instId"]
        if not inst.endswith("-USDT"):
            continue
        base = _normalize_base(inst[:-5])
        if base is None:
            continue
        try:
            _add(
                out,
                base,
                Decimal(t.get("bidPx", "0") or "0"),
                Decimal(t.get("askPx", "0") or "0"),
                Decimal(t.get("volCcy24h", "0") or "0"),
            )
        except Exception:
            continue
    return out


def fetch_bybit() -> Dict[str, Ticker]:
    """Bybit v5 spot fields: bid1Price, ask1Price, turnover24h."""
    r = requests.get(
        "https://api.bybit.com/v5/market/tickers",
        params={"category": "spot"},
        timeout=15,
    )
    r.raise_for_status()
    out: Dict[str, Ticker] = {}
    for t in r.json().get("result", {}).get("list", []):
        sym = t["symbol"]
        if not sym.endswith("USDT"):
            continue
        base = _normalize_base(sym[:-4])
        if base is None:
            continue
        try:
            _add(
                out,
                base,
                Decimal(t.get("bid1Price", "0") or "0"),
                Decimal(t.get("ask1Price", "0") or "0"),
                Decimal(t.get("turnover24h", "0") or "0"),
            )
        except Exception:
            continue
    return out


def fetch_mexc() -> Dict[str, Ticker]:
    """MEXC fields: symbol, bidPrice, askPrice, quoteVolume."""
    r = requests.get("https://api.mexc.com/api/v3/ticker/24hr", timeout=15)
    r.raise_for_status()
    out: Dict[str, Ticker] = {}
    for t in r.json():
        sym = t["symbol"]
        if not sym.endswith("USDT"):
            continue
        base = _normalize_base(sym[:-4])
        if base is None:
            continue
        try:
            _add(
                out,
                base,
                Decimal(t.get("bidPrice", "0") or "0"),
                Decimal(t.get("askPrice", "0") or "0"),
                Decimal(t.get("quoteVolume", "0") or "0"),
            )
        except Exception:
            continue
    return out


FETCHERS = {
    "HTX": fetch_htx,
    "BINANCE": fetch_binance,
    "OKX": fetch_okx,
    "BYBIT": fetch_bybit,
    "MEXC": fetch_mexc,
}


# ---- Arb computation -------------------------------------------------------

@dataclass
class ArbOpportunity:
    base: str
    buy_exchange: str       # exchange with the lowest ask
    buy_price: Decimal      # the ask we'd take (BUY price)
    buy_volume: Decimal     # 24h volume on that exchange
    sell_exchange: str      # exchange with the highest bid
    sell_price: Decimal     # the bid we'd take (SELL price)
    sell_volume: Decimal    # 24h volume on that exchange
    gross_gap_pct: Decimal  # (highest_bid - lowest_ask) / lowest_ask
    net_pct: Decimal        # gross - BREAK_EVEN_PCT
    min_volume: Decimal     # min of the two 24h volumes (USDT)


def find_arbs(
    tickers_by_exchange: Dict[str, Dict[str, Ticker]],
) -> List[ArbOpportunity]:
    """For each token base listed on >= 2 exchanges, check whether the
    highest bid (across all exchanges) exceeds the lowest ask. A real arb
    exists iff max_bid > min_ask: you BUY at that lowest ask (taker) and
    SIMULTANEOUSLY SELL into that highest bid (also taker), locking in
    (max_bid - min_ask) before fees. We require the post-fee gap to clear
    BREAK_EVEN_PCT for surface.
    """
    all_bases: set = set()
    for tickers in tickers_by_exchange.values():
        all_bases.update(tickers.keys())

    arbs: List[ArbOpportunity] = []
    for base in all_bases:
        listings = [
            (exch, t) for exch, tickers in tickers_by_exchange.items()
            if (t := tickers.get(base)) is not None
        ]
        if len(listings) < 2:
            continue
        # Lowest ASK across exchanges → cheapest place to BUY (we cross
        # someone's offer). Highest BID across exchanges → most expensive
        # place to SELL (we hit someone's bid).
        buy_exch, buy_t = min(listings, key=lambda x: x[1].ask)
        sell_exch, sell_t = max(listings, key=lambda x: x[1].bid)
        if buy_exch == sell_exch:
            # The same exchange has both the lowest ask AND the highest bid
            # — that's a normal single-venue book, no cross-venue arb.
            continue
        # CRITICAL: arb only exists if highest bid > lowest ask. If
        # max_bid <= min_ask, the books are NOT crossed across venues —
        # any apparent "gap" in mid-prices is just spread, not arb.
        if sell_t.bid <= buy_t.ask:
            continue
        gross_gap = (sell_t.bid - buy_t.ask) / buy_t.ask
        if gross_gap < BREAK_EVEN_PCT:
            continue
        arbs.append(ArbOpportunity(
            base=base,
            buy_exchange=buy_exch,
            buy_price=buy_t.ask,
            buy_volume=buy_t.volume_usdt,
            sell_exchange=sell_exch,
            sell_price=sell_t.bid,
            sell_volume=sell_t.volume_usdt,
            gross_gap_pct=gross_gap,
            net_pct=gross_gap - BREAK_EVEN_PCT,
            min_volume=min(buy_t.volume_usdt, sell_t.volume_usdt),
        ))
    return arbs


# ---- Main ------------------------------------------------------------------

def main():
    print("Fetching tickers from 5 exchanges in parallel...")
    tickers_by_exchange: Dict[str, Dict[str, Ticker]] = {}
    with ThreadPoolExecutor(max_workers=len(FETCHERS)) as pool:
        future_to_exchange = {
            pool.submit(fn): name for name, fn in FETCHERS.items()
        }
        for future, name in future_to_exchange.items():
            try:
                tickers_by_exchange[name] = future.result()
                print(f"  {name:8s}  {len(tickers_by_exchange[name])} USDT pairs")
            except Exception as e:
                print(f"  {name:8s}  FAILED: {type(e).__name__}: {e}")
                tickers_by_exchange[name] = {}

    print()
    print(
        f"Break-even gap: {float(BREAK_EVEN_PCT) * 100:.2f}% "
        f"({float(ROUND_TRIP_FEE_PCT) * 100:.2f}% fees + "
        f"{float(SLIPPAGE_BUFFER_PCT) * 100:.2f}% slippage)"
    )
    print(f"Min volume on each leg: ${MIN_VOLUME_USDT:,}")
    print()

    arbs = find_arbs(tickers_by_exchange)
    arbs.sort(key=lambda a: a.net_pct, reverse=True)

    realistic = [a for a in arbs if a.gross_gap_pct <= MAX_REALISTIC_GAP_PCT]
    suspicious = [a for a in arbs if a.gross_gap_pct > MAX_REALISTIC_GAP_PCT]

    print(f"Found {len(arbs)} above break-even: "
          f"{len(realistic)} likely real (gap ≤ {float(MAX_REALISTIC_GAP_PCT)*100:.0f}%), "
          f"{len(suspicious)} suspicious (likely symbol collision).")
    print()

    def _print_table(rows: List[ArbOpportunity], limit: int = 30) -> None:
        print(
            f"{'Rank':<5} {'Base':<10} {'Buy on':<8} {'Ask $':>14} "
            f"{'Sell on':<8} {'Bid $':>14} {'Gap%':>7} {'Net%':>7} "
            f"{'Min Vol $':>14}"
        )
        print("-" * 105)
        # Sort the realistic section ascending by gap (smallest first = closest
        # to typical arb spreads, more credible). Sort suspicious descending
        # so the user sees the most absurd cases first as a sanity check.
        for rank, a in enumerate(rows[:limit], start=1):
            print(
                f"{rank:<5} {a.base:<10} "
                f"{a.buy_exchange:<8} {float(a.buy_price):>14.8f} "
                f"{a.sell_exchange:<8} {float(a.sell_price):>14.8f} "
                f"{float(a.gross_gap_pct) * 100:>6.3f}% "
                f"{float(a.net_pct) * 100:>6.3f}% "
                f"{float(a.min_volume):>14,.0f}"
            )

    if realistic:
        # Sort by gap descending — biggest realistic gap = biggest profit potential.
        realistic.sort(key=lambda a: a.gross_gap_pct, reverse=True)
        print("=== LIKELY REAL ARB CANDIDATES ===")
        _print_table(realistic)
        print()

    if suspicious:
        print(f"=== SUSPICIOUS (gap > {float(MAX_REALISTIC_GAP_PCT)*100:.0f}%, likely symbol collision) ===")
        print("These almost certainly aren't tradeable — different projects sharing tickers,")
        print("token rebrands, or stale prices. Listed for sanity-check only.")
        suspicious.sort(key=lambda a: a.gross_gap_pct, reverse=True)
        _print_table(suspicious, limit=10)
        print()

    print("⚠ For any pick in 'LIKELY REAL':")
    print("  1. Verify it's the SAME token on both exchanges (check contract address).")
    print("  2. Check live order books — top-of-book depth may be very thin.")
    print("  3. Factor your actual taker fees (HTX 0.2%, others ~0.1%).")
    print("  4. Verify deposit AND withdrawal are open on the NON-HTX leg.")
    print("     HTX status is auto-checked; the others require manual confirmation")
    print("     (or providing API keys to extend this script).")


if __name__ == "__main__":
    main()

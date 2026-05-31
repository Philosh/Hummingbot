"""
NoClampOrderExecutor — drop-in replacement for upstream OrderExecutor that
removes the LIMIT_MAKER price clamp on both sides.

Upstream OrderExecutor.get_order_price() at
hummingbot/strategy_v2/executors/order_executor/order_executor.py:215-219 does:

    if execution_strategy == LIMIT_MAKER:
        if side == BUY:  return min(config.price, current_best_bid)
        if side == SELL: return max(config.price, current_best_ask)

Both clamps silently downgrade the intended price (target = external_best_bid + 1
tick for BUY, external_best_ask - 1 tick for SELL) toward the current top of book —
making a true "peg one tick inside the BBO" strategy structurally impossible.
Stuck loop on the BUY side:

    tick N:   want 0.4296, current_best_bid=0.4295 -> placed at 0.4295
    tick N+1: still want 0.4296 (external still 0.4295) -> placed at 0.4295 again
    ...

The legitimate LIMIT_MAKER safety check is "does this price cross the opposite
side?" — that lives in the controller (`_compute_target_price` already guards
against crossing). The upstream clamp is on the WRONG side of the book and serves
no real safety purpose for a peg-inside-the-BBO strategy.

This subclass returns `config.price` as-is for BUY and SELL LIMIT_MAKER, and falls
through to the parent for every other path (MARKET, LIMIT_CHASER, plain LIMIT).
"""

from decimal import Decimal

from hummingbot.core.data_type.common import TradeType
from hummingbot.strategy_v2.executors.order_executor.data_types import ExecutionStrategy
from hummingbot.strategy_v2.executors.order_executor.order_executor import OrderExecutor


class NoClampOrderExecutor(OrderExecutor):
    """OrderExecutor without the LIMIT_MAKER price clamp on either side.

    Behavior delta vs. parent:
      - BUY  + LIMIT_MAKER: returns `config.price` (parent clamps to min(price, best_bid))
      - SELL + LIMIT_MAKER: returns `config.price` (parent clamps to max(price, best_ask))
      - Every other case: delegates to super().get_order_price() unchanged
    """

    def get_order_price(self) -> Decimal:
        if self.config.execution_strategy == ExecutionStrategy.LIMIT_MAKER and (
            self.config.side == TradeType.BUY or self.config.side == TradeType.SELL
        ):
            # Validator on OrderExecutorConfig guarantees price is not None for
            # LIMIT_MAKER; parent's min()/max() rely on the same guarantee.
            return self.config.price  # type: ignore[return-value]  # ty: ignore[invalid-return-type]
        return super().get_order_price()

"""
NoClampOrderExecutor — drop-in replacement for upstream OrderExecutor that
removes the BUY LIMIT_MAKER price floor clamp.

Upstream OrderExecutor.get_order_price() at
hummingbot/strategy_v2/executors/order_executor/order_executor.py:215-219 does:

    if execution_strategy == LIMIT_MAKER and side == BUY:
        return min(config.price, current_best_bid)

That clamp silently downgrades our intended price (target = external_best_bid + 1
tick) to current_best_bid when we want to be the new top of book — making a true
"peg one tick above" strategy structurally impossible. Stuck loop:

    tick N:   want 0.4296, current_best_bid=0.4295 -> placed at 0.4295
    tick N+1: still want 0.4296 (external still 0.4295) -> placed at 0.4295 again
    ...

The legitimate LIMIT_MAKER safety check is "does this price cross the ask?" — that
lives in the controller (`_compute_target_price` already guards `candidate >= best_ask`).
The upstream clamp is on the WRONG side of the book and serves no real safety purpose
for a peg-above-bid strategy.

This subclass returns `config.price` as-is for BUY LIMIT_MAKER and falls through to
the parent for every other path (SELL LIMIT_MAKER, MARKET, LIMIT_CHASER, plain LIMIT).
"""

from decimal import Decimal

from hummingbot.core.data_type.common import TradeType
from hummingbot.strategy_v2.executors.order_executor.data_types import ExecutionStrategy
from hummingbot.strategy_v2.executors.order_executor.order_executor import OrderExecutor


class NoClampOrderExecutor(OrderExecutor):
    """OrderExecutor without the BUY LIMIT_MAKER min(price, best_bid) clamp.

    Behavior delta vs. parent:
      - BUY + LIMIT_MAKER: returns `config.price` (parent does min-clamp to best_bid)
      - Every other case: delegates to super().get_order_price() unchanged
    """

    def get_order_price(self) -> Decimal:
        if (
            self.config.execution_strategy == ExecutionStrategy.LIMIT_MAKER
            and self.config.side == TradeType.BUY
        ):
            # Validator on OrderExecutorConfig guarantees price is not None for
            # LIMIT_MAKER; parent's min() relies on the same guarantee.
            return self.config.price  # type: ignore[return-value]  # ty: ignore[invalid-return-type]
        return super().get_order_price()

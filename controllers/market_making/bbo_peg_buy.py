"""
BBO Peg Buy Controller — minimal V2 controller scaffold.

Pegs a single LIMIT_MAKER buy at external_best_bid + 1 tick, where
external_best_bid excludes our own resting order (so we never anchor
to ourselves). Re-quotes on any drift. One-shot: once any fill
occurs (partial or full), the controller stops creating new orders
for the lifetime of the process. No close-side, no triple barrier,
no rebalance. WS-fed via MarketDataProvider — zero HTTP polling at
the strategy layer.
"""

from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from pydantic import Field

from hummingbot.core.data_type.common import MarketDict, PriceType, TradeType
from hummingbot.strategy_v2.controllers.controller_base import (
    ControllerBase,
    ControllerConfigBase,
)
from hummingbot.strategy_v2.executors.order_executor.data_types import (
    ExecutionStrategy,
    OrderExecutorConfig,
)
from hummingbot.strategy_v2.models.executor_actions import (
    CreateExecutorAction,
    ExecutorAction,
    StopExecutorAction,
)
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo


class BBOPegBuyConfig(ControllerConfigBase):
    controller_type: str = "market_making"
    controller_name: str = "bbo_peg_buy"

    connector_name: str = Field(default="htx")
    trading_pair: str = Field(default="XNO-USDT")
    update_interval: float = Field(
        default=0.5,
        description="Seconds between controller ticks. Lower = faster reaction, more REST traffic.",
    )
    min_spread_pct: Decimal = Field(
        default=Decimal("0.02"),
        description="Anti-spoof gate. Refuse to quote when (best_ask - external_best_bid) / external_best_bid < this. Default 0.02 = 2%.",
    )

    def update_markets(self, markets: MarketDict) -> MarketDict:
        # Upstream's add_or_update signature mistypes *args as the set type
        # itself instead of a set element; mirrors MarketMakingControllerConfigBase.
        return markets.add_or_update(self.connector_name, self.trading_pair)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


class BBOPegBuyController(ControllerBase):
    config: BBOPegBuyConfig

    def __init__(self, config: BBOPegBuyConfig, *args, **kwargs):
        kwargs.setdefault("update_interval", config.update_interval)
        super().__init__(config, *args, **kwargs)
        self.config = config
        # One-shot latch. Set on first detected fill (partial or full).
        # Never reset within a process lifetime.
        self._has_filled: bool = False
        # Cache of the last logged external_best_bid value, so we only emit
        # a diagnostic line when the chosen price actually changes.
        self._last_logged_external_best_bid: Optional[Decimal] = None
        # Cache of the anti-spoof gate's last state, so we only log on
        # blocked <-> cleared transitions. False = not currently gated.
        self._gate_blocked: bool = False

    async def update_processed_data(self):
        rules = self.market_data_provider.get_trading_rules(
            self.config.connector_name, self.config.trading_pair
        )
        tick: Decimal = rules.min_price_increment

        best_ask: Decimal = self.market_data_provider.get_price_by_type(
            self.config.connector_name, self.config.trading_pair, PriceType.BestAsk
        )

        external_best_bid = self._external_best_bid()
        target_price = self._compute_target_price(external_best_bid, tick, best_ask)

        self.processed_data = {
            "tick": tick,
            "external_best_bid": external_best_bid,
            "best_ask": best_ask,
            "target_price": target_price,
        }

    def _external_best_bid(self) -> Optional[Decimal]:
        """
        Walk the bid book top-down and return the highest price level that
        has volume beyond what our own active executors are resting there.
        Prevents the controller from chasing its own quote.

        Emits a forensic INFO line whenever the chosen price changes, so we
        can later audit why a particular target_price was picked.
        """
        my_volume_by_price = self._compute_own_volume_by_price()
        result, top_levels = self._walk_bids_for_first_external(my_volume_by_price)
        self._log_external_bid_change(result, top_levels, my_volume_by_price)
        return result

    def _compute_own_volume_by_price(self) -> Dict[Decimal, Decimal]:
        """Aggregate our active executors' volume per price level.
        Filters out inactive executors and non-OrderExecutorConfig configs.
        """
        my_volume_by_price: Dict[Decimal, Decimal] = {}
        for e in self.executors_info:
            if not e.is_active:
                continue
            cfg = e.config
            if not isinstance(cfg, OrderExecutorConfig) or cfg.price is None:
                continue
            my_volume_by_price[cfg.price] = (
                my_volume_by_price.get(cfg.price, Decimal("0")) + cfg.amount
            )
        return my_volume_by_price

    def _walk_bids_for_first_external(
        self, my_volume_by_price: Dict[Decimal, Decimal]
    ) -> Tuple[Optional[Decimal], List[Tuple[float, float]]]:
        """Walk the top 10 bid levels and return:
        - the highest price where (book amount - our amount) > 0, else None
        - the top-5 levels as (price, amount) float tuples, for logging
        """
        order_book = self.market_data_provider.get_order_book(
            self.config.connector_name, self.config.trading_pair
        )
        top_levels: List[Tuple[float, float]] = []
        result: Optional[Decimal] = None
        for i, row in enumerate(order_book.bid_entries()):
            if i >= 10:
                break
            price = Decimal(str(row.price))
            amount = Decimal(str(row.amount))
            if i < 5:
                top_levels.append((float(price), float(amount)))
            if result is None:
                external_amount = amount - my_volume_by_price.get(price, Decimal("0"))
                if external_amount > 0:
                    result = price
        return result, top_levels

    def _log_external_bid_change(
        self,
        result: Optional[Decimal],
        top_levels: List[Tuple[float, float]],
        my_volume_by_price: Dict[Decimal, Decimal],
    ) -> None:
        """Emit a forensic INFO line only when the chosen external_best_bid
        changes from the previously logged value. Updates the cache after logging.
        """
        if result == self._last_logged_external_best_bid:
            return
        my_vol_str = {float(k): float(v) for k, v in my_volume_by_price.items()}
        self.logger().info(
            f"[bbo_peg] external_best_bid={result} "
            f"top5_bids={top_levels} my_volume={my_vol_str}"
        )
        self._last_logged_external_best_bid = result

    def _compute_target_price(
        self,
        external_best_bid: Optional[Decimal],
        tick: Decimal,
        best_ask: Decimal,
    ) -> Optional[Decimal]:
        """Peg one tick above the external best bid, quantized to the exchange's
        tick size. Returns None if there's no external bid to peg to, if the
        market spread is below the anti-spoof gate (config.min_spread_pct), or
        if the candidate would cross the ask (LIMIT_MAKER would be rejected).
        """
        if external_best_bid is None or external_best_bid <= 0:
            return None
        if not best_ask:
            return None
        spread_pct = self._compute_spread_pct(external_best_bid, best_ask)
        gate_blocked = spread_pct < self.config.min_spread_pct
        self._log_gate_state_change(
            gate_blocked, external_best_bid, best_ask, spread_pct
        )
        if gate_blocked:
            return None
        candidate = self.market_data_provider.quantize_order_price(
            self.config.connector_name,
            self.config.trading_pair,
            external_best_bid + tick,
        )
        if candidate >= best_ask:
            return None
        return candidate

    def _compute_spread_pct(
        self, external_best_bid: Decimal, best_ask: Decimal
    ) -> Decimal:
        """Return (best_ask - external_best_bid) / external_best_bid.

        This is the anti-spoof gate input: a spoofer that pushes the visible
        best bid up toward the ask compresses this value. Caller must ensure
        external_best_bid > 0.
        """
        return (best_ask - external_best_bid) / external_best_bid

    def _log_gate_state_change(
        self,
        gate_blocked: bool,
        external_best_bid: Decimal,
        best_ask: Decimal,
        spread_pct: Decimal,
    ) -> None:
        """Emit a forensic INFO line only when the anti-spoof gate transitions
        between blocked and cleared. Rate-limits to one line per state change
        so naturally-tight markets don't spam the log every tick.
        """
        if gate_blocked == self._gate_blocked:
            return
        state = "blocked" if gate_blocked else "cleared"
        self.logger().info(
            f"[bbo_peg] gate_{state} external_best_bid={external_best_bid} "
            f"best_ask={best_ask} spread={spread_pct} "
            f"min={self.config.min_spread_pct}"
        )
        self._gate_blocked = gate_blocked

    def determine_executor_actions(self) -> List[ExecutorAction]:
        self._update_fill_latch()

        target_price: Optional[Decimal] = self.processed_data.get("target_price")
        if target_price is None:
            # No valid target (no external bid, would cross ask, or anti-spoof
            # gate fired). Cancel any standing order rather than leaving a
            # quote exposed during the unsafe window. The cost is occasional
            # cancel-churn on transient market-data gaps; the alternative
            # leaves the gate toothless against the spoof pattern it exists
            # to defend against.
            actives = [e for e in self.executors_info if e.is_active]
            return self._build_stop_actions(actives)

        stale, in_tolerance = self._categorize_active_orders(target_price)
        actions: List[ExecutorAction] = self._build_stop_actions(stale)

        # One-shot: never create another order once we've had any fill.
        if self._has_filled:
            return actions

        if not in_tolerance:
            create = self._build_create_action(target_price)
            if create is not None:
                actions.append(create)

        return actions

    def _update_fill_latch(self) -> None:
        """Set _has_filled if any executor reports executed_amount_base > 0.
        Once set, never reset within a process lifetime.
        """
        if self._has_filled:
            return
        for e in self.executors_info:
            executed = (
                e.custom_info.get("executed_amount_base") if e.custom_info else None
            )
            if executed is not None and Decimal(str(executed)) > 0:
                self._has_filled = True
                return

    def _categorize_active_orders(
        self, target_price: Decimal
    ) -> Tuple[List[ExecutorInfo], List[ExecutorInfo]]:
        """Split active orders into (stale, in_tolerance) by exact price match.
        Filters out inactive executors and non-OrderExecutorConfig configs.
        """
        stale: List[ExecutorInfo] = []
        in_tolerance: List[ExecutorInfo] = []
        for e in self.executors_info:
            if not e.is_active:
                continue
            cfg = e.config
            if not isinstance(cfg, OrderExecutorConfig) or cfg.price is None:
                continue
            if cfg.price != target_price:
                stale.append(e)
            else:
                in_tolerance.append(e)
        return stale, in_tolerance

    def _build_stop_actions(
        self, executors: List[ExecutorInfo]
    ) -> List[ExecutorAction]:
        return [
            StopExecutorAction(controller_id=self.config.id, executor_id=e.id)
            for e in executors
        ]

    def _build_create_action(
        self, target_price: Decimal
    ) -> Optional[CreateExecutorAction]:
        """Build a LIMIT_MAKER buy at target_price.
        Returns None if quantized amount is zero (book/balance can't support an order).
        """
        amount = self.market_data_provider.quantize_order_amount(
            self.config.connector_name,
            self.config.trading_pair,
            self.config.total_amount_quote / target_price,
        )
        if amount <= 0:
            return None
        return CreateExecutorAction(
            controller_id=self.config.id,
            executor_config=OrderExecutorConfig(
                timestamp=self.market_data_provider.time(),
                connector_name=self.config.connector_name,
                trading_pair=self.config.trading_pair,
                side=TradeType.BUY,
                amount=amount,
                price=target_price,
                execution_strategy=ExecutionStrategy.LIMIT_MAKER,
            ),
        )

"""
BBO Peg Buy Controller — minimal V2 controller scaffold.

Pegs a single LIMIT_MAKER buy at external_best_bid + 1 tick, where
external_best_bid excludes our own resting order (so we never anchor
to ourselves). Re-quotes on any drift. N-shot: once `max_fills` distinct
executors have reported a non-zero fill, the controller stops creating
new orders for the lifetime of the process. Default max_fills=1 preserves
the original one-shot behavior. No close-side, no triple barrier, no
rebalance. WS-fed via MarketDataProvider — zero HTTP polling at the
strategy layer.
"""

from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple

from pydantic import Field, field_validator

from hummingbot.core.data_type.common import MarketDict, PriceType, TradeType
from hummingbot.strategy_v2.controllers.controller_base import (
    ControllerBase,
    ControllerConfigBase,
)
from hummingbot.strategy_v2.executors.executor_orchestrator import ExecutorOrchestrator
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

from controllers.market_making import fill_tracker
from controllers.market_making.no_clamp_order_executor import NoClampOrderExecutor

# Process-wide wire-up: route every "order_executor" CreateExecutorAction
# through NoClampOrderExecutor so this controller can actually peg one tick
# above current_best_bid (stock OrderExecutor silently clamps it down).
# See no_clamp_order_executor.py for the rationale. Side-effect: if any other
# controller runs in the same process and relies on the upstream clamp, it
# will get the no-clamp variant too — currently a single-controller setup so
# not a concern.
# Upstream types _executor_mapping's values as a Literal union of the originally-
# registered executor classes, so any subclass assignment is rejected even though
# it's semantically valid (NoClampOrderExecutor IS an OrderExecutor).
ExecutorOrchestrator._executor_mapping["order_executor"] = NoClampOrderExecutor  # type: ignore[assignment]  # ty: ignore[invalid-assignment]


class BBOPegBuyConfig(ControllerConfigBase):
    controller_type: str = "market_making"
    controller_name: str = "bbo_peg_buy"

    connector_name: str = Field(default="htx")
    trading_pair: str = Field(default="XNO-USDT")
    update_interval: float = Field(
        default=2.0,
        description="Seconds between controller ticks. Lower = faster reaction, more REST traffic. Default 2.0s gives the exchange order book WebSocket time to reflect our cancels before the next walker pass, avoiding the cancel-lag self-chase loop where ghost orders look like external bids.",
    )
    min_spread_pct: Decimal = Field(
        default=Decimal("0.02"),
        description="Anti-spoof gate. Refuse to quote when (best_ask - external_best_bid) / external_best_bid < this. Default 0.02 = 2%. HARD FLOOR (production): must be > 0.004 (0.4%) — that's the round-trip taker fee on HTX retail tier. Below 0.4% the strategy is structurally unprofitable regardless of adverse selection. Value 0 is allowed as an explicit gate-disabled sentinel for unit tests; production YAMLs must never use it.",
    )

    @field_validator("min_spread_pct")
    @classmethod
    def _min_spread_pct_above_floor(cls, v: Decimal) -> Decimal:
        # Allow 0 (explicit "gate disabled" sentinel used by tests) but reject
        # any positive value below the round-trip fee floor of 0.4%.
        if v != Decimal("0") and v <= Decimal("0.004"):
            raise ValueError(
                f"min_spread_pct={v} is below the 0.4% production floor "
                f"(round-trip fees 0.4% on HTX retail). Set to 0 "
                f"explicitly only for testing; production configs must use > 0.004."
            )
        return v
    cancel_debounce_seconds: float = Field(
        default=2.0,
        description="How long to remember each just-cancelled order's (price, amount) so the walker keeps treating it as ours during the cancel-propagation lag window. Without this, the walker sees our own just-cancelled orders in the exchange book WebSocket and chases them as if they were external bids. Set to 0 to disable.",
    )
    max_fills: int = Field(
        default=1,
        ge=1,
        description="Number of distinct fills allowed before the controller stops creating new orders for the process lifetime. Default 1 preserves the original one-shot behavior. Each unique executor is counted at most once (partial fills don't inflate the count). Raise to allow the controller to re-quote and capture more spread per process; lower not allowed (use a separate stop signal).",
    )
    max_buy_lead: Optional[int] = Field(
        default=None,
        ge=1,
        description="Cross-controller inventory cap. When set, the buy controller refuses to emit a Create if (buys_filled - sells_filled) >= this value, where both counts are tracked across the buy and sell controllers for the same trading pair via the shared fill_tracker module. Prevents the buy side from accumulating inventory faster than the sell side can clear it when running max_fills > 1. None (default) disables the cap — buys fire freely up to max_fills regardless of sell state.",
    )
    max_inventory_quote: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Inventory value cap, in quote currency (e.g. USDT). When set, the buy controller refuses to emit a Create once the current base inventory — marked at the live mid price — reaches this value, and re-enables buys only after the value falls back below it. Independent of max_buy_lead: max_buy_lead caps the *count* of unmatched fills, this caps the *quote value* of accumulated base. Inventory is read from the connector's total base balance, so any base already in the account (including amounts locked in resting sells) counts toward the cap. None (default) disables it.",
    )

    def update_markets(self, markets: MarketDict) -> MarketDict:
        # Upstream's add_or_update signature mistypes *args as the set type
        # itself instead of a set element; mirrors MarketMakingControllerConfigBase.
        return markets.add_or_update(self.connector_name, self.trading_pair)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    @property
    def log_prefix(self) -> str:
        """Log prefix including the base asset, so multi-token deployments
        (e.g. XNO + WAVES + ERA running together) can be disambiguated with
        a single grep. Format: '[bbo_peg <BASE>]'.
        """
        return f"[bbo_peg {self.trading_pair.split('-')[0]}]"


class BBOPegBuyController(ControllerBase):
    config: BBOPegBuyConfig

    def __init__(self, config: BBOPegBuyConfig, *args, **kwargs):
        kwargs.setdefault("update_interval", config.update_interval)
        super().__init__(config, *args, **kwargs)
        self.config = config
        # N-shot latch. Counts distinct executors that have reported a non-zero
        # fill. The controller stops creating new orders once _fill_count
        # reaches config.max_fills. Default max_fills=1 preserves the original
        # one-shot behavior. Never decremented within a process lifetime.
        self._fill_count: int = 0
        # IDs of executors already counted toward _fill_count, so partial
        # fills on the same executor don't inflate the count across ticks.
        self._counted_fill_executor_ids: Set[str] = set()
        # Cache of the inventory-cap gate's last state, so we only log on
        # capped <-> cleared transitions (matches the anti-spoof and balance
        # log patterns). False = not currently capped.
        self._buy_lead_capped_logged: bool = False
        # Same rate-limit cache for the inventory-value gate (max_inventory_quote).
        # False = not currently capped by value.
        self._inventory_value_capped_logged: bool = False
        # Cache of the last logged external_best_bid value, so we only emit
        # a diagnostic line when the chosen price actually changes.
        self._last_logged_external_best_bid: Optional[Decimal] = None
        # Cache of the anti-spoof gate's last state, so we only log on
        # blocked <-> cleared transitions. False = not currently gated.
        self._gate_blocked: bool = False
        # Cache of the last logged (executor_id, price) snapshot of active
        # executors. None on the very first tick so the initial snapshot
        # always logs. Used by _log_active_executors_snapshot to detect
        # orphaned exchange orders that aren't tracked in executors_info.
        self._last_logged_actives: Optional[List[Tuple[str, float]]] = None
        # Pending-cancel memory keyed by executor_id, value = (price, amount,
        # until_timestamp). When the controller emits a Stop, the executor
        # disappears from executors_info instantly but the exchange order
        # book WebSocket lags by ~1-2s; during that window the walker would
        # see our just-cancelled order as an external bid and chase it.
        # This dict lets _compute_own_volume_by_price keep subtracting the
        # cancelled volume from the book until cancel_debounce_seconds elapses.
        self._pending_cancels: Dict[str, Tuple[Decimal, Decimal, float]] = {}

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
        """Aggregate our active executors' volume per price level, AND any
        pending-cancel volume still echoing in the exchange book during the
        cancel-propagation lag window.

        Filters out inactive executors and non-OrderExecutorConfig configs.
        Pending cancels past their until_timestamp are skipped (inline filter,
        no mutation — eviction happens opportunistically in
        _record_pending_cancels to keep the dict bounded).
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
        # Cancel-debounce: keep subtracting our recently-cancelled orders
        # from the book until cancel_debounce_seconds elapses. Without this
        # the walker chases its own ghost orders (cancel hasn't propagated
        # to the exchange book WebSocket yet → ghost looks like external bid).
        now = self.market_data_provider.time()
        for price, amount, until in self._pending_cancels.values():
            if until <= now:
                continue
            my_volume_by_price[price] = (
                my_volume_by_price.get(price, Decimal("0")) + amount
            )
        return my_volume_by_price

    def _record_pending_cancels(
        self, executors_to_stop: List[ExecutorInfo]
    ) -> None:
        """Record (price, amount, until_timestamp) for each executor we're
        about to Stop, keyed by executor_id. Opportunistically evicts
        already-expired entries to keep the dict bounded over long sessions.

        Re-recording an executor that's already pending refreshes its
        until_timestamp — useful when the framework keeps the executor in
        is_active=True across multiple ticks while the cancel is in flight.
        """
        now = self.market_data_provider.time()
        # Opportunistic eviction (safe: expired entries are ignored in
        # _compute_own_volume_by_price anyway; this just frees memory).
        self._pending_cancels = {
            eid: entry
            for eid, entry in self._pending_cancels.items()
            if entry[2] > now
        }
        until = now + self.config.cancel_debounce_seconds
        for e in executors_to_stop:
            cfg = e.config
            if not isinstance(cfg, OrderExecutorConfig) or cfg.price is None:
                continue
            self._pending_cancels[str(e.id)] = (cfg.price, cfg.amount, until)

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
            f"{self.config.log_prefix} external_best_bid={result} "
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
            f"{self.config.log_prefix} gate_{state} external_best_bid={external_best_bid} "
            f"best_ask={best_ask} spread={spread_pct} "
            f"min={self.config.min_spread_pct}"
        )
        self._gate_blocked = gate_blocked

    def determine_executor_actions(self) -> List[ExecutorAction]:
        self._update_fill_latch()
        self._log_active_executors_snapshot()

        target_price: Optional[Decimal] = self.processed_data.get("target_price")
        if target_price is None:
            # No valid target (no external bid, would cross ask, or anti-spoof
            # gate fired). Cancel any standing order rather than leaving a
            # quote exposed during the unsafe window. The cost is occasional
            # cancel-churn on transient market-data gaps; the alternative
            # leaves the gate toothless against the spoof pattern it exists
            # to defend against.
            actives = [e for e in self.executors_info if e.is_active]
            self._record_pending_cancels(actives)
            actions = self._build_stop_actions(actives)
            self._log_actions_emitted(actions)
            return actions

        stale, in_tolerance = self._categorize_active_orders(target_price)
        self._record_pending_cancels(stale)
        actions = self._build_stop_actions(stale)

        # N-shot: never create another order once we've hit max_fills.
        if self._fill_count >= self.config.max_fills:
            self._log_actions_emitted(actions)
            return actions

        if not in_tolerance:
            create = self._build_create_action(target_price)
            if create is not None:
                actions.append(create)

        self._log_actions_emitted(actions)
        return actions

    def _log_active_executors_snapshot(self) -> None:
        """Emit a forensic INFO line whenever the set of (executor_id, price)
        pairs for active executors changes. Rate-limited to state changes only
        so steady-state ticks don't spam the log.

        This is the orphan-detection signal: if the exchange UI shows an order
        at price X but no log line ever lists X in `actives`, that order is
        orphaned from executors_info — the controller doesn't know about it
        and will never emit a Stop to cancel it.
        """
        snapshot: List[Tuple[str, float]] = sorted(
            (str(e.id), float(e.config.price))
            for e in self.executors_info
            if e.is_active
            and isinstance(e.config, OrderExecutorConfig)
            and e.config.price is not None
        )
        if snapshot == self._last_logged_actives:
            return
        self.logger().info(f"{self.config.log_prefix} actives={snapshot}")
        self._last_logged_actives = snapshot

    def _log_actions_emitted(self, actions: List[ExecutorAction]) -> None:
        """Emit a forensic INFO line summarizing the actions emitted this tick.
        Only fires when actions is non-empty so no-op ticks don't spam.

        Combined with _log_active_executors_snapshot, lets us verify that the
        controller IS attempting to cancel each order it knows about — if the
        exchange still shows an order after a Stop was logged for its
        executor_id, the cancel failed at the framework/exchange layer (not
        the controller's fault).
        """
        if not actions:
            return
        stops = [
            a.executor_id for a in actions if isinstance(a, StopExecutorAction)
        ]
        creates = [
            float(a.executor_config.price)
            for a in actions
            if isinstance(a, CreateExecutorAction)
            and isinstance(a.executor_config, OrderExecutorConfig)
            and a.executor_config.price is not None
        ]
        self.logger().info(f"{self.config.log_prefix} emit stops={stops} creates={creates}")

    def _update_fill_latch(self) -> None:
        """Increment _fill_count for each NEW executor reporting executed_amount_base > 0.
        Each executor id is counted at most once (tracked in
        _counted_fill_executor_ids), so partial fills accumulating across
        ticks don't inflate the count. Once _fill_count reaches
        config.max_fills, _has_filled becomes True and the controller stops
        emitting new Creates.

        Each newly-counted fill also records into the cross-controller
        fill_tracker so the sell side (and the max_buy_lead inventory cap)
        can see this buy's count.
        """
        for e in self.executors_info:
            eid = str(e.id)
            if eid in self._counted_fill_executor_ids:
                continue
            executed = (
                e.custom_info.get("executed_amount_base") if e.custom_info else None
            )
            if executed is not None and Decimal(str(executed)) > 0:
                self._fill_count += 1
                self._counted_fill_executor_ids.add(eid)
                fill_tracker.record_buy_fill(self.config.trading_pair)

    @property
    def _has_filled(self) -> bool:
        """Backward-compatible alias: True when fill count has reached
        max_fills (controller is fully gated, no more Creates). With the
        default max_fills=1, this is exactly the original one-shot latch.
        """
        return self._fill_count >= self.config.max_fills

    @_has_filled.setter
    def _has_filled(self, value: bool) -> None:
        """Backward-compatible setter for tests that pre-set the latch
        state directly. True → fill_count := max_fills (fully gated);
        False → fill_count := 0 (re-armed, counted IDs also cleared so
        previously-counted executors can be re-counted).
        """
        if value:
            self._fill_count = self.config.max_fills
        else:
            self._fill_count = 0
            self._counted_fill_executor_ids.clear()

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
        Returns None if quantized amount is zero (book can't support an
        order), if the cross-controller inventory cap (max_buy_lead) has
        been reached, or if the inventory-value cap (max_inventory_quote)
        has been reached.
        """
        if not self._is_within_inventory_cap():
            return None
        if not self._is_within_inventory_value_cap():
            return None
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

    def _is_within_inventory_cap(self) -> bool:
        """Return False (and log on state transition) when the buy side has
        accumulated max_buy_lead unmatched fills ahead of sells. Returns
        True if the cap is disabled (max_buy_lead is None) or lead is still
        below the cap.

        Lead is computed across processes-wide fill_tracker, so a sell-side
        fill on the same trading_pair immediately lowers the lead and can
        re-enable buy Creates within the same tick.
        """
        cap = self.config.max_buy_lead
        if cap is None:
            return True
        lead = fill_tracker.buy_lead(self.config.trading_pair)
        if lead >= cap:
            if not self._buy_lead_capped_logged:
                self.logger().warning(
                    f"{self.config.log_prefix} buy paused: lead={lead} "
                    f">= max_buy_lead={cap} "
                    f"(buys={fill_tracker.buys_filled(self.config.trading_pair)}, "
                    f"sells={fill_tracker.sells_filled(self.config.trading_pair)}). "
                    f"Waiting for a sell fill before quoting again."
                )
                self._buy_lead_capped_logged = True
            return False
        if self._buy_lead_capped_logged:
            self.logger().info(
                f"{self.config.log_prefix} buy resumed: lead={lead} "
                f"< max_buy_lead={cap}."
            )
            self._buy_lead_capped_logged = False
        return True

    def _is_within_inventory_value_cap(self) -> bool:
        """Return False (and log on state transition) when the base inventory,
        marked at the connector mid price, has reached max_inventory_quote.
        Returns True if the cap is disabled (None) or the value is below it.

        Independent of max_buy_lead: that caps the *count* of unmatched fills,
        this caps the *quote value* of accumulated base. Inventory is the
        connector's total base balance (includes base locked in our own
        resting sells), so the cap reflects the full position. Marked at the
        connector mid (PriceType.MidPrice) so the value tracks fair price
        rather than the side we happen to be quoting.
        """
        cap = self.config.max_inventory_quote
        if cap is None:
            return True
        base_asset = self.config.trading_pair.split("-")[0]
        inventory = self.market_data_provider.get_balance(
            self.config.connector_name, base_asset
        )
        mid = self.market_data_provider.get_price_by_type(
            self.config.connector_name, self.config.trading_pair, PriceType.MidPrice
        )
        value = inventory * mid
        if value >= cap:
            if not self._inventory_value_capped_logged:
                self.logger().warning(
                    f"{self.config.log_prefix} buy paused: inventory value "
                    f"{value} ({inventory} {base_asset} @ mid {mid}) >= "
                    f"max_inventory_quote={cap}. Waiting for inventory value "
                    f"to fall before quoting again."
                )
                self._inventory_value_capped_logged = True
            return False
        if self._inventory_value_capped_logged:
            self.logger().info(
                f"{self.config.log_prefix} buy resumed: inventory value "
                f"{value} ({inventory} {base_asset} @ mid {mid}) "
                f"< max_inventory_quote={cap}."
            )
            self._inventory_value_capped_logged = False
        return True

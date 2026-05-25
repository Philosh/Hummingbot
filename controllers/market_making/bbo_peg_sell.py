"""
BBO Peg Sell Controller — minimal V2 controller scaffold.

Pegs a single LIMIT_MAKER sell at external_best_ask - 1 tick, where
external_best_ask excludes our own resting order (so we never anchor
to ourselves). Re-quotes on any drift. One-shot: once any fill
occurs (partial or full), the controller stops creating new orders
for the lifetime of the process. No close-side, no triple barrier,
no rebalance. WS-fed via MarketDataProvider — zero HTTP polling at
the strategy layer.

Mirror of bbo_peg_buy.py. Runs as an independent controller instance;
inventory coupling between buy and sell sides is intentionally absent
(each side has its own _fill_count latch with configurable max_fills,
default 1 = one-shot). Designed to run alongside
bbo_peg_buy in the same process — both share the NoClampOrderExecutor
wire-up (idempotent), and each controller's executors_info is filtered
by the framework to its own controller_id, so there's no cross-side
contamination in the self-exclusion walker or the cancel-debounce table.

Known interaction (NOT fixed in this iteration): the anti-spoof gate
measures the spread between external_best_ask and `best_bid` (raw top
of book, NOT self-excluded). When the buy controller is also running,
its order may be the new best_bid, narrowing the measured spread by
1 tick. With a 2% threshold and a natural market spread near 2%, this
can produce a both-sides cancel-loop. Symmetric with the buy controller's
analogous risk on best_ask. Fix would be to walk both books for self
exclusion before computing the gate.
"""

from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple

from pydantic import Field

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
# below current_best_ask (stock OrderExecutor silently clamps it up). Idempotent
# if bbo_peg_buy is also loaded — both modules assign the same class.
# See no_clamp_order_executor.py for the rationale.
# Upstream types _executor_mapping's values as a Literal union of the originally-
# registered executor classes, so any subclass assignment is rejected even though
# it's semantically valid (NoClampOrderExecutor IS an OrderExecutor).
ExecutorOrchestrator._executor_mapping["order_executor"] = NoClampOrderExecutor  # type: ignore[assignment]  # ty: ignore[invalid-assignment]


class BBOPegSellConfig(ControllerConfigBase):
    controller_type: str = "market_making"
    controller_name: str = "bbo_peg_sell"

    connector_name: str = Field(default="htx")
    trading_pair: str = Field(default="XNO-USDT")
    update_interval: float = Field(
        default=2.0,
        description="Seconds between controller ticks. Lower = faster reaction, more REST traffic. Default 2.0s gives the exchange order book WebSocket time to reflect our cancels before the next walker pass, avoiding the cancel-lag self-chase loop where ghost orders look like external asks.",
    )
    min_spread_pct: Decimal = Field(
        default=Decimal("0.02"),
        description="Anti-spoof gate. Refuse to quote when (external_best_ask - best_bid) / external_best_ask < this. Default 0.02 = 2%.",
    )
    cancel_debounce_seconds: float = Field(
        default=2.0,
        description="How long to remember each just-cancelled order's (price, amount) so the walker keeps treating it as ours during the cancel-propagation lag window. Without this, the walker sees our own just-cancelled orders in the exchange book WebSocket and chases them as if they were external asks. Set to 0 to disable.",
    )
    max_fills: int = Field(
        default=1,
        ge=1,
        description="Number of distinct fills allowed before the controller stops creating new orders for the process lifetime. Default 1 preserves the original one-shot behavior. Each unique executor is counted at most once (partial fills don't inflate the count). Raise to allow the controller to re-quote and capture more spread per process; lower not allowed (use a separate stop signal).",
    )

    def update_markets(self, markets: MarketDict) -> MarketDict:
        # Upstream's add_or_update signature mistypes *args as the set type
        # itself instead of a set element; mirrors MarketMakingControllerConfigBase.
        return markets.add_or_update(self.connector_name, self.trading_pair)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    @property
    def log_prefix(self) -> str:
        """Log prefix including the base asset, so multi-token deployments
        (e.g. XNO + WAVES + ERA running together) can be disambiguated with
        a single grep. Format: '[bbo_peg_sell <BASE>]'.
        """
        return f"[bbo_peg_sell {self.trading_pair.split('-')[0]}]"


class BBOPegSellController(ControllerBase):
    config: BBOPegSellConfig

    def __init__(self, config: BBOPegSellConfig, *args, **kwargs):
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
        # Cache of the last logged external_best_ask value, so we only emit
        # a diagnostic line when the chosen price actually changes.
        self._last_logged_external_best_ask: Optional[Decimal] = None
        # Cache of the anti-spoof gate's last state, so we only log on
        # blocked <-> cleared transitions. False = not currently gated.
        self._gate_blocked: bool = False
        # Cache of the pre-flight balance check's last state, so we only log
        # on insufficient <-> sufficient transitions. False = balance was OK
        # last time we tried to create. Without this rate limit, an empty
        # account would emit a "not enough budget" warning every tick.
        self._balance_insufficient_logged: bool = False
        # Cache of the last logged (executor_id, price) snapshot of active
        # executors. None on the very first tick so the initial snapshot
        # always logs. Used by _log_active_executors_snapshot to detect
        # orphaned exchange orders that aren't tracked in executors_info.
        self._last_logged_actives: Optional[List[Tuple[str, float]]] = None
        # Pending-cancel memory keyed by executor_id, value = (price, amount,
        # until_timestamp). When the controller emits a Stop, the executor
        # disappears from executors_info instantly but the exchange order
        # book WebSocket lags by ~1-2s; during that window the walker would
        # see our just-cancelled order as an external ask and chase it.
        # This dict lets _compute_own_volume_by_price keep subtracting the
        # cancelled volume from the book until cancel_debounce_seconds elapses.
        self._pending_cancels: Dict[str, Tuple[Decimal, Decimal, float]] = {}

    async def update_processed_data(self):
        rules = self.market_data_provider.get_trading_rules(
            self.config.connector_name, self.config.trading_pair
        )
        tick: Decimal = rules.min_price_increment

        best_bid: Decimal = self.market_data_provider.get_price_by_type(
            self.config.connector_name, self.config.trading_pair, PriceType.BestBid
        )

        external_best_ask = self._external_best_ask()
        target_price = self._compute_target_price(external_best_ask, tick, best_bid)

        self.processed_data = {
            "tick": tick,
            "external_best_ask": external_best_ask,
            "best_bid": best_bid,
            "target_price": target_price,
        }

    def _external_best_ask(self) -> Optional[Decimal]:
        """
        Walk the ask book top-down (lowest price first) and return the lowest
        price level that has volume beyond what our own active executors are
        resting there. Prevents the controller from chasing its own quote.

        Emits a forensic INFO line whenever the chosen price changes, so we
        can later audit why a particular target_price was picked.
        """
        my_volume_by_price = self._compute_own_volume_by_price()
        result, top_levels = self._walk_asks_for_first_external(my_volume_by_price)
        self._log_external_ask_change(result, top_levels, my_volume_by_price)
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
        # to the exchange book WebSocket yet → ghost looks like external ask).
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

    def _walk_asks_for_first_external(
        self, my_volume_by_price: Dict[Decimal, Decimal]
    ) -> Tuple[Optional[Decimal], List[Tuple[float, float]]]:
        """Walk the top 10 ask levels (lowest price first) and return:
        - the lowest price where (book amount - our amount) > 0, else None
        - the top-5 levels as (price, amount) float tuples, for logging
        """
        order_book = self.market_data_provider.get_order_book(
            self.config.connector_name, self.config.trading_pair
        )
        top_levels: List[Tuple[float, float]] = []
        result: Optional[Decimal] = None
        for i, row in enumerate(order_book.ask_entries()):
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

    def _log_external_ask_change(
        self,
        result: Optional[Decimal],
        top_levels: List[Tuple[float, float]],
        my_volume_by_price: Dict[Decimal, Decimal],
    ) -> None:
        """Emit a forensic INFO line only when the chosen external_best_ask
        changes from the previously logged value. Updates the cache after logging.
        """
        if result == self._last_logged_external_best_ask:
            return
        my_vol_str = {float(k): float(v) for k, v in my_volume_by_price.items()}
        self.logger().info(
            f"{self.config.log_prefix} external_best_ask={result} "
            f"top5_asks={top_levels} my_volume={my_vol_str}"
        )
        self._last_logged_external_best_ask = result

    def _compute_target_price(
        self,
        external_best_ask: Optional[Decimal],
        tick: Decimal,
        best_bid: Decimal,
    ) -> Optional[Decimal]:
        """Peg one tick below the external best ask, quantized to the exchange's
        tick size. Returns None if there's no external ask to peg to, if the
        market spread is below the anti-spoof gate (config.min_spread_pct), or
        if the candidate would cross the bid (LIMIT_MAKER would be rejected).
        """
        if external_best_ask is None or external_best_ask <= 0:
            return None
        if not best_bid:
            return None
        spread_pct = self._compute_spread_pct(external_best_ask, best_bid)
        gate_blocked = spread_pct < self.config.min_spread_pct
        self._log_gate_state_change(
            gate_blocked, external_best_ask, best_bid, spread_pct
        )
        if gate_blocked:
            return None
        candidate = self.market_data_provider.quantize_order_price(
            self.config.connector_name,
            self.config.trading_pair,
            external_best_ask - tick,
        )
        if candidate <= best_bid:
            return None
        return candidate

    def _compute_spread_pct(
        self, external_best_ask: Decimal, best_bid: Decimal
    ) -> Decimal:
        """Return (external_best_ask - best_bid) / external_best_ask.

        This is the anti-spoof gate input: a spoofer that pushes the visible
        best ask down toward the bid compresses this value. Caller must ensure
        external_best_ask > 0.
        """
        return (external_best_ask - best_bid) / external_best_ask

    def _log_gate_state_change(
        self,
        gate_blocked: bool,
        external_best_ask: Decimal,
        best_bid: Decimal,
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
            f"{self.config.log_prefix} gate_{state} external_best_ask={external_best_ask} "
            f"best_bid={best_bid} spread={spread_pct} "
            f"min={self.config.min_spread_pct}"
        )
        self._gate_blocked = gate_blocked

    def determine_executor_actions(self) -> List[ExecutorAction]:
        self._update_fill_latch()
        self._log_active_executors_snapshot()

        target_price: Optional[Decimal] = self.processed_data.get("target_price")
        if target_price is None:
            # No valid target (no external ask, would cross bid, or anti-spoof
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
        fill_tracker so the buy side (and its max_buy_lead inventory cap)
        can see this sell's count.
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
                fill_tracker.record_sell_fill(self.config.trading_pair)

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
        """Build a LIMIT_MAKER sell at target_price.
        Returns None if quantized amount is zero (book can't support an
        order) or if the account's available base balance is below the
        required amount (pre-flight check that prevents the framework's
        downstream INSUFFICIENT_BALANCE log spam at every tick).
        """
        amount = self.market_data_provider.quantize_order_amount(
            self.config.connector_name,
            self.config.trading_pair,
            self.config.total_amount_quote / target_price,
        )
        if amount <= 0:
            return None
        if not self._has_sufficient_base_balance(amount):
            return None
        return CreateExecutorAction(
            controller_id=self.config.id,
            executor_config=OrderExecutorConfig(
                timestamp=self.market_data_provider.time(),
                connector_name=self.config.connector_name,
                trading_pair=self.config.trading_pair,
                side=TradeType.SELL,
                amount=amount,
                price=target_price,
                execution_strategy=ExecutionStrategy.LIMIT_MAKER,
            ),
        )

    def _has_sufficient_base_balance(self, required_amount: Decimal) -> bool:
        """Read the connector's available base balance and compare to the
        amount we're about to attempt to sell. Returns False if insufficient
        (caller skips emitting the Create).

        Emits a state-transition log: warning on insufficient->sufficient,
        info on recovery. Rate-limited via _balance_insufficient_logged so
        an empty account doesn't spam one warning per tick.
        """
        base_asset = self.config.trading_pair.split("-")[0]
        available = self.market_data_provider.get_available_balance(
            self.config.connector_name, base_asset
        )
        if available < required_amount:
            if not self._balance_insufficient_logged:
                self.logger().warning(
                    f"{self.config.log_prefix} insufficient {base_asset} balance: "
                    f"have {available}, need {required_amount}. "
                    f"Suppressing further warnings until balance recovers."
                )
                self._balance_insufficient_logged = True
            return False
        if self._balance_insufficient_logged:
            self.logger().info(
                f"{self.config.log_prefix} {base_asset} balance recovered: "
                f"have {available}, need {required_amount}. Resuming quotes."
            )
            self._balance_insufficient_logged = False
        return True

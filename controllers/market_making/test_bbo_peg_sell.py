import asyncio
import unittest
from decimal import Decimal
from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase
from typing import List, Optional, Tuple, cast
from unittest.mock import AsyncMock, MagicMock, PropertyMock

from hummingbot.connector.exchange_py_base import ExchangePyBase
from hummingbot.connector.trading_rule import TradingRule  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]
from hummingbot.core.data_type.common import TradeType
from hummingbot.data_feed.market_data_provider import MarketDataProvider
from hummingbot.strategy.strategy_v2_base import StrategyV2Base
from hummingbot.strategy_v2.executors.order_executor.data_types import (
    ExecutionStrategy,
    OrderExecutorConfig,
)
from hummingbot.strategy_v2.models.executor_actions import (
    CreateExecutorAction,
    StopExecutorAction,
)
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo

from controllers.market_making.bbo_peg_sell import (
    BBOPegSellConfig,
    BBOPegSellController,
)
from controllers.market_making.no_clamp_order_executor import NoClampOrderExecutor


def _fake_ask_row(price: Decimal, amount: Decimal) -> MagicMock:
    row = MagicMock()
    row.price = price
    row.amount = amount
    return row


def _fake_order_book(ask_levels: List[Tuple[Decimal, Decimal]]) -> MagicMock:
    """Mock OrderBook whose ask_entries() yields top-down (price, amount) rows."""
    book = MagicMock()
    book.ask_entries.return_value = [_fake_ask_row(p, a) for p, a in ask_levels]
    return book


def _fake_executor(
    *,
    price: Optional[Decimal],
    amount: Decimal,
    is_active: bool = True,
    use_order_executor_config: bool = True,
) -> MagicMock:
    """Mock ExecutorInfo. If use_order_executor_config is False, cfg is a plain
    MagicMock so isinstance(cfg, OrderExecutorConfig) fails (exercises the
    defensive filter in _external_best_ask). custom_info defaults to None
    so the fill latch treats this as 'no fill recorded' — fill-specific
    tests should use a custom helper that sets custom_info explicitly."""
    executor = MagicMock()
    executor.is_active = is_active
    executor.custom_info = None
    cfg = (
        MagicMock(spec=OrderExecutorConfig)
        if use_order_executor_config
        else MagicMock()
    )
    cfg.price = price
    cfg.amount = amount
    executor.config = cfg
    return executor


def _make_controller_for_walker(
    *,
    ask_levels: Optional[List[Tuple[Decimal, Decimal]]] = None,
    executors: Optional[List[MagicMock]] = None,
) -> Tuple[BBOPegSellController, MagicMock]:
    """Factory for _external_best_ask tests. Wires a fake order book and
    executors_info; returns (controller, log_mock) so tests can assert on
    log emissions via log_mock.info.call_count.
    """
    config = BBOPegSellConfig(
        id="test",
        controller_name="bbo_peg_sell",
        connector_name="htx",
        trading_pair="XNO-USDT",
        total_amount_quote=Decimal("20"),
        update_interval=0.5,
    )
    market_data_provider = MagicMock(spec=MarketDataProvider)
    market_data_provider.get_order_book.return_value = _fake_order_book(
        ask_levels or []
    )

    controller = BBOPegSellController(
        config=config,
        market_data_provider=market_data_provider,
        actions_queue=AsyncMock(spec=asyncio.Queue),
    )
    setattr(controller, "executors_info", executors or [])
    log_mock = MagicMock()
    setattr(controller, "logger", MagicMock(return_value=log_mock))
    return controller, log_mock


def _make_controller_with_market(
    *,
    tick: Decimal = Decimal("0.0001"),
    best_bid: Decimal = Decimal("0.4380"),
    external_best_ask: Optional[Decimal] = Decimal("0.4385"),
) -> BBOPegSellController:
    """Factory that wires a controller with mocked market data.

    quantize_order_price is an identity passthrough so target_price math is
    exactly external_best_ask - tick — easy to assert in tests.
    _external_best_ask is patched to return whatever the test asks for.
    """
    config = BBOPegSellConfig(
        id="test",
        controller_name="bbo_peg_sell",
        connector_name="htx",
        trading_pair="XNO-USDT",
        total_amount_quote=Decimal("20"),
        update_interval=0.5,
        # Anti-spoof gate disabled so these tests isolate peg/quantize math.
        # Gate behavior is covered separately in dedicated gate tests.
        min_spread_pct=Decimal("0"),
    )
    market_data_provider = MagicMock(spec=MarketDataProvider)
    rules = MagicMock()
    rules.min_price_increment = tick
    market_data_provider.get_trading_rules.return_value = rules
    market_data_provider.get_price_by_type.return_value = best_bid
    market_data_provider.quantize_order_price.side_effect = lambda _c, _p, price: price

    controller = BBOPegSellController(
        config=config,
        market_data_provider=market_data_provider,
        actions_queue=AsyncMock(spec=asyncio.Queue),
    )
    setattr(controller, "_external_best_ask", MagicMock(return_value=external_best_ask))
    return controller


class TestBBOPegSellControllerInit(unittest.TestCase):
    def setUp(self):
        self.config = BBOPegSellConfig(
            id="test",
            controller_name="bbo_peg_sell",
            connector_name="htx",
            trading_pair="XNO-USDT",
            total_amount_quote=Decimal("20"),
            update_interval=0.5,
        )
        self.market_data_provider = MagicMock(spec=MarketDataProvider)
        self.actions_queue = AsyncMock(spec=asyncio.Queue)

    def _make_controller(self, **kwargs) -> BBOPegSellController:
        return BBOPegSellController(
            config=self.config,
            market_data_provider=self.market_data_provider,
            actions_queue=self.actions_queue,
            **kwargs,
        )

    def test_initial_state(self):
        controller = self._make_controller()
        self.assertIs(controller.config, self.config)
        self.assertFalse(controller._has_filled)
        self.assertIsNone(controller._last_logged_external_best_ask)

    def test_update_interval_from_config(self):
        self.config.update_interval = 1.5
        controller = self._make_controller()
        self.assertEqual(controller.update_interval, 1.5)

    def test_explicit_kwarg_overrides_config(self):
        self.config.update_interval = 0.5
        controller = self._make_controller(update_interval=2.0)
        self.assertEqual(controller.update_interval, 2.0)


class TestBBOPegSellUpdateProcessedData(IsolatedAsyncioWrapperTestCase):
    async def test_golden_path_target_is_one_tick_below_ask(self):
        controller = _make_controller_with_market(
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.4380"),
            external_best_ask=Decimal("0.4385"),
        )
        await controller.update_processed_data()
        self.assertEqual(controller.processed_data["target_price"], Decimal("0.4384"))

    async def test_no_external_bid_target_is_none(self):
        controller = _make_controller_with_market(external_best_ask=None)
        await controller.update_processed_data()
        self.assertIsNone(controller.processed_data["target_price"])

    async def test_zero_external_bid_target_is_none(self):
        controller = _make_controller_with_market(external_best_ask=Decimal("0"))
        await controller.update_processed_data()
        self.assertIsNone(controller.processed_data["target_price"])

    async def test_candidate_equal_to_bid_target_is_none(self):
        # bid 0.4384 + tick 0.0001 = 0.4385 == best_bid → blocked (strict <)
        controller = _make_controller_with_market(
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.4384"),
            external_best_ask=Decimal("0.4385"),
        )
        await controller.update_processed_data()
        self.assertIsNone(controller.processed_data["target_price"])

    async def test_candidate_below_bid_target_is_none(self):
        # bid 0.4390 + tick 0.0001 = 0.4391 > best_bid 0.4385 → blocked
        controller = _make_controller_with_market(
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.4390"),
            external_best_ask=Decimal("0.4385"),
        )
        await controller.update_processed_data()
        self.assertIsNone(controller.processed_data["target_price"])

    async def test_zero_best_bid_target_is_none(self):
        # falsy ask short-circuits the crossing guard regardless of math
        controller = _make_controller_with_market(
            best_bid=Decimal("0.4380"),
            external_best_ask=Decimal("0"),
        )
        await controller.update_processed_data()
        self.assertIsNone(controller.processed_data["target_price"])

    async def test_processed_data_contains_all_fields(self):
        controller = _make_controller_with_market(
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.4380"),
            external_best_ask=Decimal("0.4385"),
        )
        await controller.update_processed_data()
        self.assertEqual(controller.processed_data["tick"], Decimal("0.0001"))
        self.assertEqual(
            controller.processed_data["external_best_ask"], Decimal("0.4385")
        )
        self.assertEqual(controller.processed_data["best_bid"], Decimal("0.4380"))
        self.assertEqual(controller.processed_data["target_price"], Decimal("0.4384"))


class TestBBOPegSellExternalBestAsk(unittest.TestCase):
    def test_empty_book_returns_none(self):
        controller, _ = _make_controller_for_walker(ask_levels=[])
        self.assertIsNone(controller._external_best_ask())

    def test_top_level_all_external_returns_top(self):
        controller, _ = _make_controller_for_walker(
            ask_levels=[(Decimal("0.4380"), Decimal("100"))]
        )
        self.assertEqual(controller._external_best_ask(), Decimal("0.4380"))

    def test_top_level_fully_ours_returns_next_level(self):
        # Top level (0.4380, 50) is entirely ours; walker drops to next external level.
        controller, _ = _make_controller_for_walker(
            ask_levels=[
                (Decimal("0.4380"), Decimal("50")),
                (Decimal("0.4379"), Decimal("100")),
            ],
            executors=[_fake_executor(price=Decimal("0.4380"), amount=Decimal("50"))],
        )
        self.assertEqual(controller._external_best_ask(), Decimal("0.4379"))

    def test_top_level_partially_ours_still_returns_top(self):
        # 30 of 100 at top is ours → 70 external → keep top.
        controller, _ = _make_controller_for_walker(
            ask_levels=[(Decimal("0.4380"), Decimal("100"))],
            executors=[_fake_executor(price=Decimal("0.4380"), amount=Decimal("30"))],
        )
        self.assertEqual(controller._external_best_ask(), Decimal("0.4380"))

    def test_inactive_executors_ignored(self):
        # Cancelled/stopped executor doesn't count toward our volume.
        controller, _ = _make_controller_for_walker(
            ask_levels=[(Decimal("0.4380"), Decimal("100"))],
            executors=[
                _fake_executor(
                    price=Decimal("0.4380"), amount=Decimal("100"), is_active=False
                )
            ],
        )
        self.assertEqual(controller._external_best_ask(), Decimal("0.4380"))

    def test_non_order_executor_configs_ignored(self):
        # Config isn't an OrderExecutorConfig instance → skip in volume map.
        controller, _ = _make_controller_for_walker(
            ask_levels=[(Decimal("0.4380"), Decimal("100"))],
            executors=[
                _fake_executor(
                    price=Decimal("0.4380"),
                    amount=Decimal("100"),
                    use_order_executor_config=False,
                )
            ],
        )
        self.assertEqual(controller._external_best_ask(), Decimal("0.4380"))

    def test_executor_with_none_price_ignored(self):
        # OrderExecutorConfig but price=None → skip in volume map.
        controller, _ = _make_controller_for_walker(
            ask_levels=[(Decimal("0.4380"), Decimal("100"))],
            executors=[_fake_executor(price=None, amount=Decimal("100"))],
        )
        self.assertEqual(controller._external_best_ask(), Decimal("0.4380"))

    def test_multiple_executors_same_price_sum_volumes(self):
        # Two 30-unit executors at 0.4380 → 60 ours → 40 external → keep top.
        controller, _ = _make_controller_for_walker(
            ask_levels=[(Decimal("0.4380"), Decimal("100"))],
            executors=[
                _fake_executor(price=Decimal("0.4380"), amount=Decimal("30")),
                _fake_executor(price=Decimal("0.4380"), amount=Decimal("30")),
            ],
        )
        self.assertEqual(controller._external_best_ask(), Decimal("0.4380"))

    def test_walker_caps_at_ten_levels(self):
        # 15 fully-ours levels; walker stops at index 10 → never finds external → None.
        ask_levels = [(Decimal(f"0.43{80 - i:02d}"), Decimal("10")) for i in range(15)]
        executors = [
            _fake_executor(price=p, amount=Decimal("10")) for p, _ in ask_levels
        ]
        controller, _ = _make_controller_for_walker(
            ask_levels=ask_levels, executors=executors
        )
        self.assertIsNone(controller._external_best_ask())

    def test_logs_only_when_result_changes(self):
        # Same book on both calls → same result → log fires once, not twice.
        controller, log_mock = _make_controller_for_walker(
            ask_levels=[(Decimal("0.4380"), Decimal("100"))]
        )
        controller._external_best_ask()
        controller._external_best_ask()
        self.assertEqual(log_mock.info.call_count, 1)

    def test_log_cache_updates_to_new_result(self):
        # Cache starts None, gets updated to the chosen price after first walk.
        controller, _ = _make_controller_for_walker(
            ask_levels=[(Decimal("0.4380"), Decimal("100"))]
        )
        self.assertIsNone(controller._last_logged_external_best_ask)
        controller._external_best_ask()
        self.assertEqual(controller._last_logged_external_best_ask, Decimal("0.4380"))

    # --- Defensive / fortification tests for the book walker ---

    def test_multi_level_book_picks_top_not_lower(self):
        # 3 fully external levels → must pick the highest (first-match-wins).
        controller, _ = _make_controller_for_walker(
            ask_levels=[
                (Decimal("0.4380"), Decimal("100")),
                (Decimal("0.4379"), Decimal("100")),
                (Decimal("0.4378"), Decimal("100")),
            ]
        )
        self.assertEqual(controller._external_best_ask(), Decimal("0.4380"))

    def test_first_match_wins_picks_top_not_largest_external(self):
        # Top has tiny external volume; deeper level has massive external.
        # Walker still picks the TOP — it's "highest external bid", not
        # "biggest external bid".
        controller, _ = _make_controller_for_walker(
            ask_levels=[
                (Decimal("0.4380"), Decimal("1")),
                (Decimal("0.4379"), Decimal("10000")),
            ]
        )
        self.assertEqual(controller._external_best_ask(), Decimal("0.4380"))

    def test_skip_two_owned_levels_returns_third(self):
        # Top 2 levels fully ours; 3rd is external → return 3rd.
        controller, _ = _make_controller_for_walker(
            ask_levels=[
                (Decimal("0.4380"), Decimal("100")),
                (Decimal("0.4379"), Decimal("50")),
                (Decimal("0.4378"), Decimal("200")),
            ],
            executors=[
                _fake_executor(price=Decimal("0.4380"), amount=Decimal("100")),
                _fake_executor(price=Decimal("0.4379"), amount=Decimal("50")),
            ],
        )
        self.assertEqual(controller._external_best_ask(), Decimal("0.4378"))

    def test_partial_ownership_chain(self):
        # Top fully ours → skip. Next level partially ours (70 external) → pick.
        # Even though level 3 is fully external, we stop at first level with
        # external > 0.
        controller, _ = _make_controller_for_walker(
            ask_levels=[
                (Decimal("0.4380"), Decimal("50")),
                (Decimal("0.4379"), Decimal("100")),
                (Decimal("0.4378"), Decimal("200")),
            ],
            executors=[
                _fake_executor(price=Decimal("0.4380"), amount=Decimal("50")),
                _fake_executor(price=Decimal("0.4379"), amount=Decimal("30")),
            ],
        )
        self.assertEqual(controller._external_best_ask(), Decimal("0.4379"))

    def test_negative_external_amount_is_skipped(self):
        # Our tracked volume EXCEEDS what's in the book (stale/buggy state):
        # external_amount = 50 - 100 = -50 → not > 0 → defensive skip.
        # Walker drops to the next level rather than returning a price we
        # might over-fill at.
        controller, _ = _make_controller_for_walker(
            ask_levels=[
                (Decimal("0.4380"), Decimal("50")),
                (Decimal("0.4379"), Decimal("100")),
            ],
            executors=[_fake_executor(price=Decimal("0.4380"), amount=Decimal("100"))],
        )
        self.assertEqual(controller._external_best_ask(), Decimal("0.4379"))

    def test_own_volume_at_different_price_does_not_affect_top(self):
        # Our order is at 0.4379, but top of book is 0.4380. The own-volume
        # map is keyed by price, so 0.4380's external count should be
        # unaffected by our 0.4379 holding.
        controller, _ = _make_controller_for_walker(
            ask_levels=[
                (Decimal("0.4380"), Decimal("100")),
                (Decimal("0.4379"), Decimal("100")),
            ],
            executors=[_fake_executor(price=Decimal("0.4379"), amount=Decimal("100"))],
        )
        self.assertEqual(controller._external_best_ask(), Decimal("0.4380"))

    def test_eleventh_level_never_reached_even_if_external(self):
        # Boundary: top 10 fully ours, 11th is fully external. The cap
        # (i >= 10 break) MUST prevent the walker from ever seeing level 11.
        # Returns None.
        ask_levels = [(Decimal(f"0.43{80 - i:02d}"), Decimal("10")) for i in range(10)]
        ask_levels.append((Decimal("0.4370"), Decimal("999")))  # external, level 11
        executors = [
            _fake_executor(price=p, amount=Decimal("10")) for p, _ in ask_levels[:10]
        ]
        controller, _ = _make_controller_for_walker(
            ask_levels=ask_levels, executors=executors
        )
        self.assertIsNone(controller._external_best_ask())

    def test_float_prices_in_book_coerce_via_str(self):
        # Real connectors emit floats. The walker does Decimal(str(row.price))
        # — using str-coercion so 0.4380 (float) becomes Decimal("0.438"),
        # which compares equal to the Decimal("0.4380") in our own-volume map.
        # If someone "simplifies" this to Decimal(row.price), the binary
        # float representation breaks dict lookups. This test would catch it.
        book = MagicMock()
        row = MagicMock()
        row.price = 0.4380  # float
        row.amount = 100.0  # float
        book.ask_entries.return_value = [row]

        config = BBOPegSellConfig(
            id="test",
            controller_name="bbo_peg_sell",
            connector_name="htx",
            trading_pair="XNO-USDT",
            total_amount_quote=Decimal("20"),
            update_interval=0.5,
        )
        market_data_provider = MagicMock(spec=MarketDataProvider)
        market_data_provider.get_order_book.return_value = book
        controller = BBOPegSellController(
            config=config,
            market_data_provider=market_data_provider,
            actions_queue=AsyncMock(spec=asyncio.Queue),
        )
        # Our own order at the same price expressed as Decimal — must
        # collide with the book's float→str→Decimal value.
        setattr(
            controller,
            "executors_info",
            [_fake_executor(price=Decimal("0.4380"), amount=Decimal("30"))],
        )
        setattr(controller, "logger", MagicMock(return_value=MagicMock()))

        # 100 in book - 30 ours = 70 external → pick top.
        self.assertEqual(controller._external_best_ask(), Decimal("0.4380"))


class TestBBOPegSellComputeTargetPrice(unittest.TestCase):
    """Unit tests for _compute_target_price in isolation.

    Tested directly (not through update_processed_data) so failures point at
    pricing logic specifically, not at the orchestration around it.
    """

    def _make_controller(self, quantize_side_effect=None) -> BBOPegSellController:
        """Builds a controller with quantize_order_price as identity passthrough
        unless a custom side_effect is provided. Stashes the mocked MDP on
        self.market_data_provider_mock so tests can assert on its calls without
        going through the spec-typed controller.market_data_provider attribute.
        """
        config = BBOPegSellConfig(
            id="test",
            controller_name="bbo_peg_sell",
            connector_name="htx",
            trading_pair="XNO-USDT",
            total_amount_quote=Decimal("20"),
            update_interval=0.5,
            # Anti-spoof gate disabled so these tests isolate peg/quantize math.
            # Gate behavior is covered separately in dedicated gate tests.
            min_spread_pct=Decimal("0"),
        )
        market_data_provider = MagicMock(spec=MarketDataProvider)
        market_data_provider.quantize_order_price.side_effect = (
            quantize_side_effect or (lambda _c, _p, price: price)
        )
        self.market_data_provider_mock = market_data_provider
        return BBOPegSellController(
            config=config,
            market_data_provider=market_data_provider,
            actions_queue=AsyncMock(spec=asyncio.Queue),
        )

    # --- Happy path ---

    def test_returns_candidate_when_bid_and_ask_provide_room(self):
        # ask 0.4385 - tick 0.0001 = 0.4384, bid 0.4380 → returns 0.4384.
        controller = self._make_controller()
        result = controller._compute_target_price(
            external_best_ask=Decimal("0.4385"),
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.4380"),
        )
        self.assertEqual(result, Decimal("0.4384"))

    # --- external_best_ask input gates ---

    def test_returns_none_when_external_ask_is_none(self):
        controller = self._make_controller()
        result = controller._compute_target_price(
            external_best_ask=None,
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.4385"),
        )
        self.assertIsNone(result)

    def test_returns_none_when_external_ask_is_zero(self):
        controller = self._make_controller()
        result = controller._compute_target_price(
            external_best_ask=Decimal("0"),
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.4385"),
        )
        self.assertIsNone(result)

    def test_returns_none_when_external_ask_is_negative(self):
        # Defensive: bids should never be negative, but guard regardless.
        controller = self._make_controller()
        result = controller._compute_target_price(
            external_best_ask=Decimal("-0.0001"),
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.4385"),
        )
        self.assertIsNone(result)

    # --- best_bid / crossing guard ---

    def test_returns_none_when_best_bid_is_zero(self):
        # Falsy ask short-circuits the crossing guard → bail.
        controller = self._make_controller()
        result = controller._compute_target_price(
            external_best_ask=Decimal("0.4380"),
            tick=Decimal("0.0001"),
            best_bid=Decimal("0"),
        )
        self.assertIsNone(result)

    def test_returns_none_when_candidate_equals_best_bid(self):
        # ask 0.4386 - tick 0.0001 = 0.4385 == best_bid → blocked (strict <=)
        controller = self._make_controller()
        result = controller._compute_target_price(
            external_best_ask=Decimal("0.4386"),
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.4385"),
        )
        self.assertIsNone(result)

    def test_returns_none_when_candidate_below_best_bid(self):
        # ask 0.4380 - tick 0.0001 = 0.4379 < best_bid 0.4390 → clearly crosses.
        controller = self._make_controller()
        result = controller._compute_target_price(
            external_best_ask=Decimal("0.4380"),
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.4390"),
        )
        self.assertIsNone(result)

    # --- Quantization plumbing ---

    def test_quantize_called_with_external_ask_minus_tick(self):
        # Verify quantize is called with the raw difference and exchange identifiers.
        controller = self._make_controller()
        controller._compute_target_price(
            external_best_ask=Decimal("0.4385"),
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.4380"),
        )
        self.market_data_provider_mock.quantize_order_price.assert_called_once_with(
            "htx", "XNO-USDT", Decimal("0.4384")
        )

    def test_returns_quantized_value_not_raw_diff(self):
        # If quantize snaps to a different price (e.g., exchange rounds up to
        # the next valid tick), we must return that snapped value, not the
        # unrounded difference. Pins that we trust the quantizer's output.
        # Quantize forces 0.4382 (between bid 0.4380 and ask 0.4385) so the
        # crossing guard (candidate <= best_bid) doesn't fire.
        controller = self._make_controller(
            quantize_side_effect=lambda _c, _p, _price: Decimal("0.4382")
        )
        result = controller._compute_target_price(
            external_best_ask=Decimal("0.4385"),
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.4380"),
        )
        self.assertEqual(result, Decimal("0.4382"))

    # --- Negative cases (what the function must NOT do) ---

    def test_does_not_call_quantize_when_external_ask_is_none(self):
        # Early return must skip the quantize side effect entirely. If someone
        # refactors and breaks the guard, we'd silently call quantize with None.
        controller = self._make_controller()
        controller._compute_target_price(
            external_best_ask=None,
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.4385"),
        )
        self.market_data_provider_mock.quantize_order_price.assert_not_called()

    def test_does_not_call_quantize_when_external_ask_is_zero(self):
        # Same short-circuit guard, zero case.
        controller = self._make_controller()
        controller._compute_target_price(
            external_best_ask=Decimal("0"),
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.4385"),
        )
        self.market_data_provider_mock.quantize_order_price.assert_not_called()


class TestBBOPegSellComputeSpreadPct(unittest.TestCase):
    """Direct unit tests for _compute_spread_pct in isolation.

    Pure function: (external_best_ask - best_bid) / external_best_ask.
    Pins the formula (denominator MUST be external_best_ask — not best_bid
    or mid-price), the Decimal return type (so the gate comparison against
    a Decimal threshold never silently coerces through float), and the
    sign convention on inverted books.
    """

    def _make_controller(self) -> BBOPegSellController:
        config = BBOPegSellConfig(
            id="test",
            controller_name="bbo_peg_sell",
            connector_name="htx",
            trading_pair="XNO-USDT",
            total_amount_quote=Decimal("20"),
            update_interval=0.5,
        )
        return BBOPegSellController(
            config=config,
            market_data_provider=MagicMock(spec=MarketDataProvider),
            actions_queue=AsyncMock(spec=asyncio.Queue),
        )

    # --- Happy path: standard spreads ---

    def test_healthy_wide_spread_returns_expected_ratio(self):
        # 5% spread under sell formula: (1.00 - 0.95) / 1.00 = 0.05
        controller = self._make_controller()
        result = controller._compute_spread_pct(
            external_best_ask=Decimal("1.00"),
            best_bid=Decimal("0.95"),
        )
        self.assertEqual(result, Decimal("0.05"))

    def test_tight_one_tick_spread_returns_tiny_ratio(self):
        # Typical spoof shape: ask one tick above bid.
        # (0.4296 - 0.4295) / 0.4296 ≈ 0.000233 — well below the 2% gate.
        controller = self._make_controller()
        result = controller._compute_spread_pct(
            external_best_ask=Decimal("0.4296"),
            best_bid=Decimal("0.4295"),
        )
        self.assertGreater(result, Decimal("0"))
        self.assertLess(result, Decimal("0.001"))

    def test_exact_two_percent_spread(self):
        # Boundary value matching the production gate threshold (0.02).
        # ask 1.00, bid 0.98 → (1.00 - 0.98) / 1.00 = exactly 0.02.
        controller = self._make_controller()
        result = controller._compute_spread_pct(
            external_best_ask=Decimal("1.00"),
            best_bid=Decimal("0.98"),
        )
        self.assertEqual(result, Decimal("0.02"))

    # --- Edge: degenerate / boundary cases ---

    def test_equal_bid_and_ask_returns_zero(self):
        # Spread = 0 exactly. Must not raise and must return Decimal("0").
        # The caller relies on this returning a comparable value so the
        # gate fires cleanly (0 < 0.02) instead of crashing the tick.
        controller = self._make_controller()
        result = controller._compute_spread_pct(
            external_best_ask=Decimal("0.4380"),
            best_bid=Decimal("0.4380"),
        )
        self.assertEqual(result, Decimal("0"))

    def test_inverted_book_returns_negative(self):
        # Ask BELOW bid (crossed/inverted book) → negative spread. The
        # function must return the signed value as-is (NOT abs()) so the
        # gate blocks (negative < min_spread_pct is trivially true) instead
        # of waving through a pathological book.
        controller = self._make_controller()
        result = controller._compute_spread_pct(
            external_best_ask=Decimal("0.4380"),
            best_bid=Decimal("0.4400"),
        )
        self.assertLess(result, Decimal("0"))

    # --- Critical: formula contract (negative assertions) ---

    def test_denominator_is_external_best_ask_not_best_bid(self):
        # CRITICAL: pin the denominator with a differential expectation.
        # ask=0.50, bid=0.40:
        #   correct (ask denom):  (0.50 - 0.40) / 0.50 = 0.20
        #   wrong   (bid denom):  (0.50 - 0.40) / 0.40 = 0.25
        # A typo swapping the denominator silently shifts gate behavior;
        # the not-equal assertion catches that exact regression.
        controller = self._make_controller()
        result = controller._compute_spread_pct(
            external_best_ask=Decimal("0.50"),
            best_bid=Decimal("0.40"),
        )
        self.assertEqual(result, Decimal("0.20"))
        self.assertNotEqual(result, Decimal("0.25"))

    def test_denominator_is_not_mid_price(self):
        # CRITICAL: a common alt-formulation uses mid-price as denominator.
        # ask=1.00, bid=0.60 → mid = 0.80
        #   correct (ask denom): (1.00 - 0.60) / 1.00 = 0.40
        #   wrong   (mid denom): (1.00 - 0.60) / 0.80 = 0.50
        # Differential test pins the ask-denominator choice.
        controller = self._make_controller()
        result = controller._compute_spread_pct(
            external_best_ask=Decimal("1.00"),
            best_bid=Decimal("0.60"),
        )
        self.assertEqual(result, Decimal("0.40"))
        self.assertNotEqual(result, Decimal("0.50"))

    def test_returns_decimal_not_float(self):
        # CRITICAL: the gate compares spread_pct < self.config.min_spread_pct
        # where min_spread_pct is a Decimal. If a refactor ever returns a
        # float, sub-tick precision drift could flip the gate decision at
        # the boundary. Pin the Decimal return type explicitly.
        controller = self._make_controller()
        result = controller._compute_spread_pct(
            external_best_ask=Decimal("0.42"),
            best_bid=Decimal("0.40"),
        )
        self.assertIsInstance(result, Decimal)
        self.assertNotIsInstance(result, float)

    def test_raises_on_zero_ask_denominator(self):
        # CONTRACT: the docstring says "Caller must ensure external_best_ask > 0".
        # If someone removes that caller-side guard, division by zero MUST
        # raise — not silently return 0/inf, which would either bypass the
        # gate (returning 0 → 0 < 0.02 fires gate, fine) or worse, return
        # something falsy that confuses callers. Pin that the function
        # itself does NOT defensively swallow — defense lives at the call
        # site (_compute_target_price already guards external_best_ask <= 0).
        controller = self._make_controller()
        with self.assertRaises(ZeroDivisionError):
            controller._compute_spread_pct(
                external_best_ask=Decimal("0"),
                best_bid=Decimal("0.42"),
            )

    # --- Precision / regression ---

    def test_exact_decimal_precision_preserved(self):
        # Tick-precision math: bid=0.4380, ask=0.4385 → diff=0.0005, then
        # 0.0005 / 0.4385. The result must be the EXACT Decimal computation
        # with no float rounding. If anyone converts intermediates to float,
        # this differential expectation breaks.
        controller = self._make_controller()
        result = controller._compute_spread_pct(
            external_best_ask=Decimal("0.4385"),
            best_bid=Decimal("0.4380"),
        )
        expected = (Decimal("0.4385") - Decimal("0.4380")) / Decimal("0.4385")
        self.assertEqual(result, expected)

    def test_scale_invariant_across_price_magnitudes(self):
        # Sanity: a ~2% spread on a $0.5 asset returns the same ratio as a
        # ~2% spread on a $5000 asset. Differential test confirms the
        # formula is truly ratio-based, not absolute-difference-based —
        # i.e., the same gate threshold works across XNO and BTC pairs.
        controller = self._make_controller()
        low = controller._compute_spread_pct(
            external_best_ask=Decimal("0.51"),
            best_bid=Decimal("0.50"),
        )
        high = controller._compute_spread_pct(
            external_best_ask=Decimal("5100"),
            best_bid=Decimal("5000"),
        )
        self.assertEqual(low, high)


class TestBBOPegSellComputeTargetPriceGate(unittest.TestCase):
    """Tests for _compute_target_price focused on the anti-spoof gate.

    Sibling to TestBBOPegSellComputeTargetPrice, which holds the gate
    OFF (min_spread_pct=0) to isolate peg/quantize math. This class flips
    the gate ON and pins:
      - Gate-fire short-circuit (returns None, skips quantize entirely)
      - Strict-less-than threshold boundary (spread == threshold → allow)
      - Custom thresholds actually drive behavior (knob not hardcoded)
      - State-transition logging from inside the full flow
      - Guard ordering (bid-validity short-circuits BEFORE gate)
    """

    def _make_controller(
        self,
        *,
        min_spread_pct: Decimal = Decimal("0.02"),
        quantize_side_effect=None,
    ) -> BBOPegSellController:
        """Builds a controller with a configurable gate threshold. Stashes
        the mocked MDP and log mock so tests can assert on quantize calls
        and log emissions without going through bound attributes.
        """
        config = BBOPegSellConfig(
            id="test",
            controller_name="bbo_peg_sell",
            connector_name="htx",
            trading_pair="XNO-USDT",
            total_amount_quote=Decimal("20"),
            update_interval=0.5,
            min_spread_pct=min_spread_pct,
        )
        market_data_provider = MagicMock(spec=MarketDataProvider)
        market_data_provider.quantize_order_price.side_effect = (
            quantize_side_effect or (lambda _c, _p, price: price)
        )
        self.market_data_provider_mock = market_data_provider
        controller = BBOPegSellController(
            config=config,
            market_data_provider=market_data_provider,
            actions_queue=AsyncMock(spec=asyncio.Queue),
        )
        log_mock = MagicMock()
        setattr(controller, "logger", MagicMock(return_value=log_mock))
        self.log_mock = log_mock
        return controller

    # --- Gate fire / clear branches ---

    def test_gate_fires_returns_none_when_spread_below_threshold(self):
        # The headline spoof-defense path: ask 1 tick above bid (spread
        # ≈0.023%) vs default 2% threshold → blocked → None. This is what
        # protects the bot from pegging right next to a spoofer's fake ask.
        controller = self._make_controller(min_spread_pct=Decimal("0.02"))
        result = controller._compute_target_price(
            external_best_ask=Decimal("0.4296"),
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.4295"),
        )
        self.assertIsNone(result)

    def test_gate_clears_returns_candidate_when_spread_above_threshold(self):
        # Healthy 5% spread vs 2% threshold → gate doesn't fire → return
        # the pegged candidate (1.00 - 0.0001).
        controller = self._make_controller(min_spread_pct=Decimal("0.02"))
        result = controller._compute_target_price(
            external_best_ask=Decimal("1.00"),
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.95"),
        )
        self.assertEqual(result, Decimal("0.9999"))

    # --- Threshold boundary (strict-less-than contract) ---

    def test_spread_exactly_at_threshold_does_not_fire_gate(self):
        # CRITICAL: gate uses `spread_pct < self.config.min_spread_pct`
        # (strict). Spread == threshold → ALLOWED. If someone changes < to
        # <=, this fails. Pins the deliberate "at the safety margin is fine,
        # below it is not" choice.
        # ask=1.00, bid=0.98 → spread = 0.02 == threshold 0.02 → allowed.
        controller = self._make_controller(min_spread_pct=Decimal("0.02"))
        result = controller._compute_target_price(
            external_best_ask=Decimal("1.00"),
            tick=Decimal("0.01"),
            best_bid=Decimal("0.98"),
        )
        self.assertEqual(result, Decimal("0.99"))

    def test_spread_just_below_threshold_fires_gate(self):
        # Differential vs the equality test: spread one ulp below threshold
        # MUST fire. Pins the strict-less-than as a tight boundary.
        # spread = (1.00 - 0.9801) / 1.00 = 0.0199 < 0.02 → blocked.
        controller = self._make_controller(min_spread_pct=Decimal("0.02"))
        result = controller._compute_target_price(
            external_best_ask=Decimal("1.00"),
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.9801"),
        )
        self.assertIsNone(result)

    # --- Threshold knob: config is actually honored ---

    def test_gate_disabled_when_min_spread_pct_is_zero(self):
        # Confirm the gate-off escape hatch used by TestBBOPegSellComputeTargetPrice
        # is real. A 0.4% spread that would block at 2% passes here.
        # spread = (0.4296 - 0.4280) / 0.4296 ≈ 0.00372, candidate = 0.4295.
        controller = self._make_controller(min_spread_pct=Decimal("0"))
        result = controller._compute_target_price(
            external_best_ask=Decimal("0.4296"),
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.4280"),
        )
        self.assertEqual(result, Decimal("0.4295"))

    def test_stricter_threshold_blocks_normally_acceptable_spread(self):
        # 3% spread is healthy at the default 2% threshold but BLOCKED at a
        # custom 5% threshold. Differential pins that the config value drives
        # the decision (not a hardcoded 2% somewhere).
        # spread = (1.00 - 0.97) / 1.00 = 0.03 < 0.05 → blocked.
        controller = self._make_controller(min_spread_pct=Decimal("0.05"))
        result = controller._compute_target_price(
            external_best_ask=Decimal("1.00"),
            tick=Decimal("0.01"),
            best_bid=Decimal("0.97"),
        )
        self.assertIsNone(result)

    # --- Critical: order of operations (gate short-circuits BEFORE quantize) ---

    def test_gate_fires_does_not_call_quantize(self):
        # CRITICAL: when the gate fires, the function must return BEFORE
        # calling quantize_order_price. On some MDPs the quantize call
        # involves a HTTP roundtrip — wasting one per blocked tick adds up
        # in tight markets. Pin the short-circuit.
        controller = self._make_controller(min_spread_pct=Decimal("0.02"))
        controller._compute_target_price(
            external_best_ask=Decimal("0.4296"),
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.4295"),
        )
        self.market_data_provider_mock.quantize_order_price.assert_not_called()

    def test_gate_clears_calls_quantize(self):
        # Mirror of above: healthy spread → quantize IS called. Together
        # with the negative test, pins the conditional ordering.
        controller = self._make_controller(min_spread_pct=Decimal("0.02"))
        controller._compute_target_price(
            external_best_ask=Decimal("1.00"),
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.95"),
        )
        self.market_data_provider_mock.quantize_order_price.assert_called_once()

    # --- Edge: pathological spreads ---

    def test_zero_spread_fires_gate(self):
        # Equal bid and ask → spread = 0 < any positive threshold → blocked.
        # Note: the crossing-ask guard further down would ALSO catch this
        # (candidate=ask), but the gate fires first in the function flow —
        # the None return here is gate-driven, not cross-driven.
        controller = self._make_controller(min_spread_pct=Decimal("0.02"))
        result = controller._compute_target_price(
            external_best_ask=Decimal("0.4380"),
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.4380"),
        )
        self.assertIsNone(result)

    def test_inverted_book_fires_gate(self):
        # Ask BELOW bid (crossed/inverted) → negative spread → blocked.
        # Pins that the strict-less-than handles negative spreads correctly
        # (negative < any positive threshold is trivially true).
        controller = self._make_controller(min_spread_pct=Decimal("0.02"))
        result = controller._compute_target_price(
            external_best_ask=Decimal("0.4380"),
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.4400"),
        )
        self.assertIsNone(result)

    # --- Logging side-effect (state transition rate-limiting) ---

    def test_gate_blocked_log_emitted_on_first_block(self):
        # Cold start (_gate_blocked=False initially) → gate fires → state
        # transitions False→True → exactly one log line with "gate_blocked".
        controller = self._make_controller(min_spread_pct=Decimal("0.02"))
        controller._compute_target_price(
            external_best_ask=Decimal("0.4296"),
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.4295"),
        )
        self.log_mock.info.assert_called_once()
        log_msg = self.log_mock.info.call_args[0][0]
        self.assertIn("gate_blocked", log_msg)

    def test_gate_not_logged_when_blocked_state_persists(self):
        # Two consecutive blocked ticks → log fires ONCE (on the transition),
        # not twice. Rate-limiting prevents log spam in naturally-tight
        # markets that sit below the threshold for sustained periods.
        controller = self._make_controller(min_spread_pct=Decimal("0.02"))
        for _ in range(2):
            controller._compute_target_price(
                external_best_ask=Decimal("0.4296"),
                tick=Decimal("0.0001"),
                best_bid=Decimal("0.4295"),
            )
        self.assertEqual(self.log_mock.info.call_count, 1)

    def test_gate_not_logged_when_blocked_state_persists_across_different_prices(self):
        # CRITICAL: rate-limiting tracks ONLY the blocked/cleared boolean,
        # NOT the (bid, ask) tuple. Two blocked ticks with DIFFERENT prices
        # still produce one log line. Catches accidental "log every blocked
        # price change" regressions where someone keys the cache on prices.
        controller = self._make_controller(min_spread_pct=Decimal("0.02"))
        controller._compute_target_price(
            external_best_ask=Decimal("0.4296"),
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.4295"),
        )
        controller._compute_target_price(
            external_best_ask=Decimal("0.5001"),
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.5000"),
        )
        self.assertEqual(self.log_mock.info.call_count, 1)

    def test_gate_logs_blocked_then_cleared_on_state_transitions(self):
        # Block then clear → two log lines, in order: "gate_blocked",
        # "gate_cleared". Pins both transition directions emit, and the
        # message body distinguishes them (so log readers can grep).
        controller = self._make_controller(min_spread_pct=Decimal("0.02"))
        controller._compute_target_price(
            external_best_ask=Decimal("0.4296"),
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.4295"),
        )
        controller._compute_target_price(
            external_best_ask=Decimal("1.00"),
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.95"),
        )
        self.assertEqual(self.log_mock.info.call_count, 2)
        first_msg = self.log_mock.info.call_args_list[0][0][0]
        second_msg = self.log_mock.info.call_args_list[1][0][0]
        self.assertIn("gate_blocked", first_msg)
        self.assertIn("gate_cleared", second_msg)

    def test_no_log_when_gate_stays_cleared_from_cold_start(self):
        # Cold start has _gate_blocked=False initial. If the very first call
        # also produces "not blocked", there's no state transition → no log.
        # Pins that healthy markets at startup produce ZERO gate log noise.
        controller = self._make_controller(min_spread_pct=Decimal("0.02"))
        controller._compute_target_price(
            external_best_ask=Decimal("1.00"),
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.95"),
        )
        self.log_mock.info.assert_not_called()

    # --- Critical: guard ordering (bid-validity runs BEFORE gate) ---

    def test_zero_ask_short_circuits_before_gate(self):
        # CRITICAL: external_best_ask <= 0 → return None BEFORE the gate is
        # evaluated. If the gate ran first, _compute_spread_pct would raise
        # ZeroDivisionError. The clean None return AND absence of log line
        # together prove the bid-validity guard runs first.
        controller = self._make_controller(min_spread_pct=Decimal("0.02"))
        result = controller._compute_target_price(
            external_best_ask=Decimal("0"),
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.42"),
        )
        self.assertIsNone(result)
        self.log_mock.info.assert_not_called()

    def test_none_ask_short_circuits_before_gate(self):
        # Same reasoning for None bid — must short-circuit before gate.
        # If gate were entered, _compute_spread_pct(None, ...) would crash
        # with TypeError. Pins the same guard ordering for the None case.
        controller = self._make_controller(min_spread_pct=Decimal("0.02"))
        result = controller._compute_target_price(
            external_best_ask=None,
            tick=Decimal("0.0001"),
            best_bid=Decimal("0.42"),
        )
        self.assertIsNone(result)
        self.log_mock.info.assert_not_called()

    def test_zero_bid_short_circuits_before_gate(self):
        # Falsy best_bid (0) → return None BEFORE the gate. If gate ran first
        # with bid=0, spread_pct = ask/ask = 1, which would clear the gate
        # spuriously and could let us quote against a missing-bid book. The
        # early return ensures we just bail cleanly on missing market data
        # instead of misclassifying it.
        controller = self._make_controller(min_spread_pct=Decimal("0.02"))
        result = controller._compute_target_price(
            external_best_ask=Decimal("0.4380"),
            tick=Decimal("0.0001"),
            best_bid=Decimal("0"),
        )
        self.assertIsNone(result)
        self.log_mock.info.assert_not_called()


class TestBBOPegSellWalkAsksForFirstExternal(unittest.TestCase):
    """Direct unit tests for _walk_asks_for_first_external in isolation.

    The result-finding behavior is also covered indirectly through
    TestBBOPegSellExternalBestAsk, but the top_levels return contract
    (size cap at 5, top-down ordering, float-tuple shape) is only
    tested directly here.
    """

    # --- Result return value ---

    def test_empty_book_returns_none_result_and_empty_top_levels(self):
        controller, _ = _make_controller_for_walker(ask_levels=[])
        result, top_levels = controller._walk_asks_for_first_external({})
        self.assertIsNone(result)
        self.assertEqual(top_levels, [])

    def test_single_external_level_returns_top_price(self):
        controller, _ = _make_controller_for_walker(
            ask_levels=[(Decimal("0.4380"), Decimal("100"))]
        )
        result, _ = controller._walk_asks_for_first_external({})
        self.assertEqual(result, Decimal("0.4380"))

    def test_returns_none_when_all_levels_fully_ours(self):
        # 3 levels, all fully ours via my_volume_by_price → no external → None.
        controller, _ = _make_controller_for_walker(
            ask_levels=[
                (Decimal("0.4380"), Decimal("100")),
                (Decimal("0.4379"), Decimal("50")),
                (Decimal("0.4378"), Decimal("200")),
            ]
        )
        my_volume = {
            Decimal("0.4380"): Decimal("100"),
            Decimal("0.4379"): Decimal("50"),
            Decimal("0.4378"): Decimal("200"),
        }
        result, _ = controller._walk_asks_for_first_external(my_volume)
        self.assertIsNone(result)

    def test_caps_at_ten_levels_for_result(self):
        # 11 levels; first 10 fully ours, 11th would be external. The cap
        # (i >= 10 break) stops the walker BEFORE seeing the 11th → None.
        ask_levels = [(Decimal(f"0.43{80 - i:02d}"), Decimal("10")) for i in range(10)]
        ask_levels.append((Decimal("0.4370"), Decimal("999")))  # 11th, external
        my_volume = {p: Decimal("10") for p, _ in ask_levels[:10]}
        controller, _ = _make_controller_for_walker(ask_levels=ask_levels)
        result, _ = controller._walk_asks_for_first_external(my_volume)
        self.assertIsNone(result)

    # --- top_levels return contract ---

    def test_top_levels_capped_at_five_when_book_is_larger(self):
        # 7 levels in book → top_levels exactly 5 (top-5 capture for logging).
        ask_levels = [(Decimal(f"0.43{80 - i:02d}"), Decimal("100")) for i in range(7)]
        controller, _ = _make_controller_for_walker(ask_levels=ask_levels)
        _, top_levels = controller._walk_asks_for_first_external({})
        self.assertEqual(len(top_levels), 5)

    def test_top_levels_contains_all_when_book_has_fewer_than_five(self):
        # 3 levels → all 3 in top_levels (no padding, no truncation).
        controller, _ = _make_controller_for_walker(
            ask_levels=[
                (Decimal("0.4380"), Decimal("100")),
                (Decimal("0.4379"), Decimal("50")),
                (Decimal("0.4378"), Decimal("200")),
            ]
        )
        _, top_levels = controller._walk_asks_for_first_external({})
        self.assertEqual(len(top_levels), 3)

    def test_top_levels_preserves_top_down_order(self):
        # First entry = top of book, descending from there.
        controller, _ = _make_controller_for_walker(
            ask_levels=[
                (Decimal("0.4380"), Decimal("100")),
                (Decimal("0.4379"), Decimal("50")),
                (Decimal("0.4378"), Decimal("200")),
            ]
        )
        _, top_levels = controller._walk_asks_for_first_external({})
        expected = [
            (float(Decimal("0.4380")), float(Decimal("100"))),
            (float(Decimal("0.4379")), float(Decimal("50"))),
            (float(Decimal("0.4378")), float(Decimal("200"))),
        ]
        self.assertEqual(top_levels, expected)

    def test_top_levels_entries_are_float_tuples(self):
        # Pin the (float, float) shape — important for log message format.
        controller, _ = _make_controller_for_walker(
            ask_levels=[(Decimal("0.4380"), Decimal("100"))]
        )
        _, top_levels = controller._walk_asks_for_first_external({})
        self.assertEqual(len(top_levels), 1)
        price, amount = top_levels[0]
        self.assertIsInstance(price, float)
        self.assertIsInstance(amount, float)

    # --- Own-volume interaction ---

    def test_subtracts_own_volume_from_external_amount(self):
        # Without subtraction, the walker would see 100 > 0 and return 0.4380.
        # WITH subtraction (100 - 100 = 0, not > 0), it skips the top and
        # returns 0.4379 instead. That differential outcome is what actually
        # pins the subtraction contract — equal book/own amounts at top must
        # cause the walker to drop down.
        controller, _ = _make_controller_for_walker(
            ask_levels=[
                (Decimal("0.4380"), Decimal("100")),
                (Decimal("0.4379"), Decimal("50")),
            ]
        )
        my_volume = {Decimal("0.4380"): Decimal("100")}
        result, _ = controller._walk_asks_for_first_external(my_volume)
        self.assertEqual(result, Decimal("0.4379"))

    # --- Negative cases (what the walker must NOT do) ---

    def test_never_returns_a_fully_owned_price_in_mixed_book(self):
        # Critical safety invariant: the walker must NEVER return a price
        # we fully own. If it did, we'd peg one tick above our own quote,
        # which is the self-anchoring runaway the walker exists to prevent.
        # Book has fully-owned levels at 0.4380 and 0.4378, external at 0.4379.
        controller, _ = _make_controller_for_walker(
            ask_levels=[
                (Decimal("0.4380"), Decimal("100")),  # fully ours
                (Decimal("0.4379"), Decimal("50")),  # external
                (Decimal("0.4378"), Decimal("200")),  # fully ours
            ]
        )
        my_volume = {
            Decimal("0.4380"): Decimal("100"),
            Decimal("0.4378"): Decimal("200"),
        }
        result, _ = controller._walk_asks_for_first_external(my_volume)
        self.assertNotEqual(result, Decimal("0.4380"))
        self.assertNotEqual(result, Decimal("0.4378"))


class TestBBOPegSellComputeOwnVolumeByPrice(unittest.TestCase):
    """Direct unit tests for _compute_own_volume_by_price in isolation.

    Aggregates active executors' volume into a per-price dict. The filters
    are also exercised indirectly through TestBBOPegSellExternalBestAsk,
    but the dict-shape contract and aggregation semantics are pinned here.
    """

    # --- Positive cases ---

    def test_empty_executors_info_returns_empty_dict(self):
        controller, _ = _make_controller_for_walker(executors=[])
        self.assertEqual(controller._compute_own_volume_by_price(), {})

    def test_single_active_executor_returns_single_entry(self):
        controller, _ = _make_controller_for_walker(
            executors=[_fake_executor(price=Decimal("0.4380"), amount=Decimal("50"))]
        )
        result = controller._compute_own_volume_by_price()
        self.assertEqual(result, {Decimal("0.4380"): Decimal("50")})

    def test_multiple_executors_different_prices_separate_entries(self):
        # Three executors at three different prices → three dict entries.
        controller, _ = _make_controller_for_walker(
            executors=[
                _fake_executor(price=Decimal("0.4380"), amount=Decimal("10")),
                _fake_executor(price=Decimal("0.4379"), amount=Decimal("20")),
                _fake_executor(price=Decimal("0.4378"), amount=Decimal("30")),
            ]
        )
        result = controller._compute_own_volume_by_price()
        self.assertEqual(
            result,
            {
                Decimal("0.4380"): Decimal("10"),
                Decimal("0.4379"): Decimal("20"),
                Decimal("0.4378"): Decimal("30"),
            },
        )

    def test_multiple_executors_same_price_sum_volumes(self):
        # Two executors at the same price → summed.
        controller, _ = _make_controller_for_walker(
            executors=[
                _fake_executor(price=Decimal("0.4380"), amount=Decimal("30")),
                _fake_executor(price=Decimal("0.4380"), amount=Decimal("50")),
            ]
        )
        result = controller._compute_own_volume_by_price()
        self.assertEqual(result, {Decimal("0.4380"): Decimal("80")})

    # --- Filter rules (each rule pinned in isolation) ---

    def test_inactive_executor_excluded(self):
        controller, _ = _make_controller_for_walker(
            executors=[
                _fake_executor(
                    price=Decimal("0.4380"),
                    amount=Decimal("100"),
                    is_active=False,
                )
            ]
        )
        self.assertEqual(controller._compute_own_volume_by_price(), {})

    def test_non_order_executor_config_excluded(self):
        controller, _ = _make_controller_for_walker(
            executors=[
                _fake_executor(
                    price=Decimal("0.4380"),
                    amount=Decimal("100"),
                    use_order_executor_config=False,
                )
            ]
        )
        self.assertEqual(controller._compute_own_volume_by_price(), {})

    def test_executor_with_none_price_excluded(self):
        controller, _ = _make_controller_for_walker(
            executors=[_fake_executor(price=None, amount=Decimal("100"))]
        )
        self.assertEqual(controller._compute_own_volume_by_price(), {})

    # --- Negative / edge cases ---

    def test_returns_empty_dict_when_all_executors_filtered_out(self):
        # Mix of filter reasons — every executor excluded by a different rule.
        # All three filters must work together, not just individually.
        controller, _ = _make_controller_for_walker(
            executors=[
                _fake_executor(
                    price=Decimal("0.4380"),
                    amount=Decimal("100"),
                    is_active=False,
                ),
                _fake_executor(
                    price=Decimal("0.4379"),
                    amount=Decimal("100"),
                    use_order_executor_config=False,
                ),
                _fake_executor(price=None, amount=Decimal("100")),
            ]
        )
        self.assertEqual(controller._compute_own_volume_by_price(), {})

    def test_filtered_executor_price_does_not_appear_in_result_keys(self):
        # Inactive executor at 0.4380 → that price must NOT show up as a key.
        # If it did (with amount 0 or otherwise), the walker would see it
        # in my_volume_by_price.get(price, 0) and incorrectly subtract.
        controller, _ = _make_controller_for_walker(
            executors=[
                _fake_executor(
                    price=Decimal("0.4380"),
                    amount=Decimal("100"),
                    is_active=False,
                ),
                _fake_executor(price=Decimal("0.4379"), amount=Decimal("50")),
            ]
        )
        result = controller._compute_own_volume_by_price()
        self.assertNotIn(Decimal("0.4380"), result)
        # Sanity: the active one IS there.
        self.assertIn(Decimal("0.4379"), result)

    def test_aggregation_does_not_overwrite_with_last_value(self):
        # CRITICAL: three executors at the same price with amounts 10, 20, 30.
        # If aggregation accidentally overwrites instead of summing, result
        # would be 30 (the last). Correct sum is 60. The differential pins
        # the contract — passes only if += is used, not =.
        controller, _ = _make_controller_for_walker(
            executors=[
                _fake_executor(price=Decimal("0.4380"), amount=Decimal("10")),
                _fake_executor(price=Decimal("0.4380"), amount=Decimal("20")),
                _fake_executor(price=Decimal("0.4380"), amount=Decimal("30")),
            ]
        )
        result = controller._compute_own_volume_by_price()
        self.assertNotEqual(result[Decimal("0.4380")], Decimal("30"))
        self.assertEqual(result[Decimal("0.4380")], Decimal("60"))

    # --- Cancel-debounce: pending cancels are included in own_volume ---

    def test_pending_cancel_within_window_included_in_own_volume(self):
        # CORE FIX: a just-cancelled order at 0.4414 still echoes in the
        # exchange book WebSocket due to cancel-lag. We track it in
        # _pending_cancels; _compute_own_volume_by_price must include it
        # so the walker keeps subtracting it from the book. Without this,
        # the walker chases its own ghost (the production self-chase loop).
        controller, _ = _make_controller_for_walker(executors=[])
        controller.market_data_provider.time.return_value = 1000.0  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
        controller._pending_cancels["pending-1"] = (
            Decimal("0.4414"),
            Decimal("45.31"),
            1002.0,  # until = now+2s, still valid
        )
        result = controller._compute_own_volume_by_price()
        self.assertEqual(result, {Decimal("0.4414"): Decimal("45.31")})

    def test_pending_cancel_past_expiry_excluded_from_own_volume(self):
        # Expired pending cancels MUST be filtered out — once the debounce
        # window has passed, the exchange book should have caught up and the
        # ghost order is presumed gone. If we keep subtracting it forever
        # we'd permanently lose visibility of that price level.
        controller, _ = _make_controller_for_walker(executors=[])
        controller.market_data_provider.time.return_value = 1000.0  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
        controller._pending_cancels["expired"] = (
            Decimal("0.4414"),
            Decimal("45.31"),
            999.0,  # until < now → expired
        )
        result = controller._compute_own_volume_by_price()
        self.assertEqual(result, {})

    def test_pending_cancel_exactly_at_expiry_excluded(self):
        # Boundary: until == now is EXCLUDED (strict > comparison).
        # Pin the strict-greater-than so a refactor to >= doesn't silently
        # extend the debounce window by one tick.
        controller, _ = _make_controller_for_walker(executors=[])
        controller.market_data_provider.time.return_value = 1000.0  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
        controller._pending_cancels["edge"] = (
            Decimal("0.4414"),
            Decimal("45.31"),
            1000.0,  # until == now → drop
        )
        result = controller._compute_own_volume_by_price()
        self.assertEqual(result, {})

    def test_pending_cancel_at_same_price_as_active_executor_sums(self):
        # A pending cancel at the same price as an existing active executor
        # must SUM with the active volume, not overwrite or be overwritten.
        # Real scenario: we just placed at 0.4296 (active) while a previous
        # 0.4296 quote is still pending-cancel during the lag window.
        controller, _ = _make_controller_for_walker(
            executors=[_fake_executor(price=Decimal("0.4296"), amount=Decimal("46"))]
        )
        controller.market_data_provider.time.return_value = 1000.0  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
        controller._pending_cancels["pending-old"] = (
            Decimal("0.4296"),
            Decimal("46.5"),
            1002.0,
        )
        result = controller._compute_own_volume_by_price()
        self.assertEqual(result, {Decimal("0.4296"): Decimal("92.5")})

    def test_multiple_pending_cancels_at_same_price_sum(self):
        # Edge: two pending cancels at the same price (e.g., two ticks of
        # cancel-replace where both are still echoing). They sum, just like
        # active executors at the same price do.
        controller, _ = _make_controller_for_walker(executors=[])
        controller.market_data_provider.time.return_value = 1000.0  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
        controller._pending_cancels["p1"] = (
            Decimal("0.4414"),
            Decimal("45"),
            1002.0,
        )
        controller._pending_cancels["p2"] = (
            Decimal("0.4414"),
            Decimal("46"),
            1002.0,
        )
        result = controller._compute_own_volume_by_price()
        self.assertEqual(result, {Decimal("0.4414"): Decimal("91")})


class TestBBOPegSellRecordPendingCancels(unittest.TestCase):
    """Direct unit tests for _record_pending_cancels.

    Pins the cancel-debounce recording contract: when the controller emits
    a Stop, record (price, amount, until=now+cancel_debounce_seconds) keyed
    by executor_id so _compute_own_volume_by_price can keep subtracting the
    cancelled volume from the exchange book during the cancel-propagation
    lag window.
    """

    def _make_controller(
        self,
        *,
        cancel_debounce_seconds: float = 2.0,
        now: float = 1000.0,
    ) -> BBOPegSellController:
        """Builds a controller with mocked time. cancel_debounce_seconds
        defaults to 2.0 (production default); tests can pass 0 to disable
        recording or other values for boundary checks.
        """
        config = BBOPegSellConfig(
            id="test",
            controller_name="bbo_peg_sell",
            connector_name="htx",
            trading_pair="XNO-USDT",
            total_amount_quote=Decimal("20"),
            update_interval=0.5,
            cancel_debounce_seconds=cancel_debounce_seconds,
        )
        market_data_provider = MagicMock(spec=MarketDataProvider)
        market_data_provider.time.return_value = now
        return BBOPegSellController(
            config=config,
            market_data_provider=market_data_provider,
            actions_queue=AsyncMock(spec=asyncio.Queue),
        )

    # --- Basic recording ---

    def test_records_single_executor_with_correct_fields(self):
        # The fundamental contract: each stopped executor becomes an entry
        # keyed by id with value (price, amount, now + debounce_seconds).
        controller = self._make_controller(cancel_debounce_seconds=2.0, now=1000.0)
        executor = _fake_executor(price=Decimal("0.4414"), amount=Decimal("45.31"))
        executor.id = "exec-1"
        controller._record_pending_cancels([executor])
        self.assertEqual(len(controller._pending_cancels), 1)
        self.assertIn("exec-1", controller._pending_cancels)
        price, amount, until = controller._pending_cancels["exec-1"]
        self.assertEqual(price, Decimal("0.4414"))
        self.assertEqual(amount, Decimal("45.31"))
        self.assertEqual(until, 1002.0)

    def test_records_multiple_executors_separately(self):
        # Two executors with distinct ids → two separate entries.
        controller = self._make_controller()
        e1 = _fake_executor(price=Decimal("0.4414"), amount=Decimal("45"))
        e1.id = "exec-1"
        e2 = _fake_executor(price=Decimal("0.4415"), amount=Decimal("46"))
        e2.id = "exec-2"
        controller._record_pending_cancels([e1, e2])
        self.assertEqual(len(controller._pending_cancels), 2)
        self.assertIn("exec-1", controller._pending_cancels)
        self.assertIn("exec-2", controller._pending_cancels)

    def test_empty_list_does_not_add_entries(self):
        # Edge: empty input → no recording → dict unchanged.
        controller = self._make_controller()
        controller._record_pending_cancels([])
        self.assertEqual(controller._pending_cancels, {})

    # --- Until timestamp math ---

    def test_until_uses_market_data_provider_time(self):
        # The mock time source must be used, not wall-clock. Pin via a
        # specific value so a refactor to time.time() would be caught.
        controller = self._make_controller(now=5555.0, cancel_debounce_seconds=3.0)
        executor = _fake_executor(price=Decimal("0.44"), amount=Decimal("10"))
        executor.id = "e"
        controller._record_pending_cancels([executor])
        _, _, until = controller._pending_cancels["e"]
        self.assertEqual(until, 5558.0)

    def test_until_reflects_config_debounce_seconds(self):
        # If config changes the debounce window, until math updates with it.
        # Use a distinctive value (7.5s) so the assertion is unambiguous.
        controller = self._make_controller(now=100.0, cancel_debounce_seconds=7.5)
        executor = _fake_executor(price=Decimal("0.44"), amount=Decimal("10"))
        executor.id = "e"
        controller._record_pending_cancels([executor])
        _, _, until = controller._pending_cancels["e"]
        self.assertEqual(until, 107.5)

    # --- Filter rules (mirror _compute_own_volume_by_price filters) ---

    def test_skips_executors_with_non_order_executor_config(self):
        # A non-OrderExecutorConfig executor shouldn't be recorded at all.
        # Otherwise the dict would hold an entry with no usable (price,
        # amount), polluting downstream consumers.
        controller = self._make_controller()
        wrong_cfg = _fake_executor(
            price=Decimal("0.44"),
            amount=Decimal("10"),
            use_order_executor_config=False,
        )
        wrong_cfg.id = "wrong"
        controller._record_pending_cancels([wrong_cfg])
        self.assertEqual(controller._pending_cancels, {})

    def test_skips_executors_with_none_price(self):
        # OrderExecutorConfig but price=None → skip. We can't add to
        # my_volume_by_price keyed by a None price, so don't record it.
        controller = self._make_controller()
        none_price = _fake_executor(price=None, amount=Decimal("10"))
        none_price.id = "noprice"
        controller._record_pending_cancels([none_price])
        self.assertEqual(controller._pending_cancels, {})

    # --- Eviction and dedupe behavior ---

    def test_evicts_expired_entries_on_record(self):
        # Opportunistic GC: when we record new pending cancels, expired
        # entries (until <= now) are dropped from the dict. Keeps memory
        # bounded over long sessions without needing a separate sweep.
        controller = self._make_controller(now=1000.0)
        controller._pending_cancels["old"] = (
            Decimal("0.40"),
            Decimal("10"),
            999.0,  # expired
        )
        controller._pending_cancels["future"] = (
            Decimal("0.41"),
            Decimal("10"),
            1001.0,  # still valid
        )
        e = _fake_executor(price=Decimal("0.42"), amount=Decimal("10"))
        e.id = "new"
        controller._record_pending_cancels([e])
        self.assertNotIn("old", controller._pending_cancels)
        self.assertIn("future", controller._pending_cancels)
        self.assertIn("new", controller._pending_cancels)

    def test_re_recording_same_executor_id_refreshes_until(self):
        # If the framework keeps an executor in is_active=True across ticks
        # (cancel in flight), _record_pending_cancels gets called multiple
        # times for the same id. The re-record refreshes until, keeping the
        # debounce alive as long as the controller keeps stopping it.
        controller = self._make_controller(cancel_debounce_seconds=2.0, now=1000.0)
        e = _fake_executor(price=Decimal("0.44"), amount=Decimal("10"))
        e.id = "exec-1"
        controller._record_pending_cancels([e])
        self.assertEqual(controller._pending_cancels["exec-1"][2], 1002.0)
        # Now advance time and re-record.
        controller.market_data_provider.time.return_value = 1001.5  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
        controller._record_pending_cancels([e])
        self.assertEqual(controller._pending_cancels["exec-1"][2], 1003.5)

    def test_zero_debounce_seconds_records_already_expired_entry(self):
        # Setting cancel_debounce_seconds=0 effectively disables the feature:
        # records an entry with until=now, which is immediately filtered out
        # by _compute_own_volume_by_price's strict-greater-than check.
        # Useful as an escape hatch and pinning the disable semantics.
        controller = self._make_controller(cancel_debounce_seconds=0.0, now=1000.0)
        e = _fake_executor(price=Decimal("0.44"), amount=Decimal("10"))
        e.id = "exec-1"
        controller._record_pending_cancels([e])
        # Entry was recorded, but with until == now → filtered downstream.
        self.assertEqual(controller._pending_cancels["exec-1"][2], 1000.0)


class TestBBOPegSellLogExternalAskChange(unittest.TestCase):
    """Direct unit tests for _log_external_ask_change in isolation.

    Pins the change-detection log behavior: when log fires, when it
    doesn't, cache updates, transitions through None, and the
    Decimal-to-float conversion in the log payload.
    """

    def _make_controller(self) -> BBOPegSellController:
        """Builds a controller and stashes the log mock on self.log_mock so
        tests can assert on .info calls without going through the bound
        controller.logger() method.
        """
        config = BBOPegSellConfig(
            id="test",
            controller_name="bbo_peg_sell",
            connector_name="htx",
            trading_pair="XNO-USDT",
            total_amount_quote=Decimal("20"),
            update_interval=0.5,
        )
        market_data_provider = MagicMock(spec=MarketDataProvider)
        log_mock = MagicMock()
        controller = BBOPegSellController(
            config=config,
            market_data_provider=market_data_provider,
            actions_queue=AsyncMock(spec=asyncio.Queue),
        )
        setattr(controller, "logger", MagicMock(return_value=log_mock))
        self.log_mock = log_mock
        return controller

    # --- Log emission decisions ---

    def test_does_not_log_when_result_equals_cache(self):
        # The whole point of the change-detection cache — no spam when the
        # walker keeps picking the same external bid tick after tick.
        controller = self._make_controller()
        controller._last_logged_external_best_ask = Decimal("0.4380")
        controller._log_external_ask_change(
            result=Decimal("0.4380"),
            top_levels=[],
            my_volume_by_price={},
        )
        self.log_mock.info.assert_not_called()

    def test_logs_when_result_changes_from_none_to_value(self):
        # Initial detection — cache starts as None, first walker observation
        # is a real value. Must fire so the first peg target is auditable.
        controller = self._make_controller()
        self.assertIsNone(controller._last_logged_external_best_ask)
        controller._log_external_ask_change(
            result=Decimal("0.4380"),
            top_levels=[(0.4380, 100.0)],
            my_volume_by_price={},
        )
        self.log_mock.info.assert_called_once()

    def test_logs_when_result_changes_from_value_to_none(self):
        # Critical edge: the book empties (or every level becomes ours).
        # Must still log the transition — otherwise sudden silence in logs
        # could hide a serious state change during incident review.
        controller = self._make_controller()
        controller._last_logged_external_best_ask = Decimal("0.4380")
        controller._log_external_ask_change(
            result=None,
            top_levels=[],
            my_volume_by_price={},
        )
        self.log_mock.info.assert_called_once()

    def test_logs_when_result_changes_between_two_values(self):
        # Walker picks a new external bid tick over tick (e.g., a level
        # ahead of us was hit). Standard transition — must log.
        controller = self._make_controller()
        controller._last_logged_external_best_ask = Decimal("0.4380")
        controller._log_external_ask_change(
            result=Decimal("0.4379"),
            top_levels=[],
            my_volume_by_price={},
        )
        self.log_mock.info.assert_called_once()

    # --- Cache update behavior ---

    def test_cache_updates_to_new_result_after_logging(self):
        controller = self._make_controller()
        self.assertIsNone(controller._last_logged_external_best_ask)
        controller._log_external_ask_change(
            result=Decimal("0.4380"),
            top_levels=[],
            my_volume_by_price={},
        )
        self.assertEqual(controller._last_logged_external_best_ask, Decimal("0.4380"))

    # --- Log payload format (regression catchers) ---

    def test_log_message_includes_all_diagnostic_fields(self):
        # Forensic log line must carry all three pieces of state so future
        # readers can reconstruct what the walker saw.
        controller = self._make_controller()
        controller._log_external_ask_change(
            result=Decimal("0.4380"),
            top_levels=[(0.4380, 100.0), (0.4379, 50.0)],
            my_volume_by_price={Decimal("0.4380"): Decimal("30")},
        )
        log_message = self.log_mock.info.call_args[0][0]
        # Log prefix now includes the base asset for multi-token disambiguation.
        self.assertIn("[bbo_peg_sell ", log_message)
        self.assertIn("external_best_ask", log_message)
        self.assertIn("top5_asks", log_message)
        self.assertIn("my_volume", log_message)

    def test_my_volume_converted_to_float_in_log_message(self):
        # The function builds {float(k): float(v) ...} before logging.
        # If someone "simplifies" this by removing the float() calls, the
        # log shows Decimal repr ("Decimal('0.4380'): Decimal('30')") instead
        # of friendly floats — ugly and harder to grep. Pin the conversion.
        controller = self._make_controller()
        controller._log_external_ask_change(
            result=Decimal("0.4380"),
            top_levels=[],
            my_volume_by_price={Decimal("0.4380"): Decimal("30")},
        )
        log_message = self.log_mock.info.call_args[0][0]
        self.assertNotIn("Decimal", log_message)


class TestBBOPegSellLogActiveExecutorsSnapshot(unittest.TestCase):
    """Direct unit tests for _log_active_executors_snapshot.

    This log exists specifically for orphan diagnosis during live runs:
    if the exchange UI shows an order at price X but no log line ever
    lists X in `actives`, that order is orphaned from executors_info and
    the controller will never emit a Stop for it.

    Pins the rate-limiting (state-change only) so steady-state ticks don't
    spam the log, the initial-snapshot logging (so we always see the first
    state on cold start), and the executor filters (inactive / non-OrderExecutorConfig
    / None-price executors are excluded from the snapshot, matching the
    filters used elsewhere in the controller).
    """

    def _make_controller(self, executors=None) -> BBOPegSellController:
        controller, log_mock = _make_controller_for_walker(executors=executors)
        self.log_mock = log_mock
        return controller

    # --- Log emission decisions ---

    def test_logs_initial_snapshot_on_cold_start(self):
        # Cold start (_last_logged_actives is None initial). First populated
        # snapshot MUST log, so the live log gets a baseline reference state.
        executor = _fake_executor(price=Decimal("0.4295"), amount=Decimal("46"))
        executor.id = "exec-1"
        controller = self._make_controller(executors=[executor])
        controller._log_active_executors_snapshot()
        self.log_mock.info.assert_called_once()
        log_msg = self.log_mock.info.call_args[0][0]
        # Log prefix now includes the base asset; see analogous comment above.
        self.assertIn("[bbo_peg_sell ", log_msg)
        self.assertIn("actives=", log_msg)
        self.assertIn("exec-1", log_msg)
        self.assertIn("0.4295", log_msg)

    def test_does_not_log_when_snapshot_unchanged_across_ticks(self):
        # Two consecutive calls with identical executors → one log line.
        # Rate-limiting prevents log spam during the (frequent) steady state.
        executor = _fake_executor(price=Decimal("0.4295"), amount=Decimal("46"))
        executor.id = "exec-1"
        controller = self._make_controller(executors=[executor])
        controller._log_active_executors_snapshot()
        controller._log_active_executors_snapshot()
        self.assertEqual(self.log_mock.info.call_count, 1)

    def test_logs_when_executor_set_changes(self):
        # State change between ticks (new executor added) → second log fires.
        executor1 = _fake_executor(price=Decimal("0.4295"), amount=Decimal("46"))
        executor1.id = "exec-1"
        controller = self._make_controller(executors=[executor1])
        controller._log_active_executors_snapshot()
        executor2 = _fake_executor(price=Decimal("0.4296"), amount=Decimal("46"))
        executor2.id = "exec-2"
        controller.executors_info.append(executor2)
        controller._log_active_executors_snapshot()
        self.assertEqual(self.log_mock.info.call_count, 2)

    def test_logs_when_executor_price_changes(self):
        # Same executor_id but different price → different snapshot → log.
        # Catches the case where the executor is mutated rather than replaced.
        executor = _fake_executor(price=Decimal("0.4295"), amount=Decimal("46"))
        executor.id = "exec-1"
        controller = self._make_controller(executors=[executor])
        controller._log_active_executors_snapshot()
        executor.config.price = Decimal("0.4296")
        controller._log_active_executors_snapshot()
        self.assertEqual(self.log_mock.info.call_count, 2)

    def test_empty_initial_state_logs_then_silence_until_change(self):
        # Cold start with NO executors. First call emits an empty-snapshot
        # baseline log; subsequent calls (still empty) stay silent until
        # something changes.
        controller = self._make_controller(executors=[])
        controller._log_active_executors_snapshot()
        controller._log_active_executors_snapshot()
        self.assertEqual(self.log_mock.info.call_count, 1)
        log_msg = self.log_mock.info.call_args[0][0]
        self.assertIn("actives=[]", log_msg)

    # --- Filter rules (executor must show up only if active + valid) ---

    def test_filters_inactive_and_invalid_executors_from_snapshot(self):
        # The snapshot must mirror the controller's executors_info filters
        # exactly. If an inactive/wrong-config/None-price executor leaks
        # into the snapshot, an actually-active orphan at the same id
        # would be misidentified as known.
        ok = _fake_executor(price=Decimal("0.4295"), amount=Decimal("46"))
        ok.id = "ok"
        inactive = _fake_executor(
            price=Decimal("0.4290"), amount=Decimal("46"), is_active=False
        )
        inactive.id = "inactive"
        wrong_cfg = _fake_executor(
            price=Decimal("0.4280"),
            amount=Decimal("46"),
            use_order_executor_config=False,
        )
        wrong_cfg.id = "wrong-cfg"
        none_price = _fake_executor(price=None, amount=Decimal("46"))
        none_price.id = "none-price"
        controller = self._make_controller(
            executors=[ok, inactive, wrong_cfg, none_price]
        )
        controller._log_active_executors_snapshot()
        log_msg = self.log_mock.info.call_args[0][0]
        self.assertIn("ok", log_msg)
        self.assertNotIn("inactive", log_msg)
        self.assertNotIn("wrong-cfg", log_msg)
        self.assertNotIn("none-price", log_msg)


class TestBBOPegSellLogActionsEmitted(unittest.TestCase):
    """Direct unit tests for _log_actions_emitted.

    This log lets us verify that the controller IS attempting to cancel each
    order it knows about. Pair with _log_active_executors_snapshot for orphan
    diagnosis: if the exchange shows an order after a Stop was logged for
    its id, the cancel failed at the framework/exchange layer (not us).

    Unlike the snapshot, this is NOT rate-limited — every non-empty action
    set gets a log line so we can audit each tick's intent. Empty action
    sets are suppressed to avoid noise on no-op ticks.
    """

    def _make_controller(self) -> BBOPegSellController:
        controller, log_mock = _make_controller_for_walker()
        self.log_mock = log_mock
        return controller

    def _make_create_action(
        self, *, price: Decimal, executor_id: str = "new"
    ) -> CreateExecutorAction:
        return CreateExecutorAction(
            controller_id="ctrl",
            executor_config=OrderExecutorConfig(
                id=executor_id,
                timestamp=1.0,
                connector_name="htx",
                trading_pair="XNO-USDT",
                side=TradeType.SELL,
                amount=Decimal("46"),
                price=price,
                execution_strategy=ExecutionStrategy.LIMIT_MAKER,
            ),
        )

    # --- Empty-vs-non-empty gating ---

    def test_does_not_log_when_actions_list_is_empty(self):
        # No-op tick → no log line. Avoids spamming the log on every idle tick.
        controller = self._make_controller()
        controller._log_actions_emitted([])
        self.log_mock.info.assert_not_called()

    # --- Stop content ---

    def test_logs_stop_executor_ids(self):
        # Each Stop's executor_id must appear in the log so we can grep for
        # which orders the controller TRIED to cancel.
        controller = self._make_controller()
        stop1 = StopExecutorAction(controller_id="ctrl", executor_id="exec-1")
        stop2 = StopExecutorAction(controller_id="ctrl", executor_id="exec-2")
        controller._log_actions_emitted([stop1, stop2])
        self.log_mock.info.assert_called_once()
        log_msg = self.log_mock.info.call_args[0][0]
        self.assertIn("stops=", log_msg)
        self.assertIn("exec-1", log_msg)
        self.assertIn("exec-2", log_msg)

    # --- Create content ---

    def test_logs_create_prices(self):
        # Each Create's intended price must appear in the log so we can
        # see what target the controller asked the framework to place.
        controller = self._make_controller()
        create = self._make_create_action(price=Decimal("0.4296"))
        controller._log_actions_emitted([create])
        self.log_mock.info.assert_called_once()
        log_msg = self.log_mock.info.call_args[0][0]
        self.assertIn("creates=", log_msg)
        self.assertIn("0.4296", log_msg)

    # --- Mixed action set ---

    def test_logs_stops_and_creates_in_single_line(self):
        # Typical action mix: cancel-and-replace. Both halves must land in
        # the same log line so the diagnostic is one greppable tick event.
        controller = self._make_controller()
        stop = StopExecutorAction(controller_id="ctrl", executor_id="old-id")
        create = self._make_create_action(price=Decimal("0.4297"))
        controller._log_actions_emitted([stop, create])
        self.log_mock.info.assert_called_once()
        log_msg = self.log_mock.info.call_args[0][0]
        self.assertIn("old-id", log_msg)
        self.assertIn("0.4297", log_msg)


class TestBBOPegSellBuildCreateAction(unittest.TestCase):
    """Direct unit tests for _build_create_action in isolation.

    Verifies the LIMIT_MAKER buy action contract: correct order shape,
    safety invariants (BUY side, LIMIT_MAKER strategy), amount quantization,
    and the zero/negative-amount short-circuit.
    """

    def _make_controller(
        self,
        *,
        quantize_amount_side_effect=None,
        timestamp: float = 1700000000.0,
    ) -> BBOPegSellController:
        """Builds a controller with quantize_order_amount and time() mocked.
        Stashes the mocked MDP on self.market_data_provider_mock so tests
        can assert on call args directly.
        """
        config = BBOPegSellConfig(
            id="test-controller-id",
            controller_name="bbo_peg_sell",
            connector_name="htx",
            trading_pair="XNO-USDT",
            total_amount_quote=Decimal("20"),
            update_interval=0.5,
        )
        market_data_provider = MagicMock(spec=MarketDataProvider)
        market_data_provider.quantize_order_amount.side_effect = (
            quantize_amount_side_effect or (lambda _c, _p, amount: amount)
        )
        market_data_provider.time.return_value = timestamp  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
        # Default to ample base balance so tests not specifically about the
        # pre-flight balance check don't have to mock it. Balance-specific
        # tests override this on the returned controller's MDP.
        market_data_provider.get_available_balance.return_value = Decimal("1000000")  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
        self.market_data_provider_mock = market_data_provider
        return BBOPegSellController(
            config=config,
            market_data_provider=market_data_provider,
            actions_queue=AsyncMock(spec=asyncio.Queue),
        )

    # --- Happy path: action shape ---

    def test_returns_create_action_when_amount_is_positive(self):
        controller = self._make_controller()
        result = controller._build_create_action(target_price=Decimal("0.4381"))
        self.assertIsInstance(result, CreateExecutorAction)

    def test_action_uses_target_price_as_order_price(self):
        controller = self._make_controller()
        result = controller._build_create_action(target_price=Decimal("0.4381"))
        assert result is not None
        config = result.executor_config
        assert isinstance(config, OrderExecutorConfig)
        self.assertEqual(config.price, Decimal("0.4381"))

    def test_action_uses_connector_and_pair_from_config(self):
        controller = self._make_controller()
        result = controller._build_create_action(target_price=Decimal("0.4381"))
        assert result is not None
        config = result.executor_config
        assert isinstance(config, OrderExecutorConfig)
        self.assertEqual(config.connector_name, "htx")
        self.assertEqual(config.trading_pair, "XNO-USDT")

    def test_action_controller_id_matches_config_id(self):
        controller = self._make_controller()
        result = controller._build_create_action(target_price=Decimal("0.4381"))
        assert result is not None
        self.assertEqual(result.controller_id, "test-controller-id")

    def test_action_timestamp_comes_from_market_data_provider_time(self):
        controller = self._make_controller(timestamp=1700000000.0)
        result = controller._build_create_action(target_price=Decimal("0.4381"))
        assert result is not None
        self.assertEqual(result.executor_config.timestamp, 1700000000.0)

    # --- Safety invariants (critical) ---

    def test_action_uses_buy_side(self):
        # CRITICAL: this is a buy-only controller. Must NEVER be SELL.
        controller = self._make_controller()
        result = controller._build_create_action(target_price=Decimal("0.4381"))
        assert result is not None
        config = result.executor_config
        assert isinstance(config, OrderExecutorConfig)
        self.assertEqual(config.side, TradeType.SELL)

    def test_action_uses_limit_maker_strategy(self):
        # CRITICAL: LIMIT_MAKER makes the exchange reject orders that would
        # cross the spread. Any other strategy (LIMIT, MARKET, TAKER) would
        # defeat the anti-crossing guard. Must NEVER be anything else.
        controller = self._make_controller()
        result = controller._build_create_action(target_price=Decimal("0.4381"))
        assert result is not None
        config = result.executor_config
        assert isinstance(config, OrderExecutorConfig)
        self.assertEqual(config.execution_strategy, ExecutionStrategy.LIMIT_MAKER)

    # --- Amount math / quantization plumbing ---

    def test_quantize_called_with_quote_divided_by_price(self):
        # total_amount_quote=20, target_price=0.4381 → 20 / 0.4381.
        controller = self._make_controller()
        controller._build_create_action(target_price=Decimal("0.4381"))
        self.market_data_provider_mock.quantize_order_amount.assert_called_once_with(
            "htx", "XNO-USDT", Decimal("20") / Decimal("0.4381")
        )

    def test_amount_is_quantized_value_not_raw_division(self):
        # If quantize snaps to a different size (e.g., exchange rounds down),
        # the action must use that snapped amount, not the raw division.
        controller = self._make_controller(
            quantize_amount_side_effect=lambda _c, _p, _amount: Decimal("45")
        )
        result = controller._build_create_action(target_price=Decimal("0.4381"))
        assert result is not None
        config = result.executor_config
        assert isinstance(config, OrderExecutorConfig)
        self.assertEqual(config.amount, Decimal("45"))

    # --- Negative / edge cases ---

    def test_returns_none_when_quantized_amount_is_zero(self):
        # Guard: if exchange snaps amount down to 0 (we're below min order
        # size), no action should be built. A zero-amount action would be
        # exchange-rejected.
        controller = self._make_controller(
            quantize_amount_side_effect=lambda _c, _p, _amount: Decimal("0")
        )
        result = controller._build_create_action(target_price=Decimal("0.4381"))
        self.assertIsNone(result)

    def test_returns_none_when_quantized_amount_is_negative(self):
        # Defensive: quantize should never return negative, but the `<= 0`
        # guard catches this too.
        controller = self._make_controller(
            quantize_amount_side_effect=lambda _c, _p, _amount: Decimal("-1")
        )
        result = controller._build_create_action(target_price=Decimal("0.4381"))
        self.assertIsNone(result)

    # --- Pre-flight balance check ---

    def test_returns_none_when_base_balance_insufficient(self):
        # Production scenario: the account has no XNO (or less than the
        # quantized amount). The framework's OrderExecutor would otherwise
        # log "Not enough budget to open position" on every tick. The
        # controller-level guard prevents that by skipping the Create
        # entirely.
        controller = self._make_controller()
        log_mock = MagicMock()
        setattr(controller, "logger", MagicMock(return_value=log_mock))
        # target=0.4381, quote=20 -> required ≈ 45.65 XNO. Account has 10.
        controller.market_data_provider.get_available_balance.return_value = Decimal(  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
            "10"
        )
        result = controller._build_create_action(target_price=Decimal("0.4381"))
        self.assertIsNone(result)

    def test_returns_create_action_when_base_balance_exactly_sufficient(self):
        # Boundary: balance == required must NOT block (strict `<`, not `<=`).
        # If a refactor changes the comparator we silently lose every order
        # at the wire-edge.
        controller = self._make_controller(
            quantize_amount_side_effect=lambda _c, _p, _amount: Decimal("46"),
        )
        controller.market_data_provider.get_available_balance.return_value = Decimal(  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
            "46"
        )
        result = controller._build_create_action(target_price=Decimal("0.4381"))
        self.assertIsInstance(result, CreateExecutorAction)

    def test_balance_check_reads_base_asset_from_trading_pair(self):
        # The base asset is parsed by splitting on "-". XNO-USDT → "XNO".
        # Pin the parse so a refactor to e.g. "/" doesn't silently read the
        # wrong asset's balance.
        controller = self._make_controller()
        controller._build_create_action(target_price=Decimal("0.4381"))
        controller.market_data_provider.get_available_balance.assert_called_with(  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
            "htx", "XNO"
        )

    def test_insufficient_balance_log_fires_once_per_episode(self):
        # Rate-limiting: an empty account on every tick must NOT spam the
        # log. First insufficient → 1 warning. Second insufficient → no new
        # log. Recovery resets the latch (covered separately).
        controller = self._make_controller()
        log_mock = MagicMock()
        setattr(controller, "logger", MagicMock(return_value=log_mock))
        controller.market_data_provider.get_available_balance.return_value = Decimal(  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
            "0"
        )
        controller._build_create_action(target_price=Decimal("0.4381"))
        controller._build_create_action(target_price=Decimal("0.4381"))
        controller._build_create_action(target_price=Decimal("0.4381"))
        self.assertEqual(log_mock.warning.call_count, 1)
        warning_msg = log_mock.warning.call_args[0][0]
        self.assertIn("insufficient", warning_msg.lower())
        self.assertIn("XNO", warning_msg)

    def test_recovery_emits_info_log_and_resets_latch(self):
        # Transition: insufficient → sufficient must log a single recovery
        # info line and re-arm the warning so a subsequent insufficient
        # episode logs again. Mirrors the gate state-transition pattern.
        controller = self._make_controller()
        log_mock = MagicMock()
        setattr(controller, "logger", MagicMock(return_value=log_mock))
        controller.market_data_provider.get_available_balance.return_value = Decimal(  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
            "0"
        )
        controller._build_create_action(
            target_price=Decimal("0.4381")
        )  # blocked → warn
        controller.market_data_provider.get_available_balance.return_value = Decimal(  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
            "100"
        )
        controller._build_create_action(
            target_price=Decimal("0.4381")
        )  # recovered → info
        self.assertEqual(log_mock.warning.call_count, 1)
        self.assertEqual(log_mock.info.call_count, 1)
        info_msg = log_mock.info.call_args[0][0]
        self.assertIn("recovered", info_msg.lower())
        # Re-arm check: drop balance again → another warning fires.
        controller.market_data_provider.get_available_balance.return_value = Decimal(  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
            "0"
        )
        controller._build_create_action(target_price=Decimal("0.4381"))
        self.assertEqual(log_mock.warning.call_count, 2)

    def test_no_log_on_cold_start_with_sufficient_balance(self):
        # Cold start has _balance_insufficient_logged=False. A sufficient
        # balance on the very first call must NOT emit any "recovered" info
        # log — that would only make sense after a prior insufficient state.
        controller = self._make_controller()
        log_mock = MagicMock()
        setattr(controller, "logger", MagicMock(return_value=log_mock))
        controller.market_data_provider.get_available_balance.return_value = Decimal(  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
            "1000"
        )
        controller._build_create_action(target_price=Decimal("0.4381"))
        log_mock.warning.assert_not_called()
        log_mock.info.assert_not_called()


class TestBBOPegSellBuildStopActions(unittest.TestCase):
    """Direct unit tests for _build_stop_actions in isolation.

    Pure list-comprehension transformer: every executor in → one
    StopExecutorAction out, no filtering. Pinning the contract so future
    refactors don't accidentally add filtering (which would silently leave
    stale orders alive on the exchange).
    """

    def _make_controller(self) -> BBOPegSellController:
        config = BBOPegSellConfig(
            id="test-controller-id",
            controller_name="bbo_peg_sell",
            connector_name="htx",
            trading_pair="XNO-USDT",
            total_amount_quote=Decimal("20"),
            update_interval=0.5,
        )
        return BBOPegSellController(
            config=config,
            market_data_provider=MagicMock(spec=MarketDataProvider),
            actions_queue=AsyncMock(spec=asyncio.Queue),
        )

    def _fake_executor_with_id(
        self, executor_id: str, is_active: bool = True
    ) -> ExecutorInfo:
        executor = MagicMock()
        executor.id = executor_id
        executor.is_active = is_active
        return cast(ExecutorInfo, executor)

    # --- Happy path / shape ---

    def test_returns_one_action_per_executor(self):
        controller = self._make_controller()
        executors = [
            self._fake_executor_with_id("exec-1"),
            self._fake_executor_with_id("exec-2"),
            self._fake_executor_with_id("exec-3"),
        ]
        result = controller._build_stop_actions(executors)
        self.assertEqual(len(result), 3)

    def test_each_action_uses_controller_config_id(self):
        controller = self._make_controller()
        executors = [self._fake_executor_with_id("exec-1")]
        result = controller._build_stop_actions(executors)
        action = result[0]
        assert isinstance(action, StopExecutorAction)
        self.assertEqual(action.controller_id, "test-controller-id")

    def test_each_action_uses_executor_id_from_input(self):
        controller = self._make_controller()
        executors = [
            self._fake_executor_with_id("alpha"),
            self._fake_executor_with_id("beta"),
        ]
        result = controller._build_stop_actions(executors)
        assert isinstance(result[0], StopExecutorAction)
        assert isinstance(result[1], StopExecutorAction)
        self.assertEqual(result[0].executor_id, "alpha")
        self.assertEqual(result[1].executor_id, "beta")

    def test_preserves_input_order(self):
        # The order of stop actions should match the order of input executors.
        # Reordering could cause subtle issues if the framework processes them
        # in a specific sequence.
        controller = self._make_controller()
        executors = [self._fake_executor_with_id(f"exec-{i}") for i in range(5)]
        result = controller._build_stop_actions(executors)
        for i, action in enumerate(result):
            assert isinstance(action, StopExecutorAction)
            self.assertEqual(action.executor_id, f"exec-{i}")

    # --- Safety: every action is a Stop ---

    def test_all_actions_are_stop_executor_actions(self):
        # CRITICAL: must NEVER produce a CreateExecutorAction or any other
        # type. If a refactor accidentally swapped action classes, this
        # catches it before the framework executes the wrong intent.
        controller = self._make_controller()
        executors = [
            self._fake_executor_with_id("exec-1"),
            self._fake_executor_with_id("exec-2"),
        ]
        result = controller._build_stop_actions(executors)
        for action in result:
            self.assertIsInstance(action, StopExecutorAction)

    # --- Negative / edge cases ---

    def test_returns_empty_list_when_input_is_empty(self):
        # Edge: no executors to stop → no actions. Empty list, not None.
        controller = self._make_controller()
        result = controller._build_stop_actions([])
        self.assertEqual(result, [])

    def test_builds_stop_for_every_executor_regardless_of_activity(self):
        # CRITICAL CONTRACT: this function does NOT filter by is_active.
        # The caller (_categorize_active_orders) is responsible for that.
        # If a refactor adds is_active filtering here, "stale-but-inactive"
        # executors would slip past stop emission and could linger on the
        # exchange. Pin the no-filter behavior with a mix of states.
        controller = self._make_controller()
        executors = [
            self._fake_executor_with_id("active-1", is_active=True),
            self._fake_executor_with_id("inactive-2", is_active=False),
            self._fake_executor_with_id("active-3", is_active=True),
        ]
        result = controller._build_stop_actions(executors)
        self.assertEqual(len(result), 3)
        executor_ids = [
            a.executor_id for a in result if isinstance(a, StopExecutorAction)
        ]
        self.assertIn("inactive-2", executor_ids)


class TestBBOPegSellCategorizeActiveOrders(unittest.TestCase):
    """Direct unit tests for _categorize_active_orders in isolation.

    Splits executors_info into (stale, in_tolerance) by exact price match
    against target_price. Pins the filter rules, strict price equality,
    and the negative invariant that filtered executors appear in NEITHER
    list (not silently dropped into stale).
    """

    # --- Positive: basic categorization ---

    def test_empty_executors_info_returns_two_empty_lists(self):
        # Edge: no executors at all → ([], []).
        controller, _ = _make_controller_for_walker(executors=[])
        stale, in_tolerance = controller._categorize_active_orders(Decimal("0.4381"))
        self.assertEqual(stale, [])
        self.assertEqual(in_tolerance, [])

    def test_executor_at_target_price_categorized_as_in_tolerance(self):
        executor = _fake_executor(price=Decimal("0.4381"), amount=Decimal("50"))
        controller, _ = _make_controller_for_walker(executors=[executor])
        stale, in_tolerance = controller._categorize_active_orders(Decimal("0.4381"))
        self.assertEqual(stale, [])
        self.assertEqual(in_tolerance, [executor])

    def test_executor_at_different_price_categorized_as_stale(self):
        executor = _fake_executor(price=Decimal("0.4380"), amount=Decimal("50"))
        controller, _ = _make_controller_for_walker(executors=[executor])
        stale, in_tolerance = controller._categorize_active_orders(Decimal("0.4381"))
        self.assertEqual(stale, [executor])
        self.assertEqual(in_tolerance, [])

    def test_multiple_executors_split_correctly_into_both_lists(self):
        # Two at target → in_tolerance; two off-target → stale.
        at_target_1 = _fake_executor(price=Decimal("0.4381"), amount=Decimal("10"))
        at_target_2 = _fake_executor(price=Decimal("0.4381"), amount=Decimal("20"))
        stale_1 = _fake_executor(price=Decimal("0.4379"), amount=Decimal("30"))
        stale_2 = _fake_executor(price=Decimal("0.4378"), amount=Decimal("40"))
        controller, _ = _make_controller_for_walker(
            executors=[at_target_1, stale_1, at_target_2, stale_2]
        )
        stale, in_tolerance = controller._categorize_active_orders(Decimal("0.4381"))
        self.assertEqual(set(in_tolerance), {at_target_1, at_target_2})
        self.assertEqual(set(stale), {stale_1, stale_2})

    # --- Filter rules ---

    def test_inactive_executor_excluded(self):
        # Inactive at target_price — must NOT appear in either list.
        executor = _fake_executor(
            price=Decimal("0.4381"),
            amount=Decimal("50"),
            is_active=False,
        )
        controller, _ = _make_controller_for_walker(executors=[executor])
        stale, in_tolerance = controller._categorize_active_orders(Decimal("0.4381"))
        self.assertEqual(stale, [])
        self.assertEqual(in_tolerance, [])

    def test_non_order_executor_config_excluded(self):
        # Wrong config type — must NOT appear in either list.
        executor = _fake_executor(
            price=Decimal("0.4381"),
            amount=Decimal("50"),
            use_order_executor_config=False,
        )
        controller, _ = _make_controller_for_walker(executors=[executor])
        stale, in_tolerance = controller._categorize_active_orders(Decimal("0.4381"))
        self.assertEqual(stale, [])
        self.assertEqual(in_tolerance, [])

    def test_executor_with_none_price_excluded(self):
        # OrderExecutorConfig but price is None — must NOT appear in either.
        executor = _fake_executor(price=None, amount=Decimal("50"))
        controller, _ = _make_controller_for_walker(executors=[executor])
        stale, in_tolerance = controller._categorize_active_orders(Decimal("0.4381"))
        self.assertEqual(stale, [])
        self.assertEqual(in_tolerance, [])

    # --- Critical: strict price equality ---

    def test_categorization_uses_strict_price_equality(self):
        # CRITICAL: any tiny price difference → stale (no tolerance window).
        # If someone changes the != comparison to an "approximately equal"
        # check, an executor that's 1 tick off (or even 0.00000001 off) would
        # be classified as in_tolerance and never re-quoted. Pin the strict
        # behavior with a near-but-not-equal price.
        executor = _fake_executor(
            price=Decimal("0.43810001"),  # off by 0.00000001
            amount=Decimal("50"),
        )
        controller, _ = _make_controller_for_walker(executors=[executor])
        stale, in_tolerance = controller._categorize_active_orders(Decimal("0.4381"))
        self.assertEqual(stale, [executor])
        self.assertEqual(in_tolerance, [])

    # --- Negative / edge cases ---

    def test_filtered_executor_appears_in_neither_list(self):
        # CRITICAL NEGATIVE: ALL three filtered types (inactive, wrong-config,
        # none-price) must be excluded from BOTH lists. A subtle refactor
        # could accidentally drop them into stale (treating them as
        # "to be cancelled") — that would corrupt downstream stop emission.
        inactive = _fake_executor(
            price=Decimal("0.4381"),
            amount=Decimal("50"),
            is_active=False,
        )
        wrong_config = _fake_executor(
            price=Decimal("0.4381"),
            amount=Decimal("50"),
            use_order_executor_config=False,
        )
        none_price = _fake_executor(price=None, amount=Decimal("50"))
        controller, _ = _make_controller_for_walker(
            executors=[inactive, wrong_config, none_price]
        )
        stale, in_tolerance = controller._categorize_active_orders(Decimal("0.4381"))
        self.assertNotIn(inactive, stale)
        self.assertNotIn(inactive, in_tolerance)
        self.assertNotIn(wrong_config, stale)
        self.assertNotIn(wrong_config, in_tolerance)
        self.assertNotIn(none_price, stale)
        self.assertNotIn(none_price, in_tolerance)

    def test_all_executors_filtered_returns_two_empty_lists(self):
        # Edge: every executor filtered for a different reason → ([], []).
        controller, _ = _make_controller_for_walker(
            executors=[
                _fake_executor(
                    price=Decimal("0.4381"),
                    amount=Decimal("50"),
                    is_active=False,
                ),
                _fake_executor(
                    price=Decimal("0.4381"),
                    amount=Decimal("50"),
                    use_order_executor_config=False,
                ),
                _fake_executor(price=None, amount=Decimal("50")),
            ]
        )
        stale, in_tolerance = controller._categorize_active_orders(Decimal("0.4381"))
        self.assertEqual(stale, [])
        self.assertEqual(in_tolerance, [])


class TestBBOPegSellUpdateFillLatch(unittest.TestCase):
    """Direct unit tests for _update_fill_latch in isolation.

    Pins the one-shot fill detection contract: latch flips to True on any
    executor reporting executed_amount_base > 0, and NEVER resets within
    a process lifetime. The latch gates whether the controller emits
    more buy orders, so the never-reset guarantee is safety-critical.
    """

    def _make_controller(self) -> BBOPegSellController:
        config = BBOPegSellConfig(
            id="test",
            controller_name="bbo_peg_sell",
            connector_name="htx",
            trading_pair="XNO-USDT",
            total_amount_quote=Decimal("20"),
            update_interval=0.5,
        )
        return BBOPegSellController(
            config=config,
            market_data_provider=MagicMock(spec=MarketDataProvider),
            actions_queue=AsyncMock(spec=asyncio.Queue),
        )

    def _fake_executor_with_fill(
        self,
        *,
        executed_amount_base=None,
        custom_info_is_none: bool = False,
        missing_key: bool = False,
        is_active: bool = True,
    ) -> ExecutorInfo:
        """Builds a fake executor for fill-latch tests.

        - custom_info_is_none=True → executor.custom_info = None
        - missing_key=True → custom_info = {} (no executed_amount_base key)
        - otherwise → custom_info = {"executed_amount_base": executed_amount_base}
        """
        executor = MagicMock()
        executor.is_active = is_active
        if custom_info_is_none:
            executor.custom_info = None
        elif missing_key:
            executor.custom_info = {}
        else:
            executor.custom_info = {"executed_amount_base": executed_amount_base}
        return cast(ExecutorInfo, executor)

    # --- Positive: latch behavior ---

    def test_latch_stays_false_when_no_executors_have_fills(self):
        controller = self._make_controller()
        setattr(
            controller,
            "executors_info",
            [
                self._fake_executor_with_fill(executed_amount_base=Decimal("0")),
            ],
        )
        controller._update_fill_latch()
        self.assertFalse(controller._has_filled)

    def test_latch_flips_to_true_on_executor_with_positive_fill(self):
        controller = self._make_controller()
        setattr(
            controller,
            "executors_info",
            [
                self._fake_executor_with_fill(executed_amount_base=Decimal("5")),
            ],
        )
        controller._update_fill_latch()
        self.assertTrue(controller._has_filled)

    # --- Filter / skip cases ---

    def test_executor_with_none_custom_info_is_skipped(self):
        # Defensive: if framework hasn't populated custom_info, skip cleanly
        # rather than crash trying to .get() on None.
        controller = self._make_controller()
        setattr(
            controller,
            "executors_info",
            [
                self._fake_executor_with_fill(custom_info_is_none=True),
            ],
        )
        controller._update_fill_latch()
        self.assertFalse(controller._has_filled)

    def test_executor_with_missing_executed_amount_base_key_is_skipped(self):
        # custom_info exists but lacks the key → dict.get returns None → skip.
        controller = self._make_controller()
        setattr(
            controller,
            "executors_info",
            [
                self._fake_executor_with_fill(missing_key=True),
            ],
        )
        controller._update_fill_latch()
        self.assertFalse(controller._has_filled)

    def test_executor_with_zero_executed_amount_does_not_flip_latch(self):
        # Strict > 0 guard: exactly zero must not trip the latch.
        controller = self._make_controller()
        setattr(
            controller,
            "executors_info",
            [
                self._fake_executor_with_fill(executed_amount_base=Decimal("0")),
            ],
        )
        controller._update_fill_latch()
        self.assertFalse(controller._has_filled)

    # --- Multiple executors ---

    def test_latch_flips_when_one_of_many_executors_has_fill(self):
        # First two have no fill; third does. The loop must continue past the
        # first negatives and trip the latch on the matching executor.
        controller = self._make_controller()
        setattr(
            controller,
            "executors_info",
            [
                self._fake_executor_with_fill(executed_amount_base=Decimal("0")),
                self._fake_executor_with_fill(custom_info_is_none=True),
                self._fake_executor_with_fill(executed_amount_base=Decimal("10")),
            ],
        )
        controller._update_fill_latch()
        self.assertTrue(controller._has_filled)

    # --- Critical: one-shot / never-reset guarantee ---

    def test_latch_stays_true_when_already_set_and_executors_have_no_fills(self):
        # CRITICAL: the latch must NEVER reset once set. Even if every
        # executor currently shows zero fills (e.g., the filled one was
        # cleaned up between ticks), the latch must remember the past fill.
        # Otherwise the one-shot guarantee breaks and the bot could place
        # another order after a fill.
        controller = self._make_controller()
        controller._has_filled = True
        setattr(
            controller,
            "executors_info",
            [
                self._fake_executor_with_fill(executed_amount_base=Decimal("0")),
                self._fake_executor_with_fill(custom_info_is_none=True),
            ],
        )
        controller._update_fill_latch()
        self.assertTrue(controller._has_filled)

    # --- Defensive / negative cases ---

    def test_negative_executed_amount_does_not_flip_latch(self):
        # Defensive: executed_amount_base should never be negative, but the
        # strict > 0 guard catches this case too. If someone replaced > 0
        # with != 0 (treating any non-zero as a fill), this would catch it.
        controller = self._make_controller()
        setattr(
            controller,
            "executors_info",
            [
                self._fake_executor_with_fill(executed_amount_base=Decimal("-1")),
            ],
        )
        controller._update_fill_latch()
        self.assertFalse(controller._has_filled)

    def test_string_executed_amount_handled_via_decimal_conversion(self):
        # Custom info may come from JSON deserialization → values are strings.
        # The Decimal(str(executed)) conversion must handle string input.
        # If someone "simplifies" this to Decimal(executed) directly, a string
        # input might fail (in some scenarios) or behave unexpectedly.
        controller = self._make_controller()
        setattr(
            controller,
            "executors_info",
            [
                self._fake_executor_with_fill(executed_amount_base="5.0"),
            ],
        )
        controller._update_fill_latch()
        self.assertTrue(controller._has_filled)

    # --- Contract pin: no is_active filter ---

    def test_inactive_executor_with_fill_still_flips_latch(self):
        # CONTRACT: unlike _categorize_active_orders, this function does NOT
        # filter by is_active. An inactive executor that recorded a fill
        # before being finalized still represents "we had a fill" — the
        # latch must trip. If a refactor adds is_active filtering here,
        # fills on cleanly-closed executors would be missed and the bot
        # would re-quote after a real fill.
        controller = self._make_controller()
        setattr(
            controller,
            "executors_info",
            [
                self._fake_executor_with_fill(
                    executed_amount_base=Decimal("5"),
                    is_active=False,
                ),
            ],
        )
        controller._update_fill_latch()
        self.assertTrue(controller._has_filled)


class TestBBOPegSellDetermineExecutorActions(unittest.TestCase):
    """End-to-end orchestration tests for determine_executor_actions.

    Composes all the helpers tested individually elsewhere — focuses on
    flow, decision branches, and the fill-latch gating that enforces the
    one-shot guarantee. Each test sets up controller state and asserts
    on the action list shape.
    """

    def _make_controller(
        self,
        *,
        target_price: Optional[Decimal] = Decimal("0.4381"),
        executors: Optional[List[ExecutorInfo]] = None,
        has_filled: bool = False,
        quantize_amount_side_effect=None,
    ) -> BBOPegSellController:
        """Builds a controller with processed_data, executors_info, and
        _has_filled pre-populated. quantize_order_amount defaults to identity.
        """
        config = BBOPegSellConfig(
            id="test-controller-id",
            controller_name="bbo_peg_sell",
            connector_name="htx",
            trading_pair="XNO-USDT",
            total_amount_quote=Decimal("20"),
            update_interval=0.5,
        )
        market_data_provider = MagicMock(spec=MarketDataProvider)
        market_data_provider.quantize_order_amount.side_effect = (
            quantize_amount_side_effect or (lambda _c, _p, amount: amount)
        )
        market_data_provider.time.return_value = 1700000000.0  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
        # Default to ample base balance — see TestBBOPegSellBuildCreateAction
        # for the rationale and tests that override this.
        market_data_provider.get_available_balance.return_value = Decimal("1000000")  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
        controller = BBOPegSellController(
            config=config,
            market_data_provider=market_data_provider,
            actions_queue=AsyncMock(spec=asyncio.Queue),
        )
        controller.processed_data = {"target_price": target_price}
        setattr(controller, "executors_info", executors or [])
        controller._has_filled = has_filled
        return controller

    # --- Happy path orchestration ---

    def test_cold_start_emits_only_create_action(self):
        # No executors yet, target price set, no fill → emit one Create.
        controller = self._make_controller(target_price=Decimal("0.4381"))
        actions = controller.determine_executor_actions()
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], CreateExecutorAction)

    def test_steady_state_emits_no_actions_when_already_at_target(self):
        # Active order at target → in_tolerance → no Stop, no Create.
        at_target = _fake_executor(price=Decimal("0.4381"), amount=Decimal("45"))
        controller = self._make_controller(
            target_price=Decimal("0.4381"),
            executors=[at_target],
        )
        actions = controller.determine_executor_actions()
        self.assertEqual(actions, [])

    def test_price_drift_emits_stop_then_create_in_order(self):
        # Stale order, no in_tolerance, no fill → Stop first, then Create.
        # Order matters: cancel old before placing new to avoid double-exposure.
        stale = _fake_executor(price=Decimal("0.4380"), amount=Decimal("45"))
        stale.id = "stale-1"
        controller = self._make_controller(
            target_price=Decimal("0.4381"),
            executors=[stale],
        )
        actions = controller.determine_executor_actions()
        self.assertEqual(len(actions), 2)
        self.assertIsInstance(actions[0], StopExecutorAction)
        self.assertIsInstance(actions[1], CreateExecutorAction)

    def test_does_not_emit_create_when_in_tolerance_exists_alongside_stale(self):
        # Mixed: one stale (different price) + one in_tolerance (at target).
        # Stale gets stopped; no new Create because in_tolerance already covers.
        stale = _fake_executor(price=Decimal("0.4380"), amount=Decimal("20"))
        stale.id = "stale-1"
        at_target = _fake_executor(price=Decimal("0.4381"), amount=Decimal("45"))
        at_target.id = "at-target-1"
        controller = self._make_controller(
            target_price=Decimal("0.4381"),
            executors=[stale, at_target],
        )
        actions = controller.determine_executor_actions()
        stops = [a for a in actions if isinstance(a, StopExecutorAction)]
        creates = [a for a in actions if isinstance(a, CreateExecutorAction)]
        self.assertEqual(len(stops), 1)
        self.assertEqual(creates, [])

    # --- Early return / no-target ---

    def test_returns_empty_list_when_target_price_is_none_and_no_active_executors(self):
        # No target price AND no active executors → nothing to cancel,
        # nothing to create. Result is empty.
        controller = self._make_controller(target_price=None)
        actions = controller.determine_executor_actions()
        self.assertEqual(actions, [])

    def test_cancels_active_orders_when_target_price_is_none(self):
        # CRITICAL: when target_price is None (no external bid, would cross
        # ask, or anti-spoof gate fired), any active order is unsafe to
        # leave resting — cancel it. The earlier behavior of returning []
        # left orders exposed during exactly the suspicious windows the
        # gate exists to guard.
        active = _fake_executor(price=Decimal("0.4380"), amount=Decimal("45"))
        active.id = "active-1"
        controller = self._make_controller(target_price=None, executors=[active])
        actions = controller.determine_executor_actions()
        self.assertEqual(len(actions), 1)
        action = actions[0]
        assert isinstance(action, StopExecutorAction)
        self.assertEqual(action.executor_id, "active-1")

    def test_does_not_emit_stops_for_inactive_executors_when_target_price_is_none(self):
        # NEGATIVE: only ACTIVE executors get cancelled in the None branch.
        # An inactive executor (already terminal) must not generate a stop
        # action — that would be a redundant or invalid exchange call.
        inactive = _fake_executor(
            price=Decimal("0.4380"), amount=Decimal("45"), is_active=False
        )
        controller = self._make_controller(target_price=None, executors=[inactive])
        actions = controller.determine_executor_actions()
        self.assertEqual(actions, [])

    # --- Fill latch gating (one-shot guarantee) ---

    def test_no_create_after_fill_latched(self):
        # CRITICAL: _has_filled=True with no stale → return [] (no Create
        # even though target exists and no in_tolerance). The one-shot
        # guarantee gates here.
        controller = self._make_controller(
            target_price=Decimal("0.4381"),
            has_filled=True,
        )
        actions = controller.determine_executor_actions()
        self.assertEqual(actions, [])

    def test_emits_stops_for_stale_even_after_fill_latched(self):
        # _has_filled=True with stale → emit Stop only, no Create. We still
        # want to clean up stale orders on the exchange (don't leave garbage).
        stale = _fake_executor(price=Decimal("0.4380"), amount=Decimal("45"))
        stale.id = "stale-1"
        controller = self._make_controller(
            target_price=Decimal("0.4381"),
            has_filled=True,
            executors=[stale],
        )
        actions = controller.determine_executor_actions()
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], StopExecutorAction)

    def test_fill_in_executors_info_trips_latch_and_blocks_create(self):
        # CRITICAL ORDERING: _update_fill_latch runs FIRST in
        # determine_executor_actions. An executor with a fill must trip the
        # latch BEFORE the create-or-not decision is made — otherwise the
        # bot would emit one final order after a fill.
        # Use an inactive executor with a fill so categorize filters it out
        # (not stale, not in_tolerance) but latch still sees it.
        filled = MagicMock()
        filled.is_active = False
        filled.custom_info = {"executed_amount_base": "5"}
        filled.config = MagicMock(spec=OrderExecutorConfig)
        filled.config.price = Decimal("0.4381")
        controller = self._make_controller(
            target_price=Decimal("0.4381"),
            executors=[cast(ExecutorInfo, filled)],
        )
        self.assertFalse(controller._has_filled)
        actions = controller.determine_executor_actions()
        self.assertTrue(controller._has_filled)
        creates = [a for a in actions if isinstance(a, CreateExecutorAction)]
        self.assertEqual(creates, [])

    # --- Defensive / edge ---

    def test_does_not_emit_create_when_amount_quantizes_to_zero(self):
        # CRITICAL DEFENSIVE: if the exchange snaps our requested amount to
        # zero (below min order size, balance insufficient, etc.),
        # _build_create_action returns None → no Create appended. The bot
        # quietly does nothing this tick rather than emitting a degenerate
        # zero-amount action that would be exchange-rejected.
        controller = self._make_controller(
            target_price=Decimal("0.4381"),
            quantize_amount_side_effect=lambda _c, _p, _amount: Decimal("0"),
        )
        actions = controller.determine_executor_actions()
        self.assertEqual(actions, [])


class TestBBOPegSellDowngradeTrapScenario(IsolatedAsyncioWrapperTestCase):
    """End-to-end integration test for the SELL-side 'stuck at best_ask' trap.

    Composes controller stale-detection with NoClampOrderExecutor's
    pass-through for SELL LIMIT_MAKER orders where the intended price is
    BELOW current_best_ask. This is the healthy-spread case — the anti-spoof
    gate plays no role here.

    Mirror of the buy-side downgrade-trap scenario. Without NoClamp, stock
    OrderExecutor would silently UPGRADE our intended price to the current
    best_ask via max(config.price, current_best_ask), leaving us stuck at
    the top of book rather than one tick below it:

      - 92 XNO of external asks at 0.4541 + our 46 XNO at the same level
      - Healthy 5.7% spread, so the anti-spoof gate stays clear
      - Walker: external_best_ask = 0.4541 (excludes our own volume)
      - Controller target = 0.4540 (1 tick below external)
      - Controller emits Stop(our 0.4541 order) + Create(price=0.4540)
      - NoClampOrderExecutor.get_order_price() returns 0.4540 — the
        stock OrderExecutor would silently upgrade it to 0.4541 here.

    Test fails if EITHER half is reverted:
      - Controller's stale detection breaks: assertions 1-3 fail.
      - NoClamp's SELL LIMIT_MAKER override is reverted: assertion 4 fails
        (final exchange price climbs back to the trap value 0.4541).
    """

    @staticmethod
    def _make_strategy_for_executor() -> MagicMock:
        """Minimal mocked StrategyV2Base sufficient to instantiate an
        OrderExecutor subclass.
        """
        strategy = MagicMock(spec=StrategyV2Base)
        type(strategy).trading_pair = PropertyMock(return_value="XNO-USDT")
        connector = MagicMock(spec=ExchangePyBase)
        type(connector).trading_rules = PropertyMock(
            return_value={"XNO-USDT": TradingRule(trading_pair="XNO-USDT")}
        )
        strategy.connectors = {"htx": connector}
        return strategy

    async def test_stuck_state_controller_emits_create_no_clamp_preserves_intent(self):
        # Our 46 XNO is already at 0.4541 (placed in some prior tick that
        # the stock-OrderExecutor upgrade trap forced up to this level).
        active_executor = _fake_executor(price=Decimal("0.4541"), amount=Decimal("46"))
        active_executor.id = "upgraded-order"

        # Ask book mirror of the buy-side log 21:47:55:
        #   0.4541   138 XNO  (92 external + 46 ours)
        #   0.4542    46 XNO  (someone copying our size)
        #   0.4570   335 XNO  (deeper real asks)
        # Best bid at 0.4295 → spread 5.7% → gate stays clear.
        stuck_asks = [
            (Decimal("0.4541"), Decimal("138")),
            (Decimal("0.4542"), Decimal("46")),
            (Decimal("0.4570"), Decimal("335")),
        ]

        config = BBOPegSellConfig(
            id="test",
            controller_name="bbo_peg_sell",
            connector_name="htx",
            trading_pair="XNO-USDT",
            total_amount_quote=Decimal("20"),
            update_interval=0.5,
            # Gate enabled at production threshold (2%), but spread is 5.7%
            # so it stays clear. Isolates this test from anti-spoof behavior
            # — the trap lives entirely in the normal "post a quote" path.
            min_spread_pct=Decimal("0.02"),
        )
        market_data_provider = MagicMock(spec=MarketDataProvider)
        rules = MagicMock()
        rules.min_price_increment = Decimal("0.0001")
        market_data_provider.get_trading_rules.return_value = rules
        # PriceType.BestBid on sell side returns the opposite-side guard.
        market_data_provider.get_price_by_type.return_value = Decimal("0.4295")
        market_data_provider.get_order_book.return_value = _fake_order_book(stuck_asks)
        market_data_provider.quantize_order_price.side_effect = lambda _c, _p, price: (
            price
        )
        market_data_provider.quantize_order_amount.side_effect = lambda _c, _p, amount: (
            amount
        )
        market_data_provider.time.return_value = 1700000000.0  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
        market_data_provider.get_available_balance.return_value = Decimal("1000000")  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]

        controller = BBOPegSellController(
            config=config,
            market_data_provider=market_data_provider,
            actions_queue=AsyncMock(spec=asyncio.Queue),
        )
        setattr(controller, "executors_info", [active_executor])
        setattr(controller, "logger", MagicMock(return_value=MagicMock()))

        # Run a single controller tick.
        await controller.update_processed_data()
        actions = controller.determine_executor_actions()

        # ASSERTION 1 — controller computed the right target.
        # External best is 0.4541 (the 92-XNO non-ours), so target = 0.4540.
        # Fails if walker logic regresses (e.g., stops excluding own volume).
        self.assertEqual(controller.processed_data["target_price"], Decimal("0.4540"))

        # ASSERTION 2 — controller flagged our 0.4541 order as stale and
        # emitted a Stop for it. Fails if categorize logic regresses
        # (e.g., starts using tolerance comparison instead of strict equality).
        stops = [a for a in actions if isinstance(a, StopExecutorAction)]
        self.assertEqual(len(stops), 1)
        stop = stops[0]
        assert isinstance(stop, StopExecutorAction)
        self.assertEqual(stop.executor_id, "upgraded-order")

        # ASSERTION 3 — controller emitted a Create at the INTENDED price
        # (0.4540), proving it isn't internally pre-clamping or doing
        # anything weird. The controller's job ends here; the price 0.4540
        # is handed off to the framework.
        creates = [a for a in actions if isinstance(a, CreateExecutorAction)]
        self.assertEqual(len(creates), 1)
        create = creates[0]
        assert isinstance(create, CreateExecutorAction)
        executor_config = create.executor_config
        assert isinstance(executor_config, OrderExecutorConfig)
        self.assertEqual(executor_config.price, Decimal("0.4540"))

        # ASSERTION 4 — NoClampOrderExecutor preserves the controller's intent.
        # Wire the Create's config through NoClamp.get_order_price() — the
        # actual code path production runs. Final exchange price MUST equal
        # 0.4540 (intent), NOT 0.4541 (stock-OrderExecutor max-clamp = the
        # trap). Fails if NoClamp's SELL override is reverted to delegate
        # to super().
        no_clamp_executor = NoClampOrderExecutor(
            strategy=self._make_strategy_for_executor(),
            config=executor_config,
        )
        # Standard test idiom for mocking a @property: type checkers don't
        # recognize this as a valid setter because current_market_price is
        # read-only in the parent class, but the runtime assignment works.
        type(no_clamp_executor).current_market_price = PropertyMock(  # type: ignore[method-assign]  # pyright: ignore[reportAttributeAccessIssue]
            return_value=Decimal("0.4541")  # current_best_ask at exchange
        )
        self.assertEqual(no_clamp_executor.get_order_price(), Decimal("0.4540"))


class TestBBOPegSellCancelLagScenario(IsolatedAsyncioWrapperTestCase):
    """End-to-end test for the self-chase loop on the SELL side at short tick
    intervals.

    The bug: when the controller emits Stop+Create, the executor disappears
    from executors_info instantly but the exchange book WebSocket lags by
    1-2s. During the lag, the walker sees the just-cancelled order in the
    book + no matching active executor → treats it as an external ask →
    targets one tick BELOW → emits Stop+Create at the lower price → loop.

    Sell mirror of the buy-side scenario. The walker walks asks lowest-first,
    so the chase direction is DOWN (not up). Hypothetical sell trace:
        Tick 1: external=0.4366, my=0.4316 → emit create 0.4365
        Tick 2: external=0.4316, my=0.4365 → emit create 0.4315   ← ghost!
        Tick 3: external=0.4366, my=0.4315 → emit create 0.4365
        Tick 4: external=0.4315, my=0.4365 → emit create 0.4314   ← ghost!

    This test pins the FIX: with the just-cancelled order recorded in
    _pending_cancels, the walker subtracts it from the book and finds the
    real external ask above, avoiding the chase.

    Fails if cancel-debounce is reverted (e.g., _pending_cancels is empty
    or _compute_own_volume_by_price stops including it).
    """

    async def test_walker_does_not_chase_just_cancelled_ghost_order(self):
        # Simulate the state right AFTER a Stop+Create has been issued on
        # the previous tick but BEFORE the exchange book WebSocket has
        # reflected the cancel:
        #   - Our new executor is at 0.4365 (the create from previous tick,
        #     recovering after being baited down by a now-pulled low spoof)
        #   - Book still shows our just-cancelled 0.4316 at the bottom of
        #     asks (cancel-lag — this was the previous tick's chase target
        #     before the spoofer pulled)
        #   - _pending_cancels has the 0.4316 order recorded, still within
        #     the debounce window
        new_executor = _fake_executor(price=Decimal("0.4365"), amount=Decimal("45.8"))
        new_executor.id = "new-after-create"

        # Ask book in increasing-price order (walker reads ask_entries()
        # lowest-first). Ghost 0.4316 sits at the bottom; without the fix
        # the walker picks it and chases one tick lower.
        lagged_asks = [
            (Decimal("0.4316"), Decimal("45.3103")),  # OUR ghost (just-cancelled)
            (Decimal("0.4365"), Decimal("45.8")),  # our active
            (Decimal("0.4366"), Decimal("71.9366")),  # real top of ask book
            (Decimal("0.4541"), Decimal("110.4634")),
        ]

        config = BBOPegSellConfig(
            id="test",
            controller_name="bbo_peg_sell",
            connector_name="htx",
            trading_pair="XNO-USDT",
            total_amount_quote=Decimal("20"),
            update_interval=2.0,
            min_spread_pct=Decimal("0.02"),
            cancel_debounce_seconds=2.0,
        )
        market_data_provider = MagicMock(spec=MarketDataProvider)
        rules = MagicMock()
        rules.min_price_increment = Decimal("0.0001")
        market_data_provider.get_trading_rules.return_value = rules
        # best_bid (opposite side) — deep enough that spread stays > 2%.
        market_data_provider.get_price_by_type.return_value = Decimal("0.4182")
        market_data_provider.get_order_book.return_value = _fake_order_book(lagged_asks)
        market_data_provider.quantize_order_price.side_effect = lambda _c, _p, price: (
            price
        )
        market_data_provider.quantize_order_amount.side_effect = lambda _c, _p, amount: (
            amount
        )
        market_data_provider.time.return_value = 1000.0

        controller = BBOPegSellController(
            config=config,
            market_data_provider=market_data_provider,
            actions_queue=AsyncMock(spec=asyncio.Queue),
        )
        setattr(controller, "executors_info", [new_executor])
        setattr(controller, "logger", MagicMock(return_value=MagicMock()))
        # Seed the pending-cancel state: the 0.4316 order was Stopped on the
        # previous tick (at t=999), well within the 2s debounce window.
        controller._pending_cancels["just-cancelled"] = (
            Decimal("0.4316"),
            Decimal("45.3103"),
            1001.0,  # until = previous_now + 2s = 999 + 2 = 1001 > 1000 (now)
        )

        # Run the tick.
        await controller.update_processed_data()

        # ASSERTION 1 — walker correctly identifies 0.4316 as ours (pending),
        # skips it (plus our active 0.4365), and picks the real external best
        # ask 0.4366. Without the fix, external_best_ask would be 0.4316 (ghost).
        self.assertEqual(
            controller.processed_data["external_best_ask"], Decimal("0.4366")
        )

        # ASSERTION 2 — target stays at 0.4365, not jumping down to 0.4315.
        # This is the self-chase prevention in action.
        self.assertEqual(controller.processed_data["target_price"], Decimal("0.4365"))

        # ASSERTION 3 — controller emits NO actions this tick: the active
        # executor is already at the correct price 0.4365 (in_tolerance).
        # Without the fix, we'd see Stop(0.4365) + Create(0.4315) here.
        actions = controller.determine_executor_actions()
        self.assertEqual(actions, [])


if __name__ == "__main__":
    unittest.main()

import asyncio
import unittest
from decimal import Decimal
from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase
from typing import List, Optional, Tuple, cast
from unittest.mock import AsyncMock, MagicMock

from hummingbot.core.data_type.common import TradeType
from hummingbot.data_feed.market_data_provider import MarketDataProvider
from hummingbot.strategy_v2.executors.order_executor.data_types import (
    ExecutionStrategy,
    OrderExecutorConfig,
)
from hummingbot.strategy_v2.models.executor_actions import (
    CreateExecutorAction,
    StopExecutorAction,
)
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo

from controllers.market_making.bbo_peg_buy import BBOPegBuyConfig, BBOPegBuyController


def _fake_bid_row(price: Decimal, amount: Decimal) -> MagicMock:
    row = MagicMock()
    row.price = price
    row.amount = amount
    return row


def _fake_order_book(bid_levels: List[Tuple[Decimal, Decimal]]) -> MagicMock:
    """Mock OrderBook whose bid_entries() yields top-down (price, amount) rows."""
    book = MagicMock()
    book.bid_entries.return_value = [_fake_bid_row(p, a) for p, a in bid_levels]
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
    defensive filter in _external_best_bid)."""
    executor = MagicMock()
    executor.is_active = is_active
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
    bid_levels: Optional[List[Tuple[Decimal, Decimal]]] = None,
    executors: Optional[List[MagicMock]] = None,
) -> Tuple[BBOPegBuyController, MagicMock]:
    """Factory for _external_best_bid tests. Wires a fake order book and
    executors_info; returns (controller, log_mock) so tests can assert on
    log emissions via log_mock.info.call_count.
    """
    config = BBOPegBuyConfig(
        id="test",
        controller_name="bbo_peg_buy",
        connector_name="htx",
        trading_pair="XNO-USDT",
        total_amount_quote=Decimal("20"),
        update_interval=0.5,
    )
    market_data_provider = MagicMock(spec=MarketDataProvider)
    market_data_provider.get_order_book.return_value = _fake_order_book(
        bid_levels or []
    )

    controller = BBOPegBuyController(
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
    best_ask: Decimal = Decimal("0.4385"),
    external_best_bid: Optional[Decimal] = Decimal("0.4380"),
) -> BBOPegBuyController:
    """Factory that wires a controller with mocked market data.

    quantize_order_price is an identity passthrough so target_price math is
    exactly external_best_bid + tick — easy to assert in tests.
    _external_best_bid is patched to return whatever the test asks for.
    """
    config = BBOPegBuyConfig(
        id="test",
        controller_name="bbo_peg_buy",
        connector_name="htx",
        trading_pair="XNO-USDT",
        total_amount_quote=Decimal("20"),
        update_interval=0.5,
    )
    market_data_provider = MagicMock(spec=MarketDataProvider)
    rules = MagicMock()
    rules.min_price_increment = tick
    market_data_provider.get_trading_rules.return_value = rules
    market_data_provider.get_price_by_type.return_value = best_ask
    market_data_provider.quantize_order_price.side_effect = lambda _c, _p, price: price

    controller = BBOPegBuyController(
        config=config,
        market_data_provider=market_data_provider,
        actions_queue=AsyncMock(spec=asyncio.Queue),
    )
    setattr(controller, "_external_best_bid", MagicMock(return_value=external_best_bid))
    return controller


class TestBBOPegBuyControllerInit(unittest.TestCase):
    def setUp(self):
        self.config = BBOPegBuyConfig(
            id="test",
            controller_name="bbo_peg_buy",
            connector_name="htx",
            trading_pair="XNO-USDT",
            total_amount_quote=Decimal("20"),
            update_interval=0.5,
        )
        self.market_data_provider = MagicMock(spec=MarketDataProvider)
        self.actions_queue = AsyncMock(spec=asyncio.Queue)

    def _make_controller(self, **kwargs) -> BBOPegBuyController:
        return BBOPegBuyController(
            config=self.config,
            market_data_provider=self.market_data_provider,
            actions_queue=self.actions_queue,
            **kwargs,
        )

    def test_initial_state(self):
        controller = self._make_controller()
        self.assertIs(controller.config, self.config)
        self.assertFalse(controller._has_filled)
        self.assertIsNone(controller._last_logged_external_best_bid)

    def test_update_interval_from_config(self):
        self.config.update_interval = 1.5
        controller = self._make_controller()
        self.assertEqual(controller.update_interval, 1.5)

    def test_explicit_kwarg_overrides_config(self):
        self.config.update_interval = 0.5
        controller = self._make_controller(update_interval=2.0)
        self.assertEqual(controller.update_interval, 2.0)


class TestBBOPegBuyUpdateProcessedData(IsolatedAsyncioWrapperTestCase):
    async def test_golden_path_target_is_one_tick_above_bid(self):
        controller = _make_controller_with_market(
            tick=Decimal("0.0001"),
            best_ask=Decimal("0.4385"),
            external_best_bid=Decimal("0.4380"),
        )
        await controller.update_processed_data()
        self.assertEqual(controller.processed_data["target_price"], Decimal("0.4381"))

    async def test_no_external_bid_target_is_none(self):
        controller = _make_controller_with_market(external_best_bid=None)
        await controller.update_processed_data()
        self.assertIsNone(controller.processed_data["target_price"])

    async def test_zero_external_bid_target_is_none(self):
        controller = _make_controller_with_market(external_best_bid=Decimal("0"))
        await controller.update_processed_data()
        self.assertIsNone(controller.processed_data["target_price"])

    async def test_candidate_equal_to_ask_target_is_none(self):
        # bid 0.4384 + tick 0.0001 = 0.4385 == best_ask → blocked (strict <)
        controller = _make_controller_with_market(
            tick=Decimal("0.0001"),
            best_ask=Decimal("0.4385"),
            external_best_bid=Decimal("0.4384"),
        )
        await controller.update_processed_data()
        self.assertIsNone(controller.processed_data["target_price"])

    async def test_candidate_above_ask_target_is_none(self):
        # bid 0.4390 + tick 0.0001 = 0.4391 > best_ask 0.4385 → blocked
        controller = _make_controller_with_market(
            tick=Decimal("0.0001"),
            best_ask=Decimal("0.4385"),
            external_best_bid=Decimal("0.4390"),
        )
        await controller.update_processed_data()
        self.assertIsNone(controller.processed_data["target_price"])

    async def test_zero_best_ask_target_is_none(self):
        # falsy ask short-circuits the crossing guard regardless of math
        controller = _make_controller_with_market(
            best_ask=Decimal("0"),
            external_best_bid=Decimal("0.4380"),
        )
        await controller.update_processed_data()
        self.assertIsNone(controller.processed_data["target_price"])

    async def test_processed_data_contains_all_fields(self):
        controller = _make_controller_with_market(
            tick=Decimal("0.0001"),
            best_ask=Decimal("0.4385"),
            external_best_bid=Decimal("0.4380"),
        )
        await controller.update_processed_data()
        self.assertEqual(controller.processed_data["tick"], Decimal("0.0001"))
        self.assertEqual(
            controller.processed_data["external_best_bid"], Decimal("0.4380")
        )
        self.assertEqual(controller.processed_data["best_ask"], Decimal("0.4385"))
        self.assertEqual(controller.processed_data["target_price"], Decimal("0.4381"))


class TestBBOPegBuyExternalBestBid(unittest.TestCase):
    def test_empty_book_returns_none(self):
        controller, _ = _make_controller_for_walker(bid_levels=[])
        self.assertIsNone(controller._external_best_bid())

    def test_top_level_all_external_returns_top(self):
        controller, _ = _make_controller_for_walker(
            bid_levels=[(Decimal("0.4380"), Decimal("100"))]
        )
        self.assertEqual(controller._external_best_bid(), Decimal("0.4380"))

    def test_top_level_fully_ours_returns_next_level(self):
        # Top level (0.4380, 50) is entirely ours; walker drops to next external level.
        controller, _ = _make_controller_for_walker(
            bid_levels=[
                (Decimal("0.4380"), Decimal("50")),
                (Decimal("0.4379"), Decimal("100")),
            ],
            executors=[_fake_executor(price=Decimal("0.4380"), amount=Decimal("50"))],
        )
        self.assertEqual(controller._external_best_bid(), Decimal("0.4379"))

    def test_top_level_partially_ours_still_returns_top(self):
        # 30 of 100 at top is ours → 70 external → keep top.
        controller, _ = _make_controller_for_walker(
            bid_levels=[(Decimal("0.4380"), Decimal("100"))],
            executors=[_fake_executor(price=Decimal("0.4380"), amount=Decimal("30"))],
        )
        self.assertEqual(controller._external_best_bid(), Decimal("0.4380"))

    def test_inactive_executors_ignored(self):
        # Cancelled/stopped executor doesn't count toward our volume.
        controller, _ = _make_controller_for_walker(
            bid_levels=[(Decimal("0.4380"), Decimal("100"))],
            executors=[
                _fake_executor(
                    price=Decimal("0.4380"), amount=Decimal("100"), is_active=False
                )
            ],
        )
        self.assertEqual(controller._external_best_bid(), Decimal("0.4380"))

    def test_non_order_executor_configs_ignored(self):
        # Config isn't an OrderExecutorConfig instance → skip in volume map.
        controller, _ = _make_controller_for_walker(
            bid_levels=[(Decimal("0.4380"), Decimal("100"))],
            executors=[
                _fake_executor(
                    price=Decimal("0.4380"),
                    amount=Decimal("100"),
                    use_order_executor_config=False,
                )
            ],
        )
        self.assertEqual(controller._external_best_bid(), Decimal("0.4380"))

    def test_executor_with_none_price_ignored(self):
        # OrderExecutorConfig but price=None → skip in volume map.
        controller, _ = _make_controller_for_walker(
            bid_levels=[(Decimal("0.4380"), Decimal("100"))],
            executors=[_fake_executor(price=None, amount=Decimal("100"))],
        )
        self.assertEqual(controller._external_best_bid(), Decimal("0.4380"))

    def test_multiple_executors_same_price_sum_volumes(self):
        # Two 30-unit executors at 0.4380 → 60 ours → 40 external → keep top.
        controller, _ = _make_controller_for_walker(
            bid_levels=[(Decimal("0.4380"), Decimal("100"))],
            executors=[
                _fake_executor(price=Decimal("0.4380"), amount=Decimal("30")),
                _fake_executor(price=Decimal("0.4380"), amount=Decimal("30")),
            ],
        )
        self.assertEqual(controller._external_best_bid(), Decimal("0.4380"))

    def test_walker_caps_at_ten_levels(self):
        # 15 fully-ours levels; walker stops at index 10 → never finds external → None.
        bid_levels = [(Decimal(f"0.43{80 - i:02d}"), Decimal("10")) for i in range(15)]
        executors = [
            _fake_executor(price=p, amount=Decimal("10")) for p, _ in bid_levels
        ]
        controller, _ = _make_controller_for_walker(
            bid_levels=bid_levels, executors=executors
        )
        self.assertIsNone(controller._external_best_bid())

    def test_logs_only_when_result_changes(self):
        # Same book on both calls → same result → log fires once, not twice.
        controller, log_mock = _make_controller_for_walker(
            bid_levels=[(Decimal("0.4380"), Decimal("100"))]
        )
        controller._external_best_bid()
        controller._external_best_bid()
        self.assertEqual(log_mock.info.call_count, 1)

    def test_log_cache_updates_to_new_result(self):
        # Cache starts None, gets updated to the chosen price after first walk.
        controller, _ = _make_controller_for_walker(
            bid_levels=[(Decimal("0.4380"), Decimal("100"))]
        )
        self.assertIsNone(controller._last_logged_external_best_bid)
        controller._external_best_bid()
        self.assertEqual(controller._last_logged_external_best_bid, Decimal("0.4380"))

    # --- Defensive / fortification tests for the book walker ---

    def test_multi_level_book_picks_top_not_lower(self):
        # 3 fully external levels → must pick the highest (first-match-wins).
        controller, _ = _make_controller_for_walker(
            bid_levels=[
                (Decimal("0.4380"), Decimal("100")),
                (Decimal("0.4379"), Decimal("100")),
                (Decimal("0.4378"), Decimal("100")),
            ]
        )
        self.assertEqual(controller._external_best_bid(), Decimal("0.4380"))

    def test_first_match_wins_picks_top_not_largest_external(self):
        # Top has tiny external volume; deeper level has massive external.
        # Walker still picks the TOP — it's "highest external bid", not
        # "biggest external bid".
        controller, _ = _make_controller_for_walker(
            bid_levels=[
                (Decimal("0.4380"), Decimal("1")),
                (Decimal("0.4379"), Decimal("10000")),
            ]
        )
        self.assertEqual(controller._external_best_bid(), Decimal("0.4380"))

    def test_skip_two_owned_levels_returns_third(self):
        # Top 2 levels fully ours; 3rd is external → return 3rd.
        controller, _ = _make_controller_for_walker(
            bid_levels=[
                (Decimal("0.4380"), Decimal("100")),
                (Decimal("0.4379"), Decimal("50")),
                (Decimal("0.4378"), Decimal("200")),
            ],
            executors=[
                _fake_executor(price=Decimal("0.4380"), amount=Decimal("100")),
                _fake_executor(price=Decimal("0.4379"), amount=Decimal("50")),
            ],
        )
        self.assertEqual(controller._external_best_bid(), Decimal("0.4378"))

    def test_partial_ownership_chain(self):
        # Top fully ours → skip. Next level partially ours (70 external) → pick.
        # Even though level 3 is fully external, we stop at first level with
        # external > 0.
        controller, _ = _make_controller_for_walker(
            bid_levels=[
                (Decimal("0.4380"), Decimal("50")),
                (Decimal("0.4379"), Decimal("100")),
                (Decimal("0.4378"), Decimal("200")),
            ],
            executors=[
                _fake_executor(price=Decimal("0.4380"), amount=Decimal("50")),
                _fake_executor(price=Decimal("0.4379"), amount=Decimal("30")),
            ],
        )
        self.assertEqual(controller._external_best_bid(), Decimal("0.4379"))

    def test_negative_external_amount_is_skipped(self):
        # Our tracked volume EXCEEDS what's in the book (stale/buggy state):
        # external_amount = 50 - 100 = -50 → not > 0 → defensive skip.
        # Walker drops to the next level rather than returning a price we
        # might over-fill at.
        controller, _ = _make_controller_for_walker(
            bid_levels=[
                (Decimal("0.4380"), Decimal("50")),
                (Decimal("0.4379"), Decimal("100")),
            ],
            executors=[_fake_executor(price=Decimal("0.4380"), amount=Decimal("100"))],
        )
        self.assertEqual(controller._external_best_bid(), Decimal("0.4379"))

    def test_own_volume_at_different_price_does_not_affect_top(self):
        # Our order is at 0.4379, but top of book is 0.4380. The own-volume
        # map is keyed by price, so 0.4380's external count should be
        # unaffected by our 0.4379 holding.
        controller, _ = _make_controller_for_walker(
            bid_levels=[
                (Decimal("0.4380"), Decimal("100")),
                (Decimal("0.4379"), Decimal("100")),
            ],
            executors=[_fake_executor(price=Decimal("0.4379"), amount=Decimal("100"))],
        )
        self.assertEqual(controller._external_best_bid(), Decimal("0.4380"))

    def test_eleventh_level_never_reached_even_if_external(self):
        # Boundary: top 10 fully ours, 11th is fully external. The cap
        # (i >= 10 break) MUST prevent the walker from ever seeing level 11.
        # Returns None.
        bid_levels = [
            (Decimal(f"0.43{80 - i:02d}"), Decimal("10")) for i in range(10)
        ]
        bid_levels.append((Decimal("0.4370"), Decimal("999")))  # external, level 11
        executors = [_fake_executor(price=p, amount=Decimal("10")) for p, _ in bid_levels[:10]]
        controller, _ = _make_controller_for_walker(
            bid_levels=bid_levels, executors=executors
        )
        self.assertIsNone(controller._external_best_bid())

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
        book.bid_entries.return_value = [row]

        config = BBOPegBuyConfig(
            id="test",
            controller_name="bbo_peg_buy",
            connector_name="htx",
            trading_pair="XNO-USDT",
            total_amount_quote=Decimal("20"),
            update_interval=0.5,
        )
        market_data_provider = MagicMock(spec=MarketDataProvider)
        market_data_provider.get_order_book.return_value = book
        controller = BBOPegBuyController(
            config=config,
            market_data_provider=market_data_provider,
            actions_queue=AsyncMock(spec=asyncio.Queue),
        )
        # Our own order at the same price expressed as Decimal — must
        # collide with the book's float→str→Decimal value.
        setattr(controller, "executors_info", [
            _fake_executor(price=Decimal("0.4380"), amount=Decimal("30"))
        ])
        setattr(controller, "logger", MagicMock(return_value=MagicMock()))

        # 100 in book - 30 ours = 70 external → pick top.
        self.assertEqual(controller._external_best_bid(), Decimal("0.4380"))


class TestBBOPegBuyComputeTargetPrice(unittest.TestCase):
    """Unit tests for _compute_target_price in isolation.

    Tested directly (not through update_processed_data) so failures point at
    pricing logic specifically, not at the orchestration around it.
    """

    def _make_controller(self, quantize_side_effect=None) -> BBOPegBuyController:
        """Builds a controller with quantize_order_price as identity passthrough
        unless a custom side_effect is provided. Stashes the mocked MDP on
        self.market_data_provider_mock so tests can assert on its calls without
        going through the spec-typed controller.market_data_provider attribute.
        """
        config = BBOPegBuyConfig(
            id="test",
            controller_name="bbo_peg_buy",
            connector_name="htx",
            trading_pair="XNO-USDT",
            total_amount_quote=Decimal("20"),
            update_interval=0.5,
        )
        market_data_provider = MagicMock(spec=MarketDataProvider)
        market_data_provider.quantize_order_price.side_effect = (
            quantize_side_effect or (lambda _c, _p, price: price)
        )
        self.market_data_provider_mock = market_data_provider
        return BBOPegBuyController(
            config=config,
            market_data_provider=market_data_provider,
            actions_queue=AsyncMock(spec=asyncio.Queue),
        )

    # --- Happy path ---

    def test_returns_candidate_when_bid_and_ask_provide_room(self):
        # bid 0.4380 + tick 0.0001 = 0.4381, ask 0.4385 → returns 0.4381.
        controller = self._make_controller()
        result = controller._compute_target_price(
            external_best_bid=Decimal("0.4380"),
            tick=Decimal("0.0001"),
            best_ask=Decimal("0.4385"),
        )
        self.assertEqual(result, Decimal("0.4381"))

    # --- external_best_bid input gates ---

    def test_returns_none_when_external_bid_is_none(self):
        controller = self._make_controller()
        result = controller._compute_target_price(
            external_best_bid=None,
            tick=Decimal("0.0001"),
            best_ask=Decimal("0.4385"),
        )
        self.assertIsNone(result)

    def test_returns_none_when_external_bid_is_zero(self):
        controller = self._make_controller()
        result = controller._compute_target_price(
            external_best_bid=Decimal("0"),
            tick=Decimal("0.0001"),
            best_ask=Decimal("0.4385"),
        )
        self.assertIsNone(result)

    def test_returns_none_when_external_bid_is_negative(self):
        # Defensive: bids should never be negative, but guard regardless.
        controller = self._make_controller()
        result = controller._compute_target_price(
            external_best_bid=Decimal("-0.0001"),
            tick=Decimal("0.0001"),
            best_ask=Decimal("0.4385"),
        )
        self.assertIsNone(result)

    # --- best_ask / crossing guard ---

    def test_returns_none_when_best_ask_is_zero(self):
        # Falsy ask short-circuits the crossing guard → bail.
        controller = self._make_controller()
        result = controller._compute_target_price(
            external_best_bid=Decimal("0.4380"),
            tick=Decimal("0.0001"),
            best_ask=Decimal("0"),
        )
        self.assertIsNone(result)

    def test_returns_none_when_candidate_equals_best_ask(self):
        # bid 0.4384 + tick 0.0001 = 0.4385 == best_ask → blocked (strict >=)
        controller = self._make_controller()
        result = controller._compute_target_price(
            external_best_bid=Decimal("0.4384"),
            tick=Decimal("0.0001"),
            best_ask=Decimal("0.4385"),
        )
        self.assertIsNone(result)

    def test_returns_none_when_candidate_above_best_ask(self):
        # bid 0.4390 + tick 0.0001 = 0.4391 > best_ask 0.4385 → clearly crosses.
        controller = self._make_controller()
        result = controller._compute_target_price(
            external_best_bid=Decimal("0.4390"),
            tick=Decimal("0.0001"),
            best_ask=Decimal("0.4385"),
        )
        self.assertIsNone(result)

    # --- Quantization plumbing ---

    def test_quantize_called_with_external_bid_plus_tick(self):
        # Verify quantize is called with the raw sum and exchange identifiers.
        controller = self._make_controller()
        controller._compute_target_price(
            external_best_bid=Decimal("0.4380"),
            tick=Decimal("0.0001"),
            best_ask=Decimal("0.4385"),
        )
        self.market_data_provider_mock.quantize_order_price.assert_called_once_with(
            "htx", "XNO-USDT", Decimal("0.4381")
        )

    def test_returns_quantized_value_not_raw_sum(self):
        # If quantize snaps to a different price (e.g., exchange rounds down to
        # the next valid tick), we must return that snapped value, not the
        # unrounded sum. Pins that we trust the quantizer's output.
        controller = self._make_controller(
            quantize_side_effect=lambda _c, _p, _price: Decimal("0.4380")
        )
        result = controller._compute_target_price(
            external_best_bid=Decimal("0.4380"),
            tick=Decimal("0.0001"),
            best_ask=Decimal("0.4385"),
        )
        self.assertEqual(result, Decimal("0.4380"))

    # --- Negative cases (what the function must NOT do) ---

    def test_does_not_call_quantize_when_external_bid_is_none(self):
        # Early return must skip the quantize side effect entirely. If someone
        # refactors and breaks the guard, we'd silently call quantize with None.
        controller = self._make_controller()
        controller._compute_target_price(
            external_best_bid=None,
            tick=Decimal("0.0001"),
            best_ask=Decimal("0.4385"),
        )
        self.market_data_provider_mock.quantize_order_price.assert_not_called()

    def test_does_not_call_quantize_when_external_bid_is_zero(self):
        # Same short-circuit guard, zero case.
        controller = self._make_controller()
        controller._compute_target_price(
            external_best_bid=Decimal("0"),
            tick=Decimal("0.0001"),
            best_ask=Decimal("0.4385"),
        )
        self.market_data_provider_mock.quantize_order_price.assert_not_called()


class TestBBOPegBuyWalkBidsForFirstExternal(unittest.TestCase):
    """Direct unit tests for _walk_bids_for_first_external in isolation.

    The result-finding behavior is also covered indirectly through
    TestBBOPegBuyExternalBestBid, but the top_levels return contract
    (size cap at 5, top-down ordering, float-tuple shape) is only
    tested directly here.
    """

    # --- Result return value ---

    def test_empty_book_returns_none_result_and_empty_top_levels(self):
        controller, _ = _make_controller_for_walker(bid_levels=[])
        result, top_levels = controller._walk_bids_for_first_external({})
        self.assertIsNone(result)
        self.assertEqual(top_levels, [])

    def test_single_external_level_returns_top_price(self):
        controller, _ = _make_controller_for_walker(
            bid_levels=[(Decimal("0.4380"), Decimal("100"))]
        )
        result, _ = controller._walk_bids_for_first_external({})
        self.assertEqual(result, Decimal("0.4380"))

    def test_returns_none_when_all_levels_fully_ours(self):
        # 3 levels, all fully ours via my_volume_by_price → no external → None.
        controller, _ = _make_controller_for_walker(
            bid_levels=[
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
        result, _ = controller._walk_bids_for_first_external(my_volume)
        self.assertIsNone(result)

    def test_caps_at_ten_levels_for_result(self):
        # 11 levels; first 10 fully ours, 11th would be external. The cap
        # (i >= 10 break) stops the walker BEFORE seeing the 11th → None.
        bid_levels = [
            (Decimal(f"0.43{80 - i:02d}"), Decimal("10")) for i in range(10)
        ]
        bid_levels.append((Decimal("0.4370"), Decimal("999")))  # 11th, external
        my_volume = {p: Decimal("10") for p, _ in bid_levels[:10]}
        controller, _ = _make_controller_for_walker(bid_levels=bid_levels)
        result, _ = controller._walk_bids_for_first_external(my_volume)
        self.assertIsNone(result)

    # --- top_levels return contract ---

    def test_top_levels_capped_at_five_when_book_is_larger(self):
        # 7 levels in book → top_levels exactly 5 (top-5 capture for logging).
        bid_levels = [
            (Decimal(f"0.43{80 - i:02d}"), Decimal("100")) for i in range(7)
        ]
        controller, _ = _make_controller_for_walker(bid_levels=bid_levels)
        _, top_levels = controller._walk_bids_for_first_external({})
        self.assertEqual(len(top_levels), 5)

    def test_top_levels_contains_all_when_book_has_fewer_than_five(self):
        # 3 levels → all 3 in top_levels (no padding, no truncation).
        controller, _ = _make_controller_for_walker(
            bid_levels=[
                (Decimal("0.4380"), Decimal("100")),
                (Decimal("0.4379"), Decimal("50")),
                (Decimal("0.4378"), Decimal("200")),
            ]
        )
        _, top_levels = controller._walk_bids_for_first_external({})
        self.assertEqual(len(top_levels), 3)

    def test_top_levels_preserves_top_down_order(self):
        # First entry = top of book, descending from there.
        controller, _ = _make_controller_for_walker(
            bid_levels=[
                (Decimal("0.4380"), Decimal("100")),
                (Decimal("0.4379"), Decimal("50")),
                (Decimal("0.4378"), Decimal("200")),
            ]
        )
        _, top_levels = controller._walk_bids_for_first_external({})
        expected = [
            (float(Decimal("0.4380")), float(Decimal("100"))),
            (float(Decimal("0.4379")), float(Decimal("50"))),
            (float(Decimal("0.4378")), float(Decimal("200"))),
        ]
        self.assertEqual(top_levels, expected)

    def test_top_levels_entries_are_float_tuples(self):
        # Pin the (float, float) shape — important for log message format.
        controller, _ = _make_controller_for_walker(
            bid_levels=[(Decimal("0.4380"), Decimal("100"))]
        )
        _, top_levels = controller._walk_bids_for_first_external({})
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
            bid_levels=[
                (Decimal("0.4380"), Decimal("100")),
                (Decimal("0.4379"), Decimal("50")),
            ]
        )
        my_volume = {Decimal("0.4380"): Decimal("100")}
        result, _ = controller._walk_bids_for_first_external(my_volume)
        self.assertEqual(result, Decimal("0.4379"))

    # --- Negative cases (what the walker must NOT do) ---

    def test_never_returns_a_fully_owned_price_in_mixed_book(self):
        # Critical safety invariant: the walker must NEVER return a price
        # we fully own. If it did, we'd peg one tick above our own quote,
        # which is the self-anchoring runaway the walker exists to prevent.
        # Book has fully-owned levels at 0.4380 and 0.4378, external at 0.4379.
        controller, _ = _make_controller_for_walker(
            bid_levels=[
                (Decimal("0.4380"), Decimal("100")),  # fully ours
                (Decimal("0.4379"), Decimal("50")),   # external
                (Decimal("0.4378"), Decimal("200")),  # fully ours
            ]
        )
        my_volume = {
            Decimal("0.4380"): Decimal("100"),
            Decimal("0.4378"): Decimal("200"),
        }
        result, _ = controller._walk_bids_for_first_external(my_volume)
        self.assertNotEqual(result, Decimal("0.4380"))
        self.assertNotEqual(result, Decimal("0.4378"))


class TestBBOPegBuyComputeOwnVolumeByPrice(unittest.TestCase):
    """Direct unit tests for _compute_own_volume_by_price in isolation.

    Aggregates active executors' volume into a per-price dict. The filters
    are also exercised indirectly through TestBBOPegBuyExternalBestBid,
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


class TestBBOPegBuyLogExternalBidChange(unittest.TestCase):
    """Direct unit tests for _log_external_bid_change in isolation.

    Pins the change-detection log behavior: when log fires, when it
    doesn't, cache updates, transitions through None, and the
    Decimal-to-float conversion in the log payload.
    """

    def _make_controller(self) -> BBOPegBuyController:
        """Builds a controller and stashes the log mock on self.log_mock so
        tests can assert on .info calls without going through the bound
        controller.logger() method.
        """
        config = BBOPegBuyConfig(
            id="test",
            controller_name="bbo_peg_buy",
            connector_name="htx",
            trading_pair="XNO-USDT",
            total_amount_quote=Decimal("20"),
            update_interval=0.5,
        )
        market_data_provider = MagicMock(spec=MarketDataProvider)
        log_mock = MagicMock()
        controller = BBOPegBuyController(
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
        controller._last_logged_external_best_bid = Decimal("0.4380")
        controller._log_external_bid_change(
            result=Decimal("0.4380"),
            top_levels=[],
            my_volume_by_price={},
        )
        self.log_mock.info.assert_not_called()

    def test_logs_when_result_changes_from_none_to_value(self):
        # Initial detection — cache starts as None, first walker observation
        # is a real value. Must fire so the first peg target is auditable.
        controller = self._make_controller()
        self.assertIsNone(controller._last_logged_external_best_bid)
        controller._log_external_bid_change(
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
        controller._last_logged_external_best_bid = Decimal("0.4380")
        controller._log_external_bid_change(
            result=None,
            top_levels=[],
            my_volume_by_price={},
        )
        self.log_mock.info.assert_called_once()

    def test_logs_when_result_changes_between_two_values(self):
        # Walker picks a new external bid tick over tick (e.g., a level
        # ahead of us was hit). Standard transition — must log.
        controller = self._make_controller()
        controller._last_logged_external_best_bid = Decimal("0.4380")
        controller._log_external_bid_change(
            result=Decimal("0.4379"),
            top_levels=[],
            my_volume_by_price={},
        )
        self.log_mock.info.assert_called_once()

    # --- Cache update behavior ---

    def test_cache_updates_to_new_result_after_logging(self):
        controller = self._make_controller()
        self.assertIsNone(controller._last_logged_external_best_bid)
        controller._log_external_bid_change(
            result=Decimal("0.4380"),
            top_levels=[],
            my_volume_by_price={},
        )
        self.assertEqual(
            controller._last_logged_external_best_bid, Decimal("0.4380")
        )

    # --- Log payload format (regression catchers) ---

    def test_log_message_includes_all_diagnostic_fields(self):
        # Forensic log line must carry all three pieces of state so future
        # readers can reconstruct what the walker saw.
        controller = self._make_controller()
        controller._log_external_bid_change(
            result=Decimal("0.4380"),
            top_levels=[(0.4380, 100.0), (0.4379, 50.0)],
            my_volume_by_price={Decimal("0.4380"): Decimal("30")},
        )
        log_message = self.log_mock.info.call_args[0][0]
        self.assertIn("[bbo_peg]", log_message)
        self.assertIn("external_best_bid", log_message)
        self.assertIn("top5_bids", log_message)
        self.assertIn("my_volume", log_message)

    def test_my_volume_converted_to_float_in_log_message(self):
        # The function builds {float(k): float(v) ...} before logging.
        # If someone "simplifies" this by removing the float() calls, the
        # log shows Decimal repr ("Decimal('0.4380'): Decimal('30')") instead
        # of friendly floats — ugly and harder to grep. Pin the conversion.
        controller = self._make_controller()
        controller._log_external_bid_change(
            result=Decimal("0.4380"),
            top_levels=[],
            my_volume_by_price={Decimal("0.4380"): Decimal("30")},
        )
        log_message = self.log_mock.info.call_args[0][0]
        self.assertNotIn("Decimal", log_message)


class TestBBOPegBuyBuildCreateAction(unittest.TestCase):
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
    ) -> BBOPegBuyController:
        """Builds a controller with quantize_order_amount and time() mocked.
        Stashes the mocked MDP on self.market_data_provider_mock so tests
        can assert on call args directly.
        """
        config = BBOPegBuyConfig(
            id="test-controller-id",
            controller_name="bbo_peg_buy",
            connector_name="htx",
            trading_pair="XNO-USDT",
            total_amount_quote=Decimal("20"),
            update_interval=0.5,
        )
        market_data_provider = MagicMock(spec=MarketDataProvider)
        market_data_provider.quantize_order_amount.side_effect = (
            quantize_amount_side_effect or (lambda _c, _p, amount: amount)
        )
        market_data_provider.time.return_value = timestamp
        self.market_data_provider_mock = market_data_provider
        return BBOPegBuyController(
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
        self.assertEqual(config.side, TradeType.BUY)

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


class TestBBOPegBuyBuildStopActions(unittest.TestCase):
    """Direct unit tests for _build_stop_actions in isolation.

    Pure list-comprehension transformer: every executor in → one
    StopExecutorAction out, no filtering. Pinning the contract so future
    refactors don't accidentally add filtering (which would silently leave
    stale orders alive on the exchange).
    """

    def _make_controller(self) -> BBOPegBuyController:
        config = BBOPegBuyConfig(
            id="test-controller-id",
            controller_name="bbo_peg_buy",
            connector_name="htx",
            trading_pair="XNO-USDT",
            total_amount_quote=Decimal("20"),
            update_interval=0.5,
        )
        return BBOPegBuyController(
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


class TestBBOPegBuyCategorizeActiveOrders(unittest.TestCase):
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
        stale, in_tolerance = controller._categorize_active_orders(
            Decimal("0.4381")
        )
        self.assertEqual(stale, [])
        self.assertEqual(in_tolerance, [])

    def test_executor_at_target_price_categorized_as_in_tolerance(self):
        executor = _fake_executor(price=Decimal("0.4381"), amount=Decimal("50"))
        controller, _ = _make_controller_for_walker(executors=[executor])
        stale, in_tolerance = controller._categorize_active_orders(
            Decimal("0.4381")
        )
        self.assertEqual(stale, [])
        self.assertEqual(in_tolerance, [executor])

    def test_executor_at_different_price_categorized_as_stale(self):
        executor = _fake_executor(price=Decimal("0.4380"), amount=Decimal("50"))
        controller, _ = _make_controller_for_walker(executors=[executor])
        stale, in_tolerance = controller._categorize_active_orders(
            Decimal("0.4381")
        )
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
        stale, in_tolerance = controller._categorize_active_orders(
            Decimal("0.4381")
        )
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
        stale, in_tolerance = controller._categorize_active_orders(
            Decimal("0.4381")
        )
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
        stale, in_tolerance = controller._categorize_active_orders(
            Decimal("0.4381")
        )
        self.assertEqual(stale, [])
        self.assertEqual(in_tolerance, [])

    def test_executor_with_none_price_excluded(self):
        # OrderExecutorConfig but price is None — must NOT appear in either.
        executor = _fake_executor(price=None, amount=Decimal("50"))
        controller, _ = _make_controller_for_walker(executors=[executor])
        stale, in_tolerance = controller._categorize_active_orders(
            Decimal("0.4381")
        )
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
        stale, in_tolerance = controller._categorize_active_orders(
            Decimal("0.4381")
        )
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
        stale, in_tolerance = controller._categorize_active_orders(
            Decimal("0.4381")
        )
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
        stale, in_tolerance = controller._categorize_active_orders(
            Decimal("0.4381")
        )
        self.assertEqual(stale, [])
        self.assertEqual(in_tolerance, [])


class TestBBOPegBuyUpdateFillLatch(unittest.TestCase):
    """Direct unit tests for _update_fill_latch in isolation.

    Pins the one-shot fill detection contract: latch flips to True on any
    executor reporting executed_amount_base > 0, and NEVER resets within
    a process lifetime. The latch gates whether the controller emits
    more buy orders, so the never-reset guarantee is safety-critical.
    """

    def _make_controller(self) -> BBOPegBuyController:
        config = BBOPegBuyConfig(
            id="test",
            controller_name="bbo_peg_buy",
            connector_name="htx",
            trading_pair="XNO-USDT",
            total_amount_quote=Decimal("20"),
            update_interval=0.5,
        )
        return BBOPegBuyController(
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
        setattr(controller, "executors_info", [
            self._fake_executor_with_fill(executed_amount_base=Decimal("0")),
        ])
        controller._update_fill_latch()
        self.assertFalse(controller._has_filled)

    def test_latch_flips_to_true_on_executor_with_positive_fill(self):
        controller = self._make_controller()
        setattr(controller, "executors_info", [
            self._fake_executor_with_fill(executed_amount_base=Decimal("5")),
        ])
        controller._update_fill_latch()
        self.assertTrue(controller._has_filled)

    # --- Filter / skip cases ---

    def test_executor_with_none_custom_info_is_skipped(self):
        # Defensive: if framework hasn't populated custom_info, skip cleanly
        # rather than crash trying to .get() on None.
        controller = self._make_controller()
        setattr(controller, "executors_info", [
            self._fake_executor_with_fill(custom_info_is_none=True),
        ])
        controller._update_fill_latch()
        self.assertFalse(controller._has_filled)

    def test_executor_with_missing_executed_amount_base_key_is_skipped(self):
        # custom_info exists but lacks the key → dict.get returns None → skip.
        controller = self._make_controller()
        setattr(controller, "executors_info", [
            self._fake_executor_with_fill(missing_key=True),
        ])
        controller._update_fill_latch()
        self.assertFalse(controller._has_filled)

    def test_executor_with_zero_executed_amount_does_not_flip_latch(self):
        # Strict > 0 guard: exactly zero must not trip the latch.
        controller = self._make_controller()
        setattr(controller, "executors_info", [
            self._fake_executor_with_fill(executed_amount_base=Decimal("0")),
        ])
        controller._update_fill_latch()
        self.assertFalse(controller._has_filled)

    # --- Multiple executors ---

    def test_latch_flips_when_one_of_many_executors_has_fill(self):
        # First two have no fill; third does. The loop must continue past the
        # first negatives and trip the latch on the matching executor.
        controller = self._make_controller()
        setattr(controller, "executors_info", [
            self._fake_executor_with_fill(executed_amount_base=Decimal("0")),
            self._fake_executor_with_fill(custom_info_is_none=True),
            self._fake_executor_with_fill(executed_amount_base=Decimal("10")),
        ])
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
        setattr(controller, "executors_info", [
            self._fake_executor_with_fill(executed_amount_base=Decimal("0")),
            self._fake_executor_with_fill(custom_info_is_none=True),
        ])
        controller._update_fill_latch()
        self.assertTrue(controller._has_filled)

    # --- Defensive / negative cases ---

    def test_negative_executed_amount_does_not_flip_latch(self):
        # Defensive: executed_amount_base should never be negative, but the
        # strict > 0 guard catches this case too. If someone replaced > 0
        # with != 0 (treating any non-zero as a fill), this would catch it.
        controller = self._make_controller()
        setattr(controller, "executors_info", [
            self._fake_executor_with_fill(executed_amount_base=Decimal("-1")),
        ])
        controller._update_fill_latch()
        self.assertFalse(controller._has_filled)

    def test_string_executed_amount_handled_via_decimal_conversion(self):
        # Custom info may come from JSON deserialization → values are strings.
        # The Decimal(str(executed)) conversion must handle string input.
        # If someone "simplifies" this to Decimal(executed) directly, a string
        # input might fail (in some scenarios) or behave unexpectedly.
        controller = self._make_controller()
        setattr(controller, "executors_info", [
            self._fake_executor_with_fill(executed_amount_base="5.0"),
        ])
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
        setattr(controller, "executors_info", [
            self._fake_executor_with_fill(
                executed_amount_base=Decimal("5"),
                is_active=False,
            ),
        ])
        controller._update_fill_latch()
        self.assertTrue(controller._has_filled)


if __name__ == "__main__":
    unittest.main()

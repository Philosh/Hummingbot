import asyncio
import unittest
from decimal import Decimal
from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase
from typing import List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

from hummingbot.data_feed.market_data_provider import MarketDataProvider
from hummingbot.strategy_v2.executors.order_executor.data_types import (
    OrderExecutorConfig,
)

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


if __name__ == "__main__":
    unittest.main()

import asyncio
import unittest
from decimal import Decimal
from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

from hummingbot.data_feed.market_data_provider import MarketDataProvider

from controllers.market_making.bbo_peg_buy import BBOPegBuyConfig, BBOPegBuyController


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


if __name__ == "__main__":
    unittest.main()

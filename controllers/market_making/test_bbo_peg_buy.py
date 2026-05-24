import asyncio
import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from hummingbot.data_feed.market_data_provider import MarketDataProvider

from controllers.market_making.bbo_peg_buy import BBOPegBuyConfig, BBOPegBuyController


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


if __name__ == "__main__":
    unittest.main()

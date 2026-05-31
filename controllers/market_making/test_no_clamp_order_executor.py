import unittest
from decimal import Decimal
from typing import Optional
from unittest.mock import MagicMock, PropertyMock

from hummingbot.connector.exchange_py_base import ExchangePyBase
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.core.data_type.common import TradeType
from hummingbot.strategy.strategy_v2_base import StrategyV2Base
from hummingbot.strategy_v2.executors.order_executor.data_types import (
    ExecutionStrategy,
    LimitChaserConfig,
    OrderExecutorConfig,
)

from controllers.market_making.no_clamp_order_executor import NoClampOrderExecutor


def _make_strategy() -> MagicMock:
    """Minimal mocked StrategyV2Base sufficient to instantiate an OrderExecutor.
    Mirrors the pattern from upstream test_order_executor.py.
    """
    strategy = MagicMock(spec=StrategyV2Base)
    type(strategy).trading_pair = PropertyMock(return_value="XNO-USDT")
    connector = MagicMock(spec=ExchangePyBase)
    type(connector).trading_rules = PropertyMock(
        return_value={"XNO-USDT": TradingRule(trading_pair="XNO-USDT")}
    )
    strategy.connectors = {"htx": connector}
    return strategy


def _make_config(
    *,
    side: TradeType = TradeType.BUY,
    price: Optional[Decimal] = Decimal("0.4296"),
    execution_strategy: ExecutionStrategy = ExecutionStrategy.LIMIT_MAKER,
    chaser_config: Optional[LimitChaserConfig] = None,
) -> OrderExecutorConfig:
    return OrderExecutorConfig(
        id="test",
        timestamp=123.0,
        connector_name="htx",
        trading_pair="XNO-USDT",
        side=side,
        amount=Decimal("46"),
        price=price,
        execution_strategy=execution_strategy,
        chaser_config=chaser_config,
    )


def _make_executor(
    config: OrderExecutorConfig, current_market_price: Decimal
) -> NoClampOrderExecutor:
    """Builds a NoClampOrderExecutor with current_market_price stubbed.

    current_market_price is the property the stock parent uses inside the
    LIMIT_MAKER clamp. Patching it lets us pin exactly what the pre-fix code
    WOULD have clamped down to.
    """
    executor = NoClampOrderExecutor(_make_strategy(), config)
    type(executor).current_market_price = PropertyMock(
        return_value=current_market_price
    )
    return executor


class TestNoClampOrderExecutorGetOrderPrice(unittest.TestCase):
    """Pin the behavior change: BUY LIMIT_MAKER no longer clamps to best_bid.

    Stock OrderExecutor.get_order_price() for BUY LIMIT_MAKER does
    min(config.price, current_best_bid), which silently downgrades the
    intended price to current_best_bid. NoClampOrderExecutor returns
    config.price as-is so the BBO peg controller can actually rest one tick
    above current_best_bid.

    Every other code path (SELL LIMIT_MAKER, MARKET, LIMIT_CHASER, plain
    LIMIT) delegates to super().get_order_price() — pinned via negative tests
    so the override is provably surgical.
    """

    # --- The fix: BUY LIMIT_MAKER returns config.price unclamped ---

    def test_buy_limit_maker_returns_config_price_above_best_bid(self):
        # The exact stuck-state scenario from the bot logs: target = 0.4296,
        # current_best_bid = 0.4295. Stock executor would return 0.4295
        # (clamp). NoClampOrderExecutor returns 0.4296 so we actually become
        # the new top of book.
        config = _make_config(side=TradeType.BUY, price=Decimal("0.4296"))
        executor = _make_executor(config, current_market_price=Decimal("0.4295"))
        self.assertEqual(executor.get_order_price(), Decimal("0.4296"))

    def test_buy_limit_maker_returns_config_price_below_best_bid(self):
        # Defensive: even if the controller decides on a price BELOW
        # best_bid (conservative bidder, or transient race), still trust the
        # controller — no clamp in either direction.
        config = _make_config(side=TradeType.BUY, price=Decimal("0.4290"))
        executor = _make_executor(config, current_market_price=Decimal("0.4295"))
        self.assertEqual(executor.get_order_price(), Decimal("0.4290"))

    def test_buy_limit_maker_returns_config_price_equal_to_best_bid(self):
        # Edge: target == best_bid. Stock would also return this value
        # (min(x, x) = x), but pin the equality case so a future refactor
        # can't accidentally regress.
        config = _make_config(side=TradeType.BUY, price=Decimal("0.4295"))
        executor = _make_executor(config, current_market_price=Decimal("0.4295"))
        self.assertEqual(executor.get_order_price(), Decimal("0.4295"))

    # --- Negative: untouched code paths delegate to parent ---

    def test_sell_limit_maker_falls_through_to_parent(self):
        # CRITICAL: ONLY BUY LIMIT_MAKER is overridden. SELL LIMIT_MAKER
        # must still use parent's max(config.price, current_best_ask) clamp.
        # config.price = 0.4400, best_ask = 0.4405 -> parent returns 0.4405.
        # If this test ever fails, the override accidentally hijacked SELLs.
        config = _make_config(side=TradeType.SELL, price=Decimal("0.4400"))
        executor = _make_executor(config, current_market_price=Decimal("0.4405"))
        self.assertEqual(executor.get_order_price(), Decimal("0.4405"))

    def test_market_order_falls_through_to_parent(self):
        # Parent's MARKET path returns Decimal("NaN"). Pin via is_nan check.
        # If the override broadens beyond BUY LIMIT_MAKER, this test would
        # likely return config.price (0.4296) instead of NaN.
        config = _make_config(
            execution_strategy=ExecutionStrategy.MARKET,
            price=Decimal("0.4296"),
        )
        executor = _make_executor(config, current_market_price=Decimal("0.4295"))
        result = executor.get_order_price()
        self.assertTrue(result.is_nan())

    def test_plain_limit_falls_through_to_parent(self):
        # Plain LIMIT (not LIMIT_MAKER): parent returns config.price as-is
        # (same outcome as our override, but via a different code path).
        # Pin that we don't accidentally hijack the LIMIT path too — the
        # override condition checks LIMIT_MAKER specifically.
        config = _make_config(
            execution_strategy=ExecutionStrategy.LIMIT,
            price=Decimal("0.4296"),
        )
        executor = _make_executor(config, current_market_price=Decimal("0.4295"))
        self.assertEqual(executor.get_order_price(), Decimal("0.4296"))

    def test_buy_limit_chaser_falls_through_to_parent(self):
        # LIMIT_CHASER does its own price computation in parent
        # (current_market_price * (1 - distance) for BUY). Must NOT be
        # overridden — pin the parent's chaser math is preserved.
        config = _make_config(
            execution_strategy=ExecutionStrategy.LIMIT_CHASER,
            chaser_config=LimitChaserConfig(
                distance=Decimal("0.01"), refresh_threshold=Decimal("0.02"),
            ),
        )
        executor = _make_executor(config, current_market_price=Decimal("0.4295"))
        expected = Decimal("0.4295") * (Decimal("1") - Decimal("0.01"))
        self.assertEqual(executor.get_order_price(), expected)


if __name__ == "__main__":
    unittest.main()

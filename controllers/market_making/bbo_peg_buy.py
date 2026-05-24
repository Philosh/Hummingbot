"""
BBO Peg Buy Controller — minimal V2 controller scaffold.

Pegs a single LIMIT_MAKER buy at external_best_bid + 1 tick, where
external_best_bid excludes our own resting order (so we never anchor
to ourselves). Re-quotes on any drift. No close-side, no triple
barrier, no rebalance. WS-fed via MarketDataProvider — zero HTTP
polling at the strategy layer.
"""

from decimal import Decimal
from typing import Dict, List, Optional

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


class BBOPegBuyConfig(ControllerConfigBase):
    controller_type: str = "market_making"
    controller_name: str = "bbo_peg_buy"

    connector_name: str = Field(default="htx")
    trading_pair: str = Field(default="XNO-USDT")

    def update_markets(self, markets: MarketDict) -> MarketDict:
        # Upstream's add_or_update signature mistypes *args as the set type
        # itself instead of a set element; mirrors MarketMakingControllerConfigBase.
        return markets.add_or_update(self.connector_name, self.trading_pair)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


class BBOPegBuyController(ControllerBase):
    config: BBOPegBuyConfig

    def __init__(self, config: BBOPegBuyConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config

    async def update_processed_data(self):
        rules = self.market_data_provider.get_trading_rules(
            self.config.connector_name, self.config.trading_pair
        )
        tick: Decimal = rules.min_price_increment

        best_ask: Decimal = self.market_data_provider.get_price_by_type(
            self.config.connector_name, self.config.trading_pair, PriceType.BestAsk
        )

        external_best_bid = self._external_best_bid()

        target_price: Optional[Decimal] = None
        if external_best_bid is not None and external_best_bid > 0:
            candidate = self.market_data_provider.quantize_order_price(
                self.config.connector_name, self.config.trading_pair, external_best_bid + tick
            )
            # LIMIT_MAKER is rejected if it would cross — skip this tick.
            if best_ask and candidate < best_ask:
                target_price = candidate

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
        """
        order_book = self.market_data_provider.get_order_book(
            self.config.connector_name, self.config.trading_pair
        )

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

        for row in order_book.bid_entries():
            price = Decimal(str(row.price))
            amount = Decimal(str(row.amount))
            external_amount = amount - my_volume_by_price.get(price, Decimal("0"))
            if external_amount > 0:
                return price
        return None

    def determine_executor_actions(self) -> List[ExecutorAction]:
        target_price: Optional[Decimal] = self.processed_data.get("target_price")
        if target_price is None:
            return []

        actions: List[ExecutorAction] = []
        active = [e for e in self.executors_info if e.is_active]

        stale = []
        in_tolerance = []
        for e in active:
            cfg = e.config
            if not isinstance(cfg, OrderExecutorConfig) or cfg.price is None:
                continue
            # Strict: any deviation from target_price triggers re-quote.
            if cfg.price != target_price:
                stale.append(e)
            else:
                in_tolerance.append(e)

        for e in stale:
            actions.append(
                StopExecutorAction(
                    controller_id=self.config.id,
                    executor_id=e.id,
                )
            )

        if not in_tolerance:
            amount = self.market_data_provider.quantize_order_amount(
                self.config.connector_name,
                self.config.trading_pair,
                self.config.total_amount_quote / target_price,
            )
            if amount > 0:
                actions.append(
                    CreateExecutorAction(
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
                )

        return actions

"""
BBO Peg Buy Controller — minimal V2 controller scaffold.

Pegs a single LIMIT_MAKER buy at best_bid + 1 tick. No close-side,
no triple barrier, no rebalance. WS-fed via MarketDataProvider — zero
HTTP polling at the strategy layer.
"""

from decimal import Decimal
from typing import List, Optional

from pydantic import Field

from hummingbot.core.data_type.common import PriceType, TradeType
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
    controller_type: str = "generic"
    controller_name: str = "bbo_peg_buy"

    connector_name: str = Field(default="htx")
    trading_pair: str = Field(default="XNO-USDT")

    requote_tolerance_ticks: int = Field(
        default=1,
        description="Re-quote when best_bid drifts more than N ticks from our quoted price.",
    )


class BBOPegBuyController(ControllerBase):
    def __init__(self, config: BBOPegBuyConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config

    async def update_processed_data(self):
        rules = self.market_data_provider.get_trading_rules(
            self.config.connector_name, self.config.trading_pair
        )
        tick: Decimal = rules.min_price_increment

        best_bid: Decimal = self.market_data_provider.get_price_by_type(
            self.config.connector_name, self.config.trading_pair, PriceType.BestBid
        )
        best_ask: Decimal = self.market_data_provider.get_price_by_type(
            self.config.connector_name, self.config.trading_pair, PriceType.BestAsk
        )

        target_price: Optional[Decimal] = None
        if best_bid and best_bid > 0:
            candidate = self.market_data_provider.quantize_order_price(
                self.config.connector_name, self.config.trading_pair, best_bid + tick
            )
            # LIMIT_MAKER is rejected if it would cross — skip this tick.
            if best_ask and candidate < best_ask:
                target_price = candidate

        self.processed_data = {
            "tick": tick,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "target_price": target_price,
        }

    def determine_executor_actions(self) -> List[ExecutorAction]:
        target_price: Optional[Decimal] = self.processed_data.get("target_price")
        if target_price is None:
            return []

        tick: Decimal = self.processed_data["tick"]
        tolerance = Decimal(self.config.requote_tolerance_ticks) * tick

        actions: List[ExecutorAction] = []
        active = [e for e in self.executors_info if e.is_active]

        stale = [
            e
            for e in active
            if abs(Decimal(str(e.config.price)) - target_price) > tolerance
        ]
        for e in stale:
            actions.append(
                StopExecutorAction(
                    controller_id=self.config.id,
                    executor_id=e.id,
                )
            )

        in_tolerance = [e for e in active if e not in stale]
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

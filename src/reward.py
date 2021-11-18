from typing import List
from tensortrade.env.default.rewards import RewardScheme
from tensortrade.env.generic.environment import TradingEnv
from tensortrade.oms.orders.trade import Trade

import numpy as np


class TradeCompletion(RewardScheme):
    registered_name = "trade-completion"

    def __init__(self, rpnl_threshold: float, reward_asymmetry: float) -> None:
        super().__init__()
        self.rpnl_threshold = rpnl_threshold
        self.reward_asymmetry = reward_asymmetry
        self.position = 0
        self.position_vwap = 0

    def process_trades(self, trades: List[Trade]) -> float:
        rel_rpnl = 0
        for trade in trades:
            price = float(trade.price)
            amount = float(trade.quantity.size) / price
            next_position = self.position + amount if trade.is_buy else -amount
            if np.sign(next_position) == np.sign(self.position) or self.position * self.position_vwap == 0:
                self.position_vwap = (self.position_vwap * self.position + price * amount) / (self.position + amount)
            else:
                # closed 100 long, price0 50 price1 70 -> spent 5000, gained 7000 -> rpnl = 7000 - 5000 = 2000, %rpnl = 2000 / 5000
                rel_rpnl += (price - self.position_vwap) / self.position_vwap
                self.position_vwap = price
            self.position = next_position
        return rel_rpnl

    def reward(self, env: TradingEnv) -> float:
        current_step = env.clock.step
        trades = env.action_scheme.broker.trades.values()
        current_trades = [trade for trade_list in trades for trade in trade_list if trade.step == current_step]
        rel_rpnl = self.process_trades(current_trades)
        if rel_rpnl >= self.reward_asymmetry * self.rpnl_threshold:
            reward = 1
        elif rel_rpnl <= -self.rpnl_threshold:
            reward = -1
        else:
            reward = rel_rpnl
        return reward

    def reset(self) -> None:
        self.position = 0
        self.position_vwap = 0

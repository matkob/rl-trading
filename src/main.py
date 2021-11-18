import tensortrade.env.default as default
import tensortrade.oms.services.execution.simulated as simulated
import tensortrade.agents as agents
import pandas as pd
import features as ft
import feed
import reward

from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.instruments import USDT, BTC
from tensortrade.oms.wallets import Wallet, Portfolio


def extract_features(quotes: pd.DataFrame) -> pd.DataFrame:
    mid = ((quotes["asks[0].price"] + quotes["bids[0].price"]) / 2).astype(float)
    features = [
        ft.lr(mid),
        ft.rsi(mid, period=20),
        ft.macd(mid, fast=10, slow=50, signal=5),
    ]
    features = pd.concat(features, axis=1)
    features.columns = ["lr", "rsi", "macd"]
    return features


feed = feed.TimeBasedFeed("data/", lambda df: df, lambda df: df, extract_features, nrows=10000)
binance = Exchange("binance", service=simulated.execute_order)(feed.get_price().rename("USDT-BTC"))
portfolio = Portfolio(USDT, [
    Wallet(binance, 10000 * USDT),
    Wallet(binance, 1 * BTC)
])
env = default.create(
    portfolio=portfolio,
    action_scheme=default.actions.ManagedRiskOrders(),
    reward_scheme=reward.TradeCompletion(rpnl_threshold=0.01, reward_asymmetry=2),
    feed=feed.get_features(),
    renderer_feed=feed.get_candles(),
    renderer=default.renderers.PlotlyTradingChart(),
    window_size=20
)

n_steps = 100
n_episodes = 500
agent = agents.A2CAgent(env)
# setting render interval to huge number will result in just one, final summary
agent.train(n_steps=n_steps, n_episodes=n_episodes, save_path="agents/", render_interval=99999999999999)
agent.env.render(episode=n_episodes, max_episodes=n_episodes, max_steps=n_steps)

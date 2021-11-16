from typing import Callable
from tensortrade.feed.core import Stream, DataFeed

import pandas as pd
import requests
import os
import features as ft


class Feed:

    @classmethod
    def fetch_tardis(cls, url: str, dest: str, **kwargs) -> pd.DataFrame:
        if not os.path.isfile(dest):
            resp = requests.get(url)
            open(dest, "wb").write(resp.content)
        return pd.read_csv(dest, compression="gzip", **kwargs)

    @classmethod
    def load(cls,
             path_prefix: str, 
             transform_quotes: Callable[[pd.DataFrame], pd.DataFrame], 
             transform_trades: Callable[[pd.DataFrame], pd.DataFrame], 
             extract_features: Callable[[pd.DataFrame], pd.DataFrame], 
             **kwargs
        ) -> "Feed":
        quotes_url = "https://datasets.tardis.dev/v1/binance-futures/book_snapshot_25/2020/02/01/BTCUSDT.csv.gz"
        quotes_path = f"{path_prefix}quotes.csv.gz"
        trades_url = "https://datasets.tardis.dev/v1/binance-futures/trades/2020/02/01/BTCUSDT.csv.gz"
        trades_path = f"{path_prefix}trades.csv.gz"
        quotes = transform_quotes(cls.fetch_tardis(quotes_url, quotes_path, **kwargs))
        trades = transform_trades(cls.fetch_tardis(trades_url, trades_path, **kwargs))
        features = extract_features(quotes)
        return Feed(quotes, features, trades)

    def __init__(self, quotes: pd.DataFrame, features: pd.DataFrame, trades: pd.DataFrame) -> None:
        self.quotes: pd.DataFrame = quotes
        self.features: pd.DataFrame = features
        self.trades: pd.DataFrame = trades

    def get_mid_price(self) -> Stream[float]:
        return Stream.source(list((self.quotes["asks[0].price"] + self.quotes["bids[0].price"]) / 2), dtype="float").rename("mid").clamp_min(0)

    def get_features(self) -> DataFeed:
        features = [Stream.source(list(self.features[c]), dtype="float").rename(c) for c in self.features.columns]
        datafeed = DataFeed(features)
        datafeed.compile()
        return datafeed

    def get_candles(self, interval="1S") -> DataFeed:
        trades = self.trades.set_index("datetime")
        candles = trades.resample(interval)
        return DataFeed([
            Stream.source(list(candles.count().index)).rename("date"),
            Stream.source(list(candles["price"].first().pad()), dtype="float").rename("open"),
            Stream.source(list(candles["price"].max().pad()), dtype="float").rename("high"),
            Stream.source(list(candles["price"].min().pad()), dtype="float").rename("low"),
            Stream.source(list(candles["price"].last().pad()), dtype="float").rename("close"),
            Stream.source(list(candles["amount"].sum()), dtype="float").rename("volume"),
        ])

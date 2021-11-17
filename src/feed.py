from abc import ABC, abstractmethod
from typing import Callable, List, Tuple
from pandas.core.resample import Resampler
from tensortrade.feed.core import Stream, DataFeed

import pandas as pd
import requests
import os
import features as ft


class ResampledFeed(ABC):

    @classmethod
    def convert_time(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df.astype({"timestamp": "datetime64[us]"}).set_index("timestamp")

    @classmethod
    def fetch_from_tardis(cls, url: str, dest: str, **kwargs) -> pd.DataFrame:
        if not os.path.isfile(dest):
            resp = requests.get(url)
            open(dest, "wb").write(resp.content)
        return pd.read_csv(dest, compression="gzip", **kwargs)

    @classmethod
    def load(
            cls,
            path_prefix: str, 
            transform_quotes: Callable[[pd.DataFrame], pd.DataFrame], 
            transform_trades: Callable[[pd.DataFrame], pd.DataFrame], 
            extract_features: Callable[[pd.DataFrame], pd.DataFrame],
            **kwargs
        ) -> List[pd.DataFrame]:
        quotes_url = "https://datasets.tardis.dev/v1/binance-futures/book_snapshot_25/2020/02/01/BTCUSDT.csv.gz"
        quotes_path = f"{path_prefix}quotes.csv.gz"
        trades_url = "https://datasets.tardis.dev/v1/binance-futures/trades/2020/02/01/BTCUSDT.csv.gz"
        trades_path = f"{path_prefix}trades.csv.gz"

        quotes = cls.convert_time(cls.fetch_from_tardis(quotes_url, quotes_path, **kwargs))
        trades = cls.convert_time(cls.fetch_from_tardis(trades_url, trades_path, **kwargs))
        return [transform_quotes(quotes), transform_trades(trades), extract_features(quotes)]

    def __init__(
            self, 
            path_prefix: str, 
            transform_quotes: Callable[[pd.DataFrame], pd.DataFrame], 
            transform_trades: Callable[[pd.DataFrame], pd.DataFrame], 
            extract_features: Callable[[pd.DataFrame], pd.DataFrame],
            **kwargs
        ) -> None:
        [quotes, trades, features] = self.load(path_prefix, transform_quotes, transform_trades, extract_features, **kwargs)
        self.quotes: pd.DataFrame = quotes
        self.features: pd.DataFrame = features
        self.trades: pd.DataFrame = trades

    @abstractmethod
    def resample(self) -> None:
        pass

    @abstractmethod
    def get_price(self) -> Stream[float]:
        pass

    @abstractmethod
    def get_features(self) -> DataFeed:
        pass

    @abstractmethod
    def get_candles(self) -> DataFeed:
        pass


class TimeBasedFeed(ResampledFeed):

    def __init__(
            self, 
            path_prefix: str, 
            transform_quotes: Callable[[pd.DataFrame], pd.DataFrame], 
            transform_trades: Callable[[pd.DataFrame], pd.DataFrame], 
            extract_features: Callable[[pd.DataFrame], pd.DataFrame],
            interval="1S", 
            **kwargs
        ) -> None:
        super().__init__(path_prefix, transform_quotes, transform_trades, extract_features, **kwargs)
        self.interval = interval
        self.resample()

    def resample(self) -> None:
        self.quotes = self.quotes.resample(self.interval)
        self.trades = self.trades.resample(self.interval)
        self.features = self.features.resample(self.interval)
        
    def get_price(self) -> Stream[float]:
        mid = (self.quotes["asks[0].price"].last().pad() + self.quotes["bids[0].price"].last().pad()) / 2
        return Stream.source(list(mid), dtype="float").rename("mid")

    def get_features(self) -> DataFeed:
        features = [Stream.source(list(self.features[c].last().pad()), dtype="float").rename(c) for c in self.features.count().columns]
        datafeed = DataFeed(features)
        datafeed.compile()
        return datafeed
    
    def get_candles(self) -> DataFeed:
        close = self.trades["price"].last().fillna(method="ffill")
        open = self.trades["price"].first().fillna(close)
        high = self.trades["price"].max().fillna(close)
        low = self.trades["price"].min().fillna(close)
        return DataFeed([
            Stream.source(list(self.trades.count().index)).rename("date"),
            Stream.source(list(open), dtype="float").rename("open"),
            Stream.source(list(high), dtype="float").rename("high"),
            Stream.source(list(low), dtype="float").rename("low"),
            Stream.source(list(close), dtype="float").rename("close"),
            Stream.source(list(self.trades["amount"].sum()), dtype="float").rename("volume"),
        ])

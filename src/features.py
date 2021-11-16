import pandas as pd
import numpy as np


def rsi(price: pd.Series, period: float) -> pd.Series:
    r = price.diff()
    upside = r.clip(lower=0).abs()
    downside = r.clip(upper=0).abs()
    rs = upside.ewm(alpha=1 / period).mean() / downside.ewm(alpha=1 / period).mean()
    return (100 *(1 - (1 + rs) ** -1))


def macd(price: pd.Series, fast: float, slow: float, signal: float) -> pd.Series:
    fm = price.ewm(span=fast, adjust=False).mean()
    sm = price.ewm(span=slow, adjust=False).mean()
    md = fm - sm
    signal = md - md.ewm(span=signal, adjust=False).mean()
    return signal


def lr(price: pd.Series) -> pd.Series:
    return np.log(price).diff()

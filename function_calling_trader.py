"""官網 function calling cookbook 的 trader.py：交易助理的十個函式，加上讀快取的 client。

原文 https://docs.typesafe.ai/cookbooks/function_calling 只說 trader.py「holds the ten functions,
plus a TypeSafe client that reads answers from a cache」，原始檔（CookSafe repo）沒有公開，
這裡照頁面上看得到的東西重寫：

  * 函式名稱、每個 closed-set 引數和它的形狀（choice / set / flag）照頁面上 closed_sets 的輸出；
    plot_price 的簽名照頁面原文；top_movers 的 limit: int = 3、rolling_correlation 預設
    「一個月、小時 K」照頁面的說明。其餘的 Literal 值和預設值是這裡選的。
  * market_summary、top_movers 的文字格式照頁面的範例輸出。

資料：官網的 1 分 K 沒有公開。load() 先找 function_calling_bars.parquet（欄位 symbol, timestamp,
open, high, low, close, volume；timestamp 是交易所當地時間），沒有就用固定 seed 產生模擬資料：
6 檔 × 67 個交易日 × 390 根 = 156,780 根，跟官網的筆數一樣；最後收盤價和每天平均成交量對齊官網
market_summary(window='1w') 那張表。走勢是模擬的（大盤因子 + 個股自己的波動，開盤和收盤附近
波動和成交量比較大），所以報酬率、三張圖的形狀不會和官網一樣。

client：官網是 TypeSafe client 加 cooksafe 的快取。這裡 make_client() 把本機分類器
（JevK5Classifier / H2OClassifier）包上重試和 JSON 快取，同一句話、同一組題目重跑時直接讀快取。
"""

import functools
import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np
import polars as pl
from langchain_core.runnables import Runnable
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)
from langchain_typesafe.types import Question
from matplotlib.figure import Figure
from numpy.lib.stride_tricks import sliding_window_view
from pydantic import TypeAdapter

HERE = Path(__file__).resolve().parent
BARS_PATH = HERE / "function_calling_bars.parquet"

Symbol = Literal["SPY", "NVDA", "AMD", "AAPL", "MSFT", "TSLA"]
Window = Literal["1d", "1w", "1mo", "3mo"]
Resolution = Literal["1m", "5m", "15m", "1h", "1d"]

SYMBOLS: tuple[str, ...] = get_args(Symbol)
WINDOW_DAYS = {"1d": 1, "1w": 5, "1mo": 21, "3mo": 63}
"""每個 window 往回幾個交易日（含最後一天）。"""


# ---- 模擬資料 ----

TRADING_DAYS, BARS_PER_DAY = 67, 390
LAST_DAY = np.datetime64("2026-10-16")
SEED = 20261004
# 官網 market_summary(window='1w') 表格的價格；成交量是那張表的一週成交量 ÷ 5
LAST_CLOSE = {
    "SPY": 664.86,
    "NVDA": 254.12,
    "AMD": 184.20,
    "AAPL": 258.71,
    "MSFT": 451.35,
    "TSLA": 320.22,
}
DAILY_VOLUME = {
    "SPY": 27.7e6,
    "NVDA": 77.9e6,
    "AMD": 36.5e6,
    "AAPL": 44.8e6,
    "MSFT": 22.7e6,
    "TSLA": 53.3e6,
}
# 每日波動（%）= BETA × 大盤 + 個股自己的部分
MARKET_DAILY_VOL = 0.9
BETA = {"SPY": 1.0, "NVDA": 1.6, "AMD": 1.8, "AAPL": 1.1, "MSFT": 0.9, "TSLA": 1.9}
IDIOSYNCRATIC = {
    "SPY": 0.0,
    "NVDA": 1.8,
    "AMD": 2.2,
    "AAPL": 1.0,
    "MSFT": 0.9,
    "TSLA": 2.8,
}


def _simulate() -> pl.DataFrame:
    rng = np.random.default_rng(SEED)
    days = np.busday_offset(LAST_DAY, np.arange(1 - TRADING_DAYS, 1), roll="backward")
    minute = np.arange(BARS_PER_DAY)
    # 開盤和收盤附近比較活躍（U 字形），整天的變異數不變
    activity = 1 + 1.6 * np.exp(-minute / 25) + 0.9 * np.exp(-(389 - minute) / 30)
    sigma_shape = activity / np.sqrt(np.mean(activity**2))
    volume_shape = activity**1.5 / np.sum(activity**1.5)
    per_minute = 1 / 100 / np.sqrt(BARS_PER_DAY)

    market = rng.standard_normal((TRADING_DAYS, BARS_PER_DAY)) * sigma_shape
    market_gap = rng.standard_normal(TRADING_DAYS) * 0.4 / 100
    timestamps = (
        (
            days.astype("datetime64[m]")[:, None]
            + np.timedelta64(9 * 60 + 30, "m")
            + minute.astype("timedelta64[m]")
        )
        .ravel()
        .astype("datetime64[us]")
    )  # polars 不收分鐘解析度

    frames = []
    for symbol in SYMBOLS:
        noise = rng.standard_normal((TRADING_DAYS, BARS_PER_DAY)) * sigma_shape
        returns = (
            BETA[symbol] * MARKET_DAILY_VOL * market + IDIOSYNCRATIC[symbol] * noise
        ) * per_minute
        gaps = BETA[symbol] * market_gap + rng.standard_normal(TRADING_DAYS) * (
            IDIOSYNCRATIC[symbol] * 0.3 / 100
        )
        gaps[0] = 0.0
        steps = returns.copy()
        steps[:, 0] += gaps
        log_close = np.cumsum(steps.ravel())
        log_close += np.log(LAST_CLOSE[symbol]) - log_close[-1]
        log_open = log_close - returns.ravel()
        bar_sigma = (
            np.sqrt(
                BETA[symbol] ** 2 * MARKET_DAILY_VOL**2 + IDIOSYNCRATIC[symbol] ** 2
            )
            * per_minute
            * np.tile(sigma_shape, TRADING_DAYS)
        )
        wick = np.abs(rng.standard_normal((2, log_close.size))) * 0.5 * bar_sigma
        close, open_ = np.exp(log_close), np.exp(log_open)
        high = np.maximum(open_, close) * np.exp(wick[0])
        low = np.minimum(open_, close) * np.exp(-wick[1])
        surprise = np.abs(returns.ravel()) / bar_sigma
        volume = (
            DAILY_VOLUME[symbol]
            * np.tile(volume_shape, TRADING_DAYS)
            * np.repeat(np.exp(rng.normal(0, 0.25, TRADING_DAYS)), BARS_PER_DAY)
            * np.exp(rng.normal(0, 0.35, log_close.size))
            * (0.6 + 0.4 * surprise)
        )
        frames.append(
            pl.DataFrame(
                {
                    "symbol": [symbol] * log_close.size,
                    "timestamp": timestamps,
                    "open": np.round(open_, 2),
                    "high": np.round(high, 2),
                    "low": np.round(low, 2),
                    "close": np.round(close, 2),
                    "volume": np.round(volume).astype(np.int64),
                }
            )
        )
    return pl.concat(frames)


@functools.cache
def load() -> pl.DataFrame:
    """全部的 1 分 K：function_calling_bars.parquet，沒有就用模擬資料。"""
    if BARS_PATH.exists():
        return pl.read_parquet(BARS_PATH).sort("symbol", "timestamp")
    return _simulate()


# ---- 共用的小工具 ----

FIGSIZE = (13.35, 4.63)
INK, MUTED, ACCENT, UP, DOWN = "#1f2937", "#9ca3af", "#d97706", "#15803d", "#b91c1c"
PALETTE = ["#2563eb", "#d97706", "#15803d", "#b91c1c", "#7c3aed", "#0891b2"]


def _bars(symbol: str, window: str, resolution: str = "1m") -> pl.DataFrame:
    """一檔股票最後 window 的 K 棒，1m 以外用 group_by_dynamic 合成（從 09:30 起算）。"""
    frame = load().filter(pl.col("symbol") == symbol)
    days = frame.get_column("timestamp").dt.date().unique().sort()
    first_day = days[max(0, len(days) - WINDOW_DAYS[window])]
    frame = frame.filter(pl.col("timestamp").dt.date() >= first_day)
    if resolution == "1m":
        return frame
    return frame.group_by_dynamic(
        "timestamp", every=resolution, start_by="datapoint"
    ).agg(
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
        pl.col("volume").sum(),
    )


def _return(bars: pl.DataFrame) -> float:
    return bars["close"][-1] / bars["open"][0] - 1


def _figure(rows: int = 1, height_ratios: list[int] | None = None):
    figure = Figure(figsize=FIGSIZE, layout="constrained")
    axes = figure.subplots(
        rows,
        1,
        sharex=True,
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios} if height_ratios else None,
    )[:, 0]
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        axis.set_axisbelow(True)
    return figure, axes


def _time_axis(axis, timestamps: pl.Series) -> None:
    """x 軸是 K 棒序號（跳過收盤後和週末），刻度標上時間。"""
    times = timestamps.to_list()
    if not times:
        return
    one_day = times[0].date() == times[-1].date()
    daily = len(times) > 1 and all(t.time() == times[0].time() for t in times)
    fmt = "%H:%M" if one_day else "%m-%d" if daily else "%m-%d %H:%M"
    ticks = sorted(set(np.linspace(0, len(times) - 1, 7).round().astype(int)))
    axis.set_xticks(ticks, [times[i].strftime(fmt) for i in ticks])
    axis.set_xlim(-0.5, len(times) - 0.5)


def _rolling_mean(values: np.ndarray, n: int) -> np.ndarray:
    out = np.full(values.size, np.nan)
    if values.size >= n:
        out[n - 1 :] = sliding_window_view(values, n).mean(axis=1)
    return out


def _rolling_corr(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    out = np.full(x.size, np.nan)
    if x.size >= n:
        wx, wy = sliding_window_view(x, n), sliding_window_view(y, n)
        dx = wx - wx.mean(axis=1, keepdims=True)
        dy = wy - wy.mean(axis=1, keepdims=True)
        denominator = np.sqrt((dx**2).sum(axis=1) * (dy**2).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[n - 1 :] = np.where(
                denominator > 0, (dx * dy).sum(axis=1) / denominator, np.nan
            )
    return out


# ---- 十個函式 ----


def list_symbols() -> str:
    """有哪些股票、載入了多少歷史。"""
    bars = load()
    days = bars.get_column("timestamp").dt.date()
    lines = [
        f"{bars.height:,} one-minute bars over {days.n_unique()} trading days, "
        f"{days.min()} to {days.max()}"
    ]
    counts = bars.group_by("symbol", maintain_order=True).len()
    for symbol, count in counts.iter_rows():
        lines.append(f"  {symbol:<6}{count:>9,} bars")
    return "\n".join(lines)


def market_summary(window: Window = "1d") -> str:
    """每檔一行：最後價格、window 內的報酬、成交量，照報酬排。"""
    rows = []
    for symbol in SYMBOLS:
        bars = _bars(symbol, window)
        rows.append((symbol, bars["close"][-1], _return(bars), bars["volume"].sum()))
    rows.sort(key=lambda row: -row[2])
    return "\n".join(
        [f"the board over {window}"]
        + [
            f"  {symbol:<6}{price:>9.2f}{change:>9.2%}{volume:>15,}"
            for symbol, price, change, volume in rows
        ]
    )


def plot_price(
    symbol: Literal["SPY", "NVDA", "AMD", "AAPL", "MSFT", "TSLA"],
    style: Literal["line", "candles"] = "line",
    resolution: Literal["1m", "5m", "15m", "1h", "1d"] = "15m",
    window: Literal["1d", "1w", "1mo", "3mo"] = "1w",
    include_volume: bool = False,
    moving_average: Literal["9", "20", "50"] | None = None,
    log_scale: bool = False,
) -> Figure:
    """一檔股票的價格圖（簽名照官網原文）。"""
    bars = _bars(symbol, window, resolution)
    figure, axes = _figure(
        2 if include_volume else 1, [3, 1] if include_volume else None
    )
    price_axis = axes[0]
    x = np.arange(bars.height)
    close = bars["close"].to_numpy()
    if style == "candles":
        open_ = bars["open"].to_numpy()
        rising = close >= open_
        colors = np.where(rising, UP, DOWN)
        price_axis.vlines(
            x,
            bars["low"].to_numpy(),
            bars["high"].to_numpy(),
            colors=colors,
            linewidth=0.7,
        )
        body = np.where(close == open_, (close * 2e-4), close - open_)
        price_axis.bar(x, body, bottom=open_, width=0.65, color=colors, linewidth=0)
    else:
        price_axis.plot(x, close, color=INK, linewidth=1.3)
    if moving_average is not None:
        bars_in_average = int(moving_average)
        price_axis.plot(
            x,
            _rolling_mean(close, bars_in_average),
            color=ACCENT,
            linewidth=1.4,
            label=f"{bars_in_average}-bar moving average",
        )
        price_axis.legend(frameon=False, loc="upper left")
    if log_scale:
        price_axis.set_yscale("log")
    price_axis.set_ylabel("price")
    price_axis.set_title(
        f"{symbol} · {style} · {resolution} bars · {window}", loc="left", color=INK
    )
    if include_volume:
        axes[1].bar(x, bars["volume"].to_numpy(), width=0.8, color=MUTED, linewidth=0)
        axes[1].set_ylabel("volume")
    _time_axis(axes[-1], bars["timestamp"])
    return figure


def intraday_pattern(
    symbol: Symbol,
    window: Window = "1mo",
    metric: Literal["volume", "range", "return"] = "volume",
) -> Figure:
    """一天裡每 15 分鐘的平均：成交量、高低價差（%）或報酬（%）。"""
    value = {
        "volume": pl.col("volume").cast(pl.Float64),
        "range": (pl.col("high") - pl.col("low")) / pl.col("open") * 100,
        "return": (pl.col("close") / pl.col("open") - 1) * 100,
    }[metric]
    label = {
        "volume": "shares traded",
        "range": "high-low range (%)",
        "return": "return (%)",
    }[metric]
    slots = (
        _bars(symbol, window, "15m")
        .with_columns(
            slot=pl.col("timestamp").dt.strftime("%H:%M"), value=value.alias("value")
        )
        .group_by("slot")
        .agg(pl.col("value").mean())
        .sort("slot")
    )
    figure, (axis,) = _figure()
    heights = slots["value"].to_numpy()
    x = np.arange(heights.size)
    if metric == "return":
        colors = np.where(heights >= 0, UP, DOWN)
        axis.axhline(0, color=MUTED, linewidth=0.8)
    else:
        colors = np.where(heights == heights.max(), ACCENT, MUTED)
    axis.bar(x, heights, width=0.8, color=colors, linewidth=0)
    axis.set_xticks(x[::2], slots["slot"].to_list()[::2])
    axis.set_ylabel(f"average {label}")
    axis.set_title(
        f"{symbol} · {label} by time of day, 15-minute slots · {window}",
        loc="left",
        color=INK,
    )
    return figure


def compare_returns(
    symbols: list[Symbol], window: Window = "1mo", normalize: bool = True
) -> Figure:
    """幾檔股票畫在同一張圖；normalize 時每條線從 0% 起算。"""
    resolution = {"1d": "5m", "1w": "15m", "1mo": "1h", "3mo": "1d"}[window]
    figure, (axis,) = _figure()
    timestamps = None
    for color, symbol in zip(PALETTE, symbols, strict=False):
        bars = _bars(symbol, window, resolution)
        close = bars["close"].to_numpy()
        y = (close / bars["open"][0] - 1) * 100 if normalize else close
        axis.plot(
            np.arange(close.size),
            y,
            color=color,
            linewidth=1.5,
            label=f"{symbol} {_return(bars):+.1%}",
        )
        if timestamps is None:
            timestamps = bars["timestamp"]
    if normalize:
        axis.axhline(0, color=MUTED, linewidth=0.8)
        axis.set_ylabel("return since start (%)")
    else:
        axis.set_ylabel("price")
    axis.legend(frameon=False, loc="upper left")
    axis.set_title(f"{', '.join(symbols)} · {window}", loc="left", color=INK)
    if timestamps is not None:
        _time_axis(axis, timestamps)
    return figure


ROLLING_BARS = 20


def rolling_correlation(
    symbol: Symbol,
    benchmark: Symbol = "SPY",
    window: Window = "1mo",
    resolution: Resolution = "1h",
) -> Figure:
    """兩檔股票報酬的滾動相關係數（每 20 根 K 棒，不夠就用三分之一）。"""
    joined = (
        _bars(symbol, window, resolution)
        .select("timestamp", pl.col("close").alias("a"))
        .join(
            _bars(benchmark, window, resolution).select(
                "timestamp", pl.col("close").alias("b")
            ),
            on="timestamp",
        )
        .sort("timestamp")
    )
    a = np.diff(np.log(joined["a"].to_numpy()))
    b = np.diff(np.log(joined["b"].to_numpy()))
    figure, (axis,) = _figure()
    if a.size < 3:
        axis.set_title(
            f"{symbol} vs {benchmark} · {resolution} returns · {window}: "
            f"only {a.size} returns, not enough for a correlation",
            loc="left",
            color=INK,
        )
        return figure
    span = min(ROLLING_BARS, max(3, a.size // 3))
    overall = float(np.corrcoef(a, b)[0, 1])
    axis.plot(
        np.arange(a.size), _rolling_corr(a, b, span), color=PALETTE[0], linewidth=1.5
    )
    axis.axhline(
        overall,
        color=ACCENT,
        linewidth=1,
        linestyle="--",
        label=f"whole window {overall:.2f}",
    )
    axis.axhline(0, color=MUTED, linewidth=0.8)
    axis.set_ylim(-1.05, 1.05)
    axis.set_ylabel(f"{span}-bar rolling correlation")
    axis.legend(frameon=False, loc="lower left")
    axis.set_title(
        f"{symbol} vs {benchmark} · {resolution} returns · {window}",
        loc="left",
        color=INK,
    )
    _time_axis(axis, joined["timestamp"][1:])
    return figure


def summary_stats(symbol: Symbol, window: Window = "1w") -> str:
    """一檔股票的數字表：開高低收、總報酬、平均每天的漲跌幅、平均每天的成交量。"""
    bars = _bars(symbol, window)
    daily = _bars(symbol, window, "1d")
    day_moves = (daily["close"] / daily["open"] - 1).abs()
    rows = [
        ("open", f"{bars['open'][0]:.2f}"),
        ("high", f"{bars['high'].max():.2f}"),
        ("low", f"{bars['low'].min():.2f}"),
        ("last", f"{bars['close'][-1]:.2f}"),
        ("total return", f"{_return(bars):.2%}"),
        ("average day move", f"{day_moves.mean():.2%}"),
        ("average day volume", f"{daily['volume'].mean():,.0f}"),
    ]
    return "\n".join(
        [f"{symbol} over {window}"]
        + [f"  {name:<20}{value:>14}" for name, value in rows]
    )


def volatility(symbol: Symbol, window: Window = "1mo", annualized: bool = True) -> str:
    """1 分 K 報酬（不含隔夜跳空）的標準差，換算成每天，annualized 再乘 √252。"""
    bars = _bars(symbol, window)
    returns = np.log(bars["close"].to_numpy() / bars["open"].to_numpy())
    per_day = float(np.std(returns) * np.sqrt(BARS_PER_DAY))
    if annualized:
        return (
            f"{symbol} volatility over {window}: "
            f"{per_day * np.sqrt(252):.1%} annualized ({per_day:.2%} a day)"
        )
    return f"{symbol} volatility over {window}: {per_day:.2%} a day"


def top_movers(
    window: Window = "1d",
    direction: Literal["gainers", "losers"] = "gainers",
    limit: int = 3,
) -> str:
    """window 內漲最多（或跌最多）的前 limit 檔。"""
    rows = [(symbol, _bars(symbol, window)) for symbol in SYMBOLS]
    rows = [(symbol, _return(bars), bars["close"][-1]) for symbol, bars in rows]
    rows.sort(key=lambda row: row[1], reverse=direction == "gainers")
    return "\n".join(
        [f"top {limit} {direction} over {window}"]
        + [
            f"  {symbol:<6}{change:>9.2%}  ->  {price:.2f}"
            for symbol, change, price in rows[:limit]
        ]
    )


def drawdown(
    symbol: Symbol, window: Window = "3mo", plot: bool = False
) -> str | Figure:
    """離前高最深的一段跌幅；plot 時畫成水下圖（underwater chart）。"""
    bars = _bars(symbol, window)
    close = bars["close"].to_numpy()
    times = bars["timestamp"].to_list()
    below_peak = close / np.maximum.accumulate(close) - 1
    trough = int(np.argmin(below_peak))
    peak = int(np.argmax(close[: trough + 1]))
    text = (
        f"{symbol} worst drawdown over {window}: {below_peak[trough]:.2%}, "
        f"from {close[peak]:.2f} on {times[peak]:%Y-%m-%d %H:%M} "
        f"to {close[trough]:.2f} on {times[trough]:%Y-%m-%d %H:%M}"
    )
    if not plot:
        return text
    figure, (axis,) = _figure()
    x = np.arange(close.size)
    axis.fill_between(x, below_peak * 100, 0, color=DOWN, alpha=0.25, linewidth=0)
    axis.plot(x, below_peak * 100, color=DOWN, linewidth=1)
    axis.scatter([trough], [below_peak[trough] * 100], color=DOWN, zorder=3)
    axis.set_ylabel("below previous peak (%)")
    axis.set_title(text, loc="left", color=INK)
    _time_axis(axis, bars["timestamp"])
    return figure


TOOLS = {
    fn.__name__: fn
    for fn in (
        list_symbols,
        market_summary,
        plot_price,
        intraday_pattern,
        compare_returns,
        rolling_correlation,
        summary_stats,
        volatility,
        top_movers,
        drawdown,
    )
}


# ---- client ----

_QUESTION: TypeAdapter = TypeAdapter(Question)


class CachedClient:
    """取代官網 trader.py 的 client（TypeSafe client + cooksafe 快取）。

    ask(state, questions) 回 {題目 id: 答案的 JSON}。key 是句子加上整組題目的 sha256，
    所以改了 spec 就會重問，不會讀到舊題目的答案。快取檔存句子和答案，方便直接打開看。
    """

    def __init__(self, classifier: Runnable, path: Path) -> None:
        self.classifier = classifier
        self.path = path
        self.lock = threading.Lock()
        self.data: dict = json.loads(path.read_text()) if path.exists() else {}

    def ask(self, state: str, questions: dict[str, dict]) -> dict[str, dict]:
        key = hashlib.sha256(
            json.dumps([state, questions], sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        with self.lock:
            if key in self.data:
                return json.loads(json.dumps(self.data[key]["answers"]))
        response = self.classifier.invoke(
            {
                "state": state,
                "questions": {
                    qid: _QUESTION.validate_python(question)
                    for qid, question in questions.items()
                },
            }
        )
        answers = {
            qid: answer.model_dump(mode="json")
            for qid, answer in response.answers.items()
        }
        with self.lock:
            self.data[key] = {"state": state, "answers": answers}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.path.with_suffix(".tmp")
            temporary_path.write_text(
                json.dumps(self.data, ensure_ascii=False, indent=1)
            )
            temporary_path.replace(self.path)
        return json.loads(json.dumps(answers))


def make_client(classifier: Runnable, cache_path: Path) -> CachedClient:
    """本機分類器包上重試和 JSON 快取。

    重試跟 hierarchical_cpc.py 一樣（官網 TypeSafe client 的 RetryPolicy(max_retries=5,
    backoff 1~20 秒) 換成 LangChain 的 with_retry：第一次 + 重試 5 次）。
    """
    retrying = classifier.with_retry(
        retry_if_exception_type=(
            TypeSafeAPIConnectionError,  # 包含 TypeSafeAPITimeoutError
            TypeSafeRateLimitError,
            TypeSafeInternalServerError,
        ),
        stop_after_attempt=6,
        exponential_jitter_params={"initial": 1.0, "max": 20.0},
    )
    return CachedClient(retrying, cache_path)


def show(result: Any, path: Path) -> None:
    """取代 IPython 的 display：圖存成 PNG，文字直接印。"""
    if isinstance(result, Figure):
        path.parent.mkdir(parents=True, exist_ok=True)
        result.savefig(path, dpi=100)
        print(f"[chart saved to {path}]")
    else:
        print(result)

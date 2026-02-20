"""
Модуль расчёта технических индикаторов из pandas DataFrame.
Все индикаторы рассчитываются на чистом pandas без внешних библиотек.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI (Relative Strength Index)."""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """EMA (Exponential Moving Average)."""
    return df["close"].ewm(span=period, adjust=False).mean()


def calculate_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> Dict[str, pd.Series]:
    """MACD (Moving Average Convergence Divergence)."""
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def calculate_bollinger_bands(
    df: pd.DataFrame, period: int = 20, num_std: float = 2.0
) -> Dict[str, pd.Series]:
    """Bollinger Bands."""
    sma = df["close"].rolling(window=period).mean()
    std = df["close"].rolling(window=period).std()
    return {
        "upper": sma + num_std * std,
        "middle": sma,
        "lower": sma - num_std * std,
    }


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR (Average True Range)."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def compute_all_indicators(df: pd.DataFrame) -> Optional[Dict]:
    """
    Вычисляет все индикаторы и возвращает словарь с текущими значениями.

    Args:
        df: DataFrame с колонками open, high, low, close, volume

    Returns:
        Словарь с последними значениями индикаторов или None при ошибке
    """
    if df.empty or len(df) < 200:
        logger.warning(f"Недостаточно данных для расчёта индикаторов: {len(df)} свечей")
        return None

    try:
        rsi = calculate_rsi(df)
        ema20 = calculate_ema(df, 20)
        ema50 = calculate_ema(df, 50)
        ema200 = calculate_ema(df, 200)
        macd = calculate_macd(df)
        bb = calculate_bollinger_bands(df)
        atr = calculate_atr(df)

        current = df["close"].iloc[-1]

        # Определяем тренд по EMA
        if current > ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1]:
            ema_trend = "сильный восходящий"
        elif current > ema50.iloc[-1] > ema200.iloc[-1]:
            ema_trend = "восходящий"
        elif current < ema20.iloc[-1] < ema50.iloc[-1] < ema200.iloc[-1]:
            ema_trend = "сильный нисходящий"
        elif current < ema50.iloc[-1] < ema200.iloc[-1]:
            ema_trend = "нисходящий"
        else:
            ema_trend = "боковой"

        # RSI интерпретация
        rsi_val = rsi.iloc[-1]
        if rsi_val > 70:
            rsi_signal = "перекуплен"
        elif rsi_val < 30:
            rsi_signal = "перепродан"
        else:
            rsi_signal = "нейтральный"

        # MACD сигнал
        macd_val = macd["macd"].iloc[-1]
        signal_val = macd["signal"].iloc[-1]
        hist_val = macd["histogram"].iloc[-1]
        hist_prev = macd["histogram"].iloc[-2]
        if macd_val > signal_val and hist_val > hist_prev:
            macd_signal = "бычий (MACD выше сигнальной, гистограмма растёт)"
        elif macd_val < signal_val and hist_val < hist_prev:
            macd_signal = "медвежий (MACD ниже сигнальной, гистограмма падает)"
        else:
            macd_signal = "нейтральный"

        # Bollinger Bands позиция
        bb_upper = bb["upper"].iloc[-1]
        bb_lower = bb["lower"].iloc[-1]
        bb_middle = bb["middle"].iloc[-1]
        bb_width = (bb_upper - bb_lower) / bb_middle * 100  # ширина в %
        if current >= bb_upper:
            bb_signal = "цена у верхней границы (возможен откат)"
        elif current <= bb_lower:
            bb_signal = "цена у нижней границы (возможен отскок)"
        else:
            bb_position = (current - bb_lower) / (bb_upper - bb_lower) * 100
            bb_signal = f"цена на {bb_position:.0f}% от нижней границы"

        return {
            "rsi": {"value": rsi_val, "signal": rsi_signal},
            "ema": {
                "ema20": ema20.iloc[-1],
                "ema50": ema50.iloc[-1],
                "ema200": ema200.iloc[-1],
                "trend": ema_trend,
            },
            "macd": {
                "macd": macd_val,
                "signal": signal_val,
                "histogram": hist_val,
                "interpretation": macd_signal,
            },
            "bollinger": {
                "upper": bb_upper,
                "middle": bb_middle,
                "lower": bb_lower,
                "width_pct": bb_width,
                "interpretation": bb_signal,
            },
            "atr": {"value": atr.iloc[-1]},
        }

    except Exception as e:
        logger.error(f"Ошибка расчёта индикаторов: {e}", exc_info=True)
        return None


def format_indicators_for_prompt(indicators: Dict, symbol: str) -> str:
    """
    Форматирует индикаторы в текст для промпта AI.

    Args:
        indicators: Словарь из compute_all_indicators()
        symbol: Символ инструмента

    Returns:
        Отформатированная строка
    """
    if not indicators:
        return f"Технические индикаторы для {symbol} недоступны"

    lines = [f"=== Технические индикаторы {symbol} ==="]

    rsi = indicators["rsi"]
    lines.append(f"RSI(14): {rsi['value']:.1f} — {rsi['signal']}")

    ema = indicators["ema"]
    lines.append(
        f"EMA: 20={ema['ema20']:.2f}, 50={ema['ema50']:.2f}, 200={ema['ema200']:.2f} — тренд: {ema['trend']}"
    )

    m = indicators["macd"]
    lines.append(
        f"MACD(12,26,9): MACD={m['macd']:.4f}, Signal={m['signal']:.4f}, Hist={m['histogram']:.4f} — {m['interpretation']}"
    )

    bb = indicators["bollinger"]
    lines.append(
        f"Bollinger(20,2): Upper={bb['upper']:.2f}, Mid={bb['middle']:.2f}, Lower={bb['lower']:.2f}, Ширина={bb['width_pct']:.1f}% — {bb['interpretation']}"
    )

    atr = indicators["atr"]
    lines.append(f"ATR(14): {atr['value']:.4f}")

    return "\n".join(lines)

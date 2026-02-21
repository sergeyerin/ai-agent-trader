"""
Бэктестер для проверки торговых стратегий на исторических данных.
Использует правила на основе технических индикаторов (без AI).

Запуск:
    python backtester.py --symbol BTCUSDT --days 7 --balance 100
"""

import argparse
import logging
import math
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from bybit_client import BybitClient
from indicators import (
    calculate_rsi,
    calculate_ema,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_atr,
)
from config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BacktestResult:
    """Результаты бэктеста."""

    def __init__(self):
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.initial_balance: float = 0
        self.final_balance: float = 0

    @property
    def total_return_pct(self) -> float:
        if self.initial_balance == 0:
            return 0
        return (self.final_balance - self.initial_balance) / self.initial_balance * 100

    @property
    def total_trades(self) -> int:
        return len([t for t in self.trades if t["action"] != "hold"])

    @property
    def winning_trades(self) -> int:
        return len([t for t in self.trades if t.get("pnl", 0) > 0])

    @property
    def losing_trades(self) -> int:
        return len([t for t in self.trades if t.get("pnl", 0) < 0])

    @property
    def win_rate(self) -> float:
        closed = self.winning_trades + self.losing_trades
        if closed == 0:
            return 0
        return self.winning_trades / closed * 100

    @property
    def max_drawdown_pct(self) -> float:
        if not self.equity_curve:
            return 0
        peak = self.equity_curve[0]
        max_dd = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            max_dd = max(max_dd, dd)
        return max_dd

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe Ratio (без risk-free rate)."""
        if len(self.equity_curve) < 2:
            return 0
        returns = []
        for i in range(1, len(self.equity_curve)):
            r = (self.equity_curve[i] - self.equity_curve[i - 1]) / self.equity_curve[i - 1]
            returns.append(r)
        if not returns:
            return 0
        avg = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0
        # Аннуализация: предполагаем, что каждый шаг = 15 минут
        # ~35000 шагов в году (365 * 24 * 60 / 15)
        periods_per_year = 365 * 24 * 4
        return (avg / std) * math.sqrt(periods_per_year)

    @property
    def avg_profit(self) -> float:
        profits = [t["pnl"] for t in self.trades if t.get("pnl", 0) > 0]
        return np.mean(profits) if profits else 0

    @property
    def avg_loss(self) -> float:
        losses = [t["pnl"] for t in self.trades if t.get("pnl", 0) < 0]
        return np.mean(losses) if losses else 0

    def print_report(self):
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ БЭКТЕСТИНГА")
        print("=" * 60)
        print(f"Начальный баланс:  {self.initial_balance:.2f} USDT")
        print(f"Конечный баланс:   {self.final_balance:.2f} USDT")
        print(f"Доходность:        {self.total_return_pct:+.2f}%")
        print(f"Макс. просадка:    {self.max_drawdown_pct:.2f}%")
        print(f"Sharpe Ratio:      {self.sharpe_ratio:.2f}")
        print("-" * 60)
        print(f"Всего сделок:      {self.total_trades}")
        print(f"Прибыльных:        {self.winning_trades}")
        print(f"Убыточных:         {self.losing_trades}")
        print(f"Win Rate:          {self.win_rate:.1f}%")
        print(f"Ср. прибыль:       {self.avg_profit:+.4f} USDT")
        print(f"Ср. убыток:        {self.avg_loss:+.4f} USDT")
        print("=" * 60)


class Backtester:
    """
    Прогоняет rules-based стратегию на исторических данных.
    НЕ использует AI — чисто индикаторы.
    """

    FEE_PCT = 0.1  # Комиссия Bybit 0.1%

    def __init__(
        self,
        initial_balance: float = 100.0,
        position_size_pct: float = 5.0,
        stop_loss_pct: float = 5.0,
        take_profit_pct: float = 3.0,
    ):
        self.initial_balance = initial_balance
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def run(self, df: pd.DataFrame, symbol: str) -> BacktestResult:
        """
        Запускает бэктест на DataFrame с OHLCV данными.

        Стратегия:
        - BUY когда RSI < 35 И EMA20 > EMA50 И MACD гистограмма > 0
        - SELL когда RSI > 65 ИЛИ stop-loss ИЛИ take-profit
        - Иначе HOLD
        """
        result = BacktestResult()
        result.initial_balance = self.initial_balance

        if df.empty or len(df) < 200:
            logger.error(f"Недостаточно данных: {len(df)} свечей (нужно >= 200)")
            result.final_balance = self.initial_balance
            return result

        # Вычисляем индикаторы
        rsi = calculate_rsi(df)
        ema20 = calculate_ema(df, 20)
        ema50 = calculate_ema(df, 50)
        macd = calculate_macd(df)
        bb = calculate_bollinger_bands(df)
        atr = calculate_atr(df)

        balance = self.initial_balance
        position = None  # {qty, entry_price, entry_idx}

        # Пропускаем первые 200 свечей (прогрев индикаторов)
        for i in range(200, len(df)):
            price = df["close"].iloc[i]
            current_rsi = rsi.iloc[i]
            current_ema20 = ema20.iloc[i]
            current_ema50 = ema50.iloc[i]
            current_macd_hist = macd["histogram"].iloc[i]
            current_bb_lower = bb["lower"].iloc[i]
            current_atr = atr.iloc[i]

            if pd.isna(current_rsi) or pd.isna(current_atr):
                result.equity_curve.append(balance)
                continue

            # Текущая стоимость портфеля
            equity = balance
            if position:
                equity += position["qty"] * price

            result.equity_curve.append(equity)

            if position:
                # Проверяем stop-loss
                loss_pct = (price - position["entry_price"]) / position["entry_price"] * 100
                if loss_pct <= -self.stop_loss_pct:
                    pnl = self._close_position(position, price)
                    balance += position["qty"] * price - position["qty"] * price * self.FEE_PCT / 100
                    result.trades.append({
                        "action": "sell", "reason": "stop_loss",
                        "price": price, "pnl": pnl,
                        "idx": i, "timestamp": df["timestamp"].iloc[i],
                    })
                    position = None
                    continue

                # Проверяем take-profit
                if loss_pct >= self.take_profit_pct:
                    pnl = self._close_position(position, price)
                    balance += position["qty"] * price - position["qty"] * price * self.FEE_PCT / 100
                    result.trades.append({
                        "action": "sell", "reason": "take_profit",
                        "price": price, "pnl": pnl,
                        "idx": i, "timestamp": df["timestamp"].iloc[i],
                    })
                    position = None
                    continue

                # Сигнал на продажу: RSI > 65
                if current_rsi > 65:
                    pnl = self._close_position(position, price)
                    balance += position["qty"] * price - position["qty"] * price * self.FEE_PCT / 100
                    result.trades.append({
                        "action": "sell", "reason": "rsi_overbought",
                        "price": price, "pnl": pnl,
                        "idx": i, "timestamp": df["timestamp"].iloc[i],
                    })
                    position = None

            else:
                # Сигнал на покупку: RSI < 35 И EMA20 > EMA50 И MACD hist > 0
                if (
                    current_rsi < 35
                    and current_ema20 > current_ema50
                    and current_macd_hist > 0
                ):
                    # Размер позиции
                    trade_amount = balance * (self.position_size_pct / 100)
                    trade_amount = min(trade_amount, balance - 1)  # Оставляем 1 USDT

                    if trade_amount < 1:
                        continue

                    fee = trade_amount * self.FEE_PCT / 100
                    qty = (trade_amount - fee) / price
                    balance -= trade_amount
                    position = {
                        "qty": qty,
                        "entry_price": price,
                        "entry_amount": trade_amount,
                        "entry_idx": i,
                    }
                    result.trades.append({
                        "action": "buy", "reason": "signal",
                        "price": price, "pnl": 0,
                        "idx": i, "timestamp": df["timestamp"].iloc[i],
                    })

        # Закрываем позицию если осталась открытой
        if position:
            final_price = df["close"].iloc[-1]
            pnl = self._close_position(position, final_price)
            balance += position["qty"] * final_price
            result.trades.append({
                "action": "sell", "reason": "end_of_data",
                "price": final_price, "pnl": pnl,
                "idx": len(df) - 1,
            })

        result.final_balance = balance
        return result

    def _close_position(self, position: Dict, sell_price: float) -> float:
        """Рассчитывает P&L позиции с учётом комиссий."""
        sell_value = position["qty"] * sell_price
        fee = sell_value * self.FEE_PCT / 100
        pnl = sell_value - fee - position["entry_amount"]
        return pnl


def load_data_from_csv(symbol: str) -> Optional[pd.DataFrame]:
    """Загружает последний CSV файл из директории data/."""
    data_dir = "data"
    if not os.path.exists(data_dir):
        return None

    files = sorted(
        [f for f in os.listdir(data_dir) if f.startswith(symbol) and f.endswith(".csv")],
        reverse=True,
    )
    if not files:
        return None

    filepath = os.path.join(data_dir, files[0])
    logger.info(f"Загружаем данные из {filepath}")
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    return df


def load_data_from_bybit(symbol: str, days: int) -> pd.DataFrame:
    """Загружает данные напрямую с Bybit."""
    client = BybitClient()
    return client.get_historical_data(symbol, days)


def main():
    parser = argparse.ArgumentParser(description="Бэктестер торговой стратегии")
    parser.add_argument("--symbol", default="BTCUSDT", help="Торговая пара")
    parser.add_argument("--days", type=int, default=7, help="Дней истории")
    parser.add_argument("--balance", type=float, default=100.0, help="Начальный баланс USDT")
    parser.add_argument("--position-size", type=float, default=5.0, help="Размер позиции %% от портфеля")
    parser.add_argument("--stop-loss", type=float, default=5.0, help="Stop-loss %%")
    parser.add_argument("--take-profit", type=float, default=3.0, help="Take-profit %%")
    parser.add_argument("--from-csv", action="store_true", help="Загрузить из CSV вместо Bybit")
    args = parser.parse_args()

    # Загружаем данные
    if args.from_csv:
        df = load_data_from_csv(args.symbol)
        if df is None:
            logger.error(f"CSV файл для {args.symbol} не найден в data/")
            sys.exit(1)
    else:
        logger.info(f"Загрузка данных {args.symbol} за {args.days} дней с Bybit...")
        df = load_data_from_bybit(args.symbol, args.days)

    if df.empty:
        logger.error("Нет данных для бэктеста")
        sys.exit(1)

    logger.info(f"Загружено {len(df)} свечей для {args.symbol}")

    # Запускаем бэктест
    bt = Backtester(
        initial_balance=args.balance,
        position_size_pct=args.position_size,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
    )
    result = bt.run(df, args.symbol)
    result.print_report()


if __name__ == "__main__":
    main()

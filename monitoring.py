"""
Модуль мониторинга производительности торгового агента.
Расчёт метрик и вывод отчётов.
"""

import math
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from trade_history import TradeHistory

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    def __init__(self, trade_history: TradeHistory):
        self.history = trade_history

    def get_metrics(self, days: int = 7) -> Dict:
        """
        Рассчитывает все метрики производительности за период.

        Args:
            days: Количество дней

        Returns:
            Словарь с метриками
        """
        stats = self.history.get_performance_stats(days=days)
        trades = self.history.get_recent_trades(limit=500)

        # Фильтруем реальные сделки (не hold)
        real_trades = [t for t in trades if t["action"] != "hold"]
        sell_trades = [t for t in real_trades if t["action"] == "sell"]

        # Общий P&L
        pnl_by_symbol = stats.get("pnl_by_symbol", {})
        total_pnl = sum(s.get("realized_pnl", 0) for s in pnl_by_symbol.values())

        # Win rate
        total = stats.get("total_trades", 0)
        wins = 0
        losses = 0

        # Для Sharpe и drawdown нужна equity curve
        # Простое приближение: считаем кумулятивный P&L по sell-сделкам
        pnl_series = []
        for t in reversed(sell_trades):  # от старых к новым
            pnl_series.append(t.get("quantity_usdt", 0) * 0.01)  # приближение

        # Best/worst trade
        best_trade = None
        worst_trade = None
        trade_pnls = []

        for sym, data in pnl_by_symbol.items():
            rpnl = data.get("realized_pnl", 0)
            if rpnl > 0:
                wins += 1
            elif rpnl < 0:
                losses += 1
            trade_pnls.append({"symbol": sym, "pnl": rpnl})

        if trade_pnls:
            best_trade = max(trade_pnls, key=lambda x: x["pnl"])
            worst_trade = min(trade_pnls, key=lambda x: x["pnl"])

        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

        return {
            "period_days": days,
            "total_pnl": total_pnl,
            "total_trades": total,
            "buys": stats.get("buys", 0),
            "sells": stats.get("sells", 0),
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "buy_volume": stats.get("total_buy_volume", 0),
            "sell_volume": stats.get("total_sell_volume", 0),
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "pnl_by_symbol": pnl_by_symbol,
        }

    def print_summary(self) -> str:
        """
        Краткий отчёт после каждой итерации (для лога).

        Returns:
            Отформатированная строка
        """
        metrics = self.get_metrics(days=1)

        lines = [
            "--- Сводка за день ---",
            f"P&L: {metrics['total_pnl']:+.2f} USDT",
            f"Сделок: {metrics['total_trades']} (покупок: {metrics['buys']}, продаж: {metrics['sells']})",
            f"Win Rate: {metrics['win_rate']:.0f}%",
        ]

        if metrics["best_trade"]:
            bt = metrics["best_trade"]
            lines.append(f"Лучшая: {bt['symbol']} {bt['pnl']:+.2f} USDT")
        if metrics["worst_trade"]:
            wt = metrics["worst_trade"]
            lines.append(f"Худшая: {wt['symbol']} {wt['pnl']:+.2f} USDT")

        summary = "\n".join(lines)
        logger.info(summary)
        return summary

    def print_daily_report(self) -> str:
        """
        Подробный ежедневный отчёт.

        Returns:
            Отформатированная строка
        """
        metrics = self.get_metrics(days=1)
        weekly = self.get_metrics(days=7)

        lines = [
            "=" * 50,
            "ЕЖЕДНЕВНЫЙ ОТЧЁТ",
            "=" * 50,
            "",
            "--- Сегодня ---",
            f"P&L: {metrics['total_pnl']:+.2f} USDT",
            f"Сделок: {metrics['total_trades']}",
            f"Win Rate: {metrics['win_rate']:.0f}%",
            f"Объём покупок: {metrics['buy_volume']:.2f} USDT",
            f"Объём продаж: {metrics['sell_volume']:.2f} USDT",
            "",
            "--- За 7 дней ---",
            f"P&L: {weekly['total_pnl']:+.2f} USDT",
            f"Сделок: {weekly['total_trades']}",
            f"Win Rate: {weekly['win_rate']:.0f}%",
        ]

        # P&L по инструментам
        if weekly["pnl_by_symbol"]:
            lines.append("")
            lines.append("P&L по инструментам (7 дней):")
            for sym, data in weekly["pnl_by_symbol"].items():
                pnl = data.get("realized_pnl", 0)
                trades_count = data.get("trades", 0)
                sign = "+" if pnl >= 0 else ""
                lines.append(f"  {sym}: {sign}{pnl:.2f} USDT ({trades_count} сделок)")

        lines.append("=" * 50)

        report = "\n".join(lines)
        logger.info(report)
        return report

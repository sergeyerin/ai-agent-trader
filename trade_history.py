"""
Модуль истории сделок.
Хранит все сделки в SQLite и предоставляет статистику.
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from config import config

logger = logging.getLogger(__name__)


class TradeHistory:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.TRADES_DB_PATH
        self._init_db()

    def _init_db(self):
        """Создаёт таблицу trades если её нет."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        action TEXT NOT NULL,
                        price REAL NOT NULL,
                        quantity_usdt REAL NOT NULL,
                        quantity_crypto REAL,
                        reasoning TEXT,
                        order_result TEXT,
                        success INTEGER DEFAULT 1
                    )
                """)
                conn.commit()
            logger.info(f"База данных сделок инициализирована: {self.db_path}")
        except Exception as e:
            logger.error(f"Ошибка инициализации БД: {e}", exc_info=True)

    def record_trade(
        self,
        symbol: str,
        action: str,
        price: float,
        quantity_usdt: float,
        quantity_crypto: float = 0.0,
        reasoning: str = "",
        order_result: str = "",
        success: bool = True,
    ):
        """
        Записывает сделку в историю.

        Args:
            symbol: Торговая пара (BTCUSDT)
            action: buy / sell / hold
            price: Цена на момент сделки
            quantity_usdt: Сумма в USDT
            quantity_crypto: Количество криптовалюты
            reasoning: Обоснование от AI
            order_result: Результат ордера (JSON строка)
            success: Успешна ли сделка
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO trades
                        (timestamp, symbol, action, price, quantity_usdt,
                         quantity_crypto, reasoning, order_result, success)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now().isoformat(),
                        symbol,
                        action,
                        price,
                        quantity_usdt,
                        quantity_crypto,
                        reasoning,
                        order_result,
                        1 if success else 0,
                    ),
                )
                conn.commit()
            logger.info(f"Сделка записана: {action} {symbol} {quantity_usdt} USDT @ {price}")
        except Exception as e:
            logger.error(f"Ошибка записи сделки: {e}", exc_info=True)

    def get_recent_trades(self, limit: int = 10, symbol: Optional[str] = None) -> List[Dict]:
        """
        Возвращает последние N сделок.

        Args:
            limit: Максимальное количество
            symbol: Фильтр по символу (опционально)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                if symbol:
                    rows = conn.execute(
                        "SELECT * FROM trades WHERE symbol = ? ORDER BY id DESC LIMIT ?",
                        (symbol, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM trades ORDER BY id DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Ошибка чтения истории: {e}", exc_info=True)
            return []

    def get_performance_stats(self, days: int = 7) -> Dict:
        """
        Рассчитывает статистику торговли за период.

        Args:
            days: Количество дней для анализа

        Returns:
            Словарь со статистикой
        """
        try:
            since = (datetime.now() - timedelta(days=days)).isoformat()
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Все сделки за период (кроме hold)
                trades = conn.execute(
                    "SELECT * FROM trades WHERE timestamp >= ? AND action != 'hold' ORDER BY timestamp",
                    (since,),
                ).fetchall()

                if not trades:
                    return {
                        "total_trades": 0,
                        "period_days": days,
                        "message": "Нет сделок за период",
                    }

                total_trades = len(trades)
                buys = [t for t in trades if t["action"] == "buy"]
                sells = [t for t in trades if t["action"] == "sell"]
                successful = [t for t in trades if t["success"]]

                total_buy_volume = sum(t["quantity_usdt"] for t in buys)
                total_sell_volume = sum(t["quantity_usdt"] for t in sells)

                # Подсчёт P&L по парам buy-sell для каждого символа
                pnl_by_symbol = self._calculate_pnl_by_symbol(trades)

                return {
                    "period_days": days,
                    "total_trades": total_trades,
                    "buys": len(buys),
                    "sells": len(sells),
                    "success_rate": len(successful) / total_trades * 100 if total_trades else 0,
                    "total_buy_volume": total_buy_volume,
                    "total_sell_volume": total_sell_volume,
                    "pnl_by_symbol": pnl_by_symbol,
                }

        except Exception as e:
            logger.error(f"Ошибка расчёта статистики: {e}", exc_info=True)
            return {"error": str(e)}

    def _calculate_pnl_by_symbol(self, trades) -> Dict[str, Dict]:
        """
        Простой расчёт P&L: сравнивает среднюю цену покупки со средней ценой продажи.
        """
        by_symbol = {}
        for t in trades:
            sym = t["symbol"]
            if sym not in by_symbol:
                by_symbol[sym] = {"buys": [], "sells": []}
            if t["action"] == "buy":
                by_symbol[sym]["buys"].append(t)
            elif t["action"] == "sell":
                by_symbol[sym]["sells"].append(t)

        result = {}
        for sym, data in by_symbol.items():
            buy_vol = sum(t["quantity_usdt"] for t in data["buys"])
            sell_vol = sum(t["quantity_usdt"] for t in data["sells"])

            buy_crypto = sum(t["quantity_crypto"] for t in data["buys"] if t["quantity_crypto"])
            sell_crypto = sum(t["quantity_crypto"] for t in data["sells"] if t["quantity_crypto"])

            avg_buy_price = buy_vol / buy_crypto if buy_crypto > 0 else 0
            avg_sell_price = sell_vol / sell_crypto if sell_crypto > 0 else 0

            # Реализованный P&L = проданный объём * (ср.цена продажи - ср.цена покупки)
            realized_pnl = 0
            if avg_buy_price > 0 and sell_crypto > 0:
                realized_pnl = sell_crypto * (avg_sell_price - avg_buy_price)

            result[sym] = {
                "trades": len(data["buys"]) + len(data["sells"]),
                "buy_volume_usdt": buy_vol,
                "sell_volume_usdt": sell_vol,
                "avg_buy_price": avg_buy_price,
                "avg_sell_price": avg_sell_price,
                "realized_pnl": realized_pnl,
            }

        return result

    def get_daily_pnl(self) -> float:
        """Возвращает реализованный P&L за сегодня."""
        stats = self.get_performance_stats(days=1)
        pnl_by_symbol = stats.get("pnl_by_symbol", {})
        return sum(s.get("realized_pnl", 0) for s in pnl_by_symbol.values())

    def format_history_for_prompt(self, limit: int = 10) -> str:
        """
        Форматирует историю сделок для промпта AI.

        Args:
            limit: Количество последних сделок

        Returns:
            Отформатированная строка
        """
        trades = self.get_recent_trades(limit=limit)
        stats = self.get_performance_stats(days=7)

        lines = ["=== История сделок ==="]

        # Общая статистика
        if stats.get("total_trades", 0) > 0:
            lines.append(f"За последние {stats['period_days']} дней:")
            lines.append(
                f"  Сделок: {stats['total_trades']} "
                f"(покупок: {stats.get('buys', 0)}, продаж: {stats.get('sells', 0)})"
            )
            lines.append(f"  Объём покупок: {stats.get('total_buy_volume', 0):.2f} USDT")
            lines.append(f"  Объём продаж: {stats.get('total_sell_volume', 0):.2f} USDT")

            pnl_by_symbol = stats.get("pnl_by_symbol", {})
            if pnl_by_symbol:
                lines.append("  P&L по инструментам:")
                for sym, pnl_data in pnl_by_symbol.items():
                    pnl = pnl_data.get("realized_pnl", 0)
                    sign = "+" if pnl >= 0 else ""
                    lines.append(f"    {sym}: {sign}{pnl:.2f} USDT")
        else:
            lines.append("Сделок за последние 7 дней нет")

        # Последние сделки
        if trades:
            lines.append("")
            lines.append(f"Последние {len(trades)} сделок:")
            for t in trades:
                ts = t["timestamp"][:16]  # Обрезаем до минут
                action = t["action"].upper()
                symbol = t["symbol"]
                price = t["price"]
                usdt = t["quantity_usdt"]
                ok = "✓" if t["success"] else "✗"
                lines.append(f"  [{ts}] {ok} {action} {symbol} {usdt:.2f} USDT @ {price:.2f}")
                if t.get("reasoning"):
                    # Обрезаем reasoning до 100 символов
                    reason = t["reasoning"][:100]
                    lines.append(f"    Причина: {reason}")

        return "\n".join(lines)

"""
Модуль риск-менеджмента.
Stop-loss, take-profit, position sizing, проверка лимитов.
"""

import logging
from typing import Dict, List, Optional, Tuple
from config import config

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, trade_history, portfolio_manager):
        self.history = trade_history
        self.portfolio = portfolio_manager
        self.stop_loss_pct = config.STOP_LOSS_PCT
        self.take_profit_pct = config.TAKE_PROFIT_PCT
        self.max_daily_loss = config.MAX_DAILY_LOSS
        self.position_size_pct = config.POSITION_SIZE_PCT

    def check_stop_loss(self, symbol: str, current_price: float) -> Optional[Dict]:
        """
        Проверяет, нужно ли закрыть позицию по stop-loss.

        Ищет последнюю покупку этого символа и сравнивает с текущей ценой.

        Returns:
            Словарь с рекомендацией продажи или None если SL не сработал
        """
        entry = self._get_entry_price(symbol)
        if entry is None:
            return None

        entry_price = entry["price"]
        loss_pct = (current_price - entry_price) / entry_price * 100

        if loss_pct <= -self.stop_loss_pct:
            logger.warning(
                f"STOP-LOSS {symbol}: убыток {loss_pct:.2f}% "
                f"(порог: -{self.stop_loss_pct}%). "
                f"Entry: {entry_price:.2f}, Current: {current_price:.2f}"
            )
            return {
                "action": "sell",
                "reason": "stop_loss",
                "entry_price": entry_price,
                "current_price": current_price,
                "loss_pct": loss_pct,
                "quantity_usdt": entry.get("quantity_usdt", 0),
            }

        return None

    def check_take_profit(self, symbol: str, current_price: float) -> Optional[Dict]:
        """
        Проверяет, достигнута ли цель прибыли.

        Returns:
            Словарь с рекомендацией продажи или None если TP не достигнут
        """
        entry = self._get_entry_price(symbol)
        if entry is None:
            return None

        entry_price = entry["price"]
        profit_pct = (current_price - entry_price) / entry_price * 100

        if profit_pct >= self.take_profit_pct:
            logger.info(
                f"TAKE-PROFIT {symbol}: прибыль {profit_pct:.2f}% "
                f"(порог: +{self.take_profit_pct}%). "
                f"Entry: {entry_price:.2f}, Current: {current_price:.2f}"
            )
            return {
                "action": "sell",
                "reason": "take_profit",
                "entry_price": entry_price,
                "current_price": current_price,
                "profit_pct": profit_pct,
                "quantity_usdt": entry.get("quantity_usdt", 0),
            }

        return None

    def check_positions(self, current_prices: Dict[str, float]) -> List[Dict]:
        """
        Проверяет все открытые позиции на stop-loss и take-profit.

        Args:
            current_prices: {symbol: current_price}

        Returns:
            Список рекомендаций на закрытие позиций
        """
        actions = []
        positions = self.portfolio.get_all_positions()

        for coin, info in positions.items():
            symbol = f"{coin}USDT"
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]

            # Проверяем stop-loss
            sl = self.check_stop_loss(symbol, current_price)
            if sl:
                sl["symbol"] = symbol
                actions.append(sl)
                continue  # Не проверяем TP если сработал SL

            # Проверяем take-profit
            tp = self.check_take_profit(symbol, current_price)
            if tp:
                tp["symbol"] = symbol
                actions.append(tp)

        return actions

    def calculate_position_size(self, symbol: str = None) -> float:
        """
        Рассчитывает размер позиции как % от портфеля.
        Вместо фиксированного MAX_TRADE_AMOUNT.

        Returns:
            Рекомендуемая сумма сделки в USDT
        """
        equity = self.portfolio.total_equity
        if equity <= 0:
            return config.MAX_TRADE_AMOUNT

        size = equity * (self.position_size_pct / 100)

        # Не превышаем максимум
        size = min(size, config.MAX_TRADE_AMOUNT)
        # Минимум 1 USDT
        size = max(size, 1.0)

        logger.debug(
            f"Position size: {size:.2f} USDT "
            f"({self.position_size_pct}% от {equity:.2f} equity)"
        )
        return size

    def is_trading_allowed(self) -> Tuple[bool, str]:
        """
        Проверяет, разрешена ли торговля (дневной лимит убытков).

        Returns:
            (разрешено, причина)
        """
        daily_pnl = self.history.get_daily_pnl()
        if daily_pnl < -self.max_daily_loss:
            reason = (
                f"Дневной убыток {daily_pnl:.2f} USDT "
                f"превышает лимит {self.max_daily_loss} USDT"
            )
            return False, reason

        return True, ""

    def filter_by_confidence(self, recommendation: Dict) -> bool:
        """
        Фильтрует рекомендации по уровню уверенности AI.

        Returns:
            True если рекомендация проходит фильтр
        """
        if recommendation.get("action") == "hold":
            return True

        confidence = recommendation.get("confidence", 0)
        min_confidence = config.MIN_CONFIDENCE

        if confidence < min_confidence:
            logger.info(
                f"Рекомендация отклонена: confidence {confidence}% "
                f"< порог {min_confidence}%"
            )
            return False

        return True

    def _get_entry_price(self, symbol: str) -> Optional[Dict]:
        """
        Получает цену входа для символа — последняя нераспроданная покупка.
        """
        # Проверяем, есть ли вообще позиция
        pos = self.portfolio.get_position(symbol)
        if not pos:
            return None

        # Ищем последнюю покупку этого символа
        trades = self.history.get_recent_trades(limit=50, symbol=symbol)
        for trade in trades:
            if trade["action"] == "buy" and trade["success"]:
                return {
                    "price": trade["price"],
                    "quantity_usdt": trade["quantity_usdt"],
                    "timestamp": trade["timestamp"],
                }

        return None

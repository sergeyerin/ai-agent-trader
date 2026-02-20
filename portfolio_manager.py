"""
Модуль управления портфелем.
Отслеживает баланс, текущие позиции и unrealized P&L.
"""

import logging
from typing import Dict, Optional, List
from bybit_client import BybitClient

logger = logging.getLogger(__name__)


class PortfolioManager:
    def __init__(self, bybit_client: BybitClient):
        self.bybit = bybit_client
        self._balances: Dict[str, float] = {}
        self._total_equity: float = 0.0

    def refresh(self) -> bool:
        """
        Обновляет данные о балансе с Bybit.

        Returns:
            True если обновление успешно
        """
        try:
            response = self.bybit.get_account_balance()
            if not response or response.get("retCode") != 0:
                logger.error(f"Ошибка получения баланса: {response}")
                return False

            self._balances = {}
            self._total_equity = 0.0

            accounts = response.get("result", {}).get("list", [])
            for account in accounts:
                coins = account.get("coin", [])
                for coin_info in coins:
                    coin = coin_info.get("coin", "")
                    wallet_balance = float(coin_info.get("walletBalance", 0))
                    usd_value = float(coin_info.get("usdValue", 0))

                    if wallet_balance > 0:
                        self._balances[coin] = {
                            "balance": wallet_balance,
                            "usd_value": usd_value,
                        }
                        self._total_equity += usd_value

                total_equity = float(account.get("totalEquity", 0))
                if total_equity > 0:
                    self._total_equity = total_equity

            logger.info(
                f"Портфель обновлён. Активов: {len(self._balances)}, "
                f"Общая стоимость: {self._total_equity:.2f} USD"
            )
            return True

        except Exception as e:
            logger.error(f"Ошибка обновления портфеля: {e}", exc_info=True)
            return False

    @property
    def total_equity(self) -> float:
        return self._total_equity

    @property
    def usdt_balance(self) -> float:
        return self._balances.get("USDT", {}).get("balance", 0.0)

    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Получает текущую позицию по символу.

        Args:
            symbol: Символ (например 'BTC', 'SOL', не 'BTCUSDT')

        Returns:
            Словарь с balance и usd_value или None
        """
        coin = symbol.replace("USDT", "")
        pos = self._balances.get(coin)
        if pos and pos["balance"] > 0.001:  # Игнорируем пылевые балансы
            return pos
        return None

    def has_sufficient_funds(self, amount_usdt: float) -> bool:
        """Проверяет, хватает ли USDT для покупки."""
        available = self.usdt_balance
        # Оставляем 1 USDT буфер для комиссий
        return available >= amount_usdt + 1.0

    def can_sell(self, symbol: str, quantity_usdt: float, current_price: float) -> bool:
        """
        Проверяет, есть ли достаточно монет для продажи.

        Args:
            symbol: Полный символ (BTCUSDT)
            quantity_usdt: Сумма продажи в USDT
            current_price: Текущая цена
        """
        pos = self.get_position(symbol)
        if not pos:
            return False
        required_qty = quantity_usdt / current_price
        return pos["balance"] >= required_qty

    def get_all_positions(self) -> Dict[str, Dict]:
        """Возвращает все ненулевые позиции (исключая USDT)."""
        return {
            coin: info
            for coin, info in self._balances.items()
            if coin != "USDT" and info["balance"] > 0.001
        }

    def format_portfolio_for_prompt(self, prices: Optional[Dict[str, float]] = None) -> str:
        """
        Форматирует состояние портфеля для промпта AI.

        Args:
            prices: Словарь текущих цен {symbol: price} для расчёта P&L

        Returns:
            Отформатированная строка
        """
        lines = ["=== Текущий портфель ==="]
        lines.append(f"Общая стоимость портфеля: {self._total_equity:.2f} USD")
        lines.append(f"Свободные USDT: {self.usdt_balance:.2f}")
        lines.append("")

        positions = self.get_all_positions()
        if positions:
            lines.append("Открытые позиции:")
            for coin, info in positions.items():
                line = f"  {coin}: {info['balance']:.6f} (≈{info['usd_value']:.2f} USD)"
                lines.append(line)
        else:
            lines.append("Открытых позиций нет (только USDT)")

        return "\n".join(lines)

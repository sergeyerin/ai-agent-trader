from datetime import datetime
from typing import Optional, Dict
import logging

from bybit_client import BybitClient
from polymarket_client import PolymarketClient
from deepseek_client import DeepSeekClient
from config import config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingAgent:
    def __init__(self):
        self.bybit = BybitClient()
        self.polymarket = PolymarketClient()
        self.deepseek = DeepSeekClient()
        self.trading_pair = config.TRADING_PAIR
        self.max_trading_volume = config.MAX_TRADING_VOLUME
        self.max_trade_amount = config.MAX_TRADE_AMOUNT
        
        logger.info(f"Торговый агент инициализирован. Пара: {self.trading_pair}")
        logger.info(f"Максимальный объем торговли: {self.max_trading_volume} USDT")
        logger.info(f"Максимальная сумма сделки: {self.max_trade_amount} USDT")
    
    def collect_market_data(self) -> Optional[Dict]:
        """
        Сбор всех необходимых рыночных данных.
        
        Returns:
            Словарь с данными или None в случае ошибки
        """
        logger.info("Начало сбора рыночных данных...")
        
        try:
            # Получаем исторические данные
            logger.info("Получение данных BTC/USDT...")
            btc_data = self.bybit.get_historical_data(
                config.BTC_SYMBOL,
                config.HISTORICAL_MONTHS
            )
            
            logger.info("Получение данных золота XAUT/USDT...")
            gold_data = self.bybit.get_historical_data(
                config.GOLD_SYMBOL,
                config.HISTORICAL_MONTHS
            )
            
            logger.info("Получение данных серебра XAGUSD...")
            silver_data = self.bybit.get_historical_data(
                config.SILVER_SYMBOL,
                config.HISTORICAL_MONTHS
            )
            
            # Получаем данные Polymarket
            logger.info("Получение данных Polymarket...")
            polymarket_info = self.polymarket.get_formatted_data()
            
            # Получаем текущую цену
            current_price = self.bybit.get_current_price(self.trading_pair)
            
            if current_price is None:
                logger.error("Не удалось получить текущую цену")
                return None
            
            logger.info(f"Текущая цена {self.trading_pair}: {current_price} USDT")
            
            return {
                "btc_data": btc_data,
                "gold_data": gold_data,
                "silver_data": silver_data,
                "polymarket_info": polymarket_info,
                "current_price": current_price
            }
            
        except Exception as e:
            logger.error(f"Ошибка при сборе рыночных данных: {e}")
            return None
    
    def get_trading_decision(self, market_data: Dict) -> Optional[Dict]:
        """
        Получение торгового решения от DeepSeek.
        
        Args:
            market_data: Собранные рыночные данные
        
        Returns:
            Рекомендация по торговле или None
        """
        logger.info("Подготовка данных для DeepSeek...")
        
        # Подготавливаем данные
        formatted_data = self.deepseek.prepare_market_data(
            market_data["btc_data"],
            market_data["gold_data"],
            market_data["silver_data"],
            market_data["polymarket_info"]
        )
        
        logger.info("Отправка запроса в DeepSeek...")
        
        # Получаем рекомендацию
        recommendation = self.deepseek.get_trading_recommendation(
            formatted_data,
            market_data["current_price"],
            self.max_trade_amount
        )
        
        if recommendation:
            logger.info(f"Получена рекомендация: {recommendation}")
        else:
            logger.error("Не удалось получить рекомендацию от DeepSeek")
        
        return recommendation
    
    def execute_trade(self, recommendation: Dict, current_price: float) -> bool:
        """
        Выполнение торговой операции на основе рекомендации.
        
        Args:
            recommendation: Рекомендация от DeepSeek
            current_price: Текущая цена
        
        Returns:
            True если операция выполнена успешно, False иначе
        """
        action = recommendation.get("action")
        
        if action == "hold":
            logger.info("Рекомендация: не проводить сделку")
            logger.info(f"Причина: {recommendation.get('reasoning', 'Не указана')}")
            return True
        
        # Проверяем ценовой диапазон
        price_from = recommendation.get("price_from")
        price_to = recommendation.get("price_to")
        quantity_usdt = recommendation.get("quantity_usdt")
        
        if price_from and price_to:
            if not (price_from <= current_price <= price_to):
                logger.info(
                    f"Текущая цена {current_price} вне рекомендованного диапазона "
                    f"[{price_from}, {price_to}]. Сделка не будет выполнена."
                )
                return False
        
        # Проверяем количество
        if not quantity_usdt or quantity_usdt <= 0:
            logger.error("Некорректная сумма сделки")
            return False
        
        if quantity_usdt > self.max_trade_amount:
            logger.warning(
                f"Рекомендованная сумма {quantity_usdt} USDT превышает максимум. "
                f"Используем {self.max_trade_amount} USDT"
            )
            quantity_usdt = self.max_trade_amount
        
        # Рассчитываем количество BTC
        if action == "buy":
            qty_btc = quantity_usdt / current_price
            side = "Buy"
            logger.info(
                f"Выполнение покупки: {qty_btc:.6f} BTC на сумму {quantity_usdt} USDT "
                f"по цене ~{current_price} USDT"
            )
        elif action == "sell":
            qty_btc = quantity_usdt / current_price
            side = "Sell"
            logger.info(
                f"Выполнение продажи: {qty_btc:.6f} BTC на сумму {quantity_usdt} USDT "
                f"по цене ~{current_price} USDT"
            )
        else:
            logger.error(f"Неизвестное действие: {action}")
            return False
        
        logger.info(f"Причина: {recommendation.get('reasoning', 'Не указана')}")
        
        # Размещаем рыночный ордер
        result = self.bybit.place_order(
            symbol=self.trading_pair,
            side=side,
            qty=qty_btc,
            order_type="Market"
        )
        
        if result.get("retCode") == 0:
            logger.info(f"Ордер успешно размещен: {result}")
            return True
        else:
            logger.error(f"Ошибка при размещении ордера: {result}")
            return False
    
    def run(self):
        """
        Основной цикл работы агента: сбор данных, анализ, торговля.
        """
        logger.info("=" * 60)
        logger.info(f"Запуск торгового агента: {datetime.now()}")
        logger.info("=" * 60)
        
        # 1. Собираем данные
        market_data = self.collect_market_data()
        if not market_data:
            logger.error("Не удалось собрать рыночные данные. Пропускаем итерацию.")
            return
        
        # 2. Получаем торговое решение
        recommendation = self.get_trading_decision(market_data)
        if not recommendation:
            logger.error("Не удалось получить торговую рекомендацию. Пропускаем итерацию.")
            return
        
        # 3. Выполняем торговую операцию
        success = self.execute_trade(recommendation, market_data["current_price"])
        
        if success:
            logger.info("Итерация завершена успешно")
        else:
            logger.warning("Итерация завершена с предупреждениями")
        
        logger.info("=" * 60)

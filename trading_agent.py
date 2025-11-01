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
        self.trading_pairs = config.TRADING_PAIRS
        self.max_trading_volume = config.MAX_TRADING_VOLUME
        self.max_trade_amount = config.MAX_TRADE_AMOUNT
        
        logger.info(f"Торговый агент инициализирован. Инструменты: {', '.join(self.trading_pairs)}")
        logger.info(f"Максимальный объем торговли: {self.max_trading_volume} USDT")
        logger.info(f"Максимальная сумма сделки: {self.max_trade_amount} USDT")
    
    def collect_market_data(self) -> Optional[Dict]:
        """
        Сбор всех необходимых рыночных данных для всех торговых инструментов.
        
        Returns:
            Словарь с данными или None в случае ошибки
        """
        logger.info("Начало сбора рыночных данных...")
        
        try:
            # Получаем данные для всех торговых пар
            trading_pairs_data = {}
            for pair in self.trading_pairs:
                logger.info(f"Получение данных {pair}...")
                pair_data = self.bybit.get_historical_data(
                    pair,
                    config.HISTORICAL_DAYS
                )
                current_price = self.bybit.get_current_price(pair)
                
                if current_price is None:
                    logger.error(f"Не удалось получить текущую цену для {pair}")
                    continue
                
                trading_pairs_data[pair] = {
                    "data": pair_data,
                    "current_price": current_price
                }
                logger.info(f"Текущая цена {pair}: {current_price} USDT")
            
            if not trading_pairs_data:
                logger.error("Не удалось получить данные ни для одного инструмента")
                return None
            
            logger.info("Получение данных золота XAUT/USDT...")
            gold_data = self.bybit.get_historical_data(
                config.GOLD_SYMBOL,
                config.HISTORICAL_DAYS
            )
            
            logger.info("Получение данных серебра XAGUSD...")
            silver_data = self.bybit.get_historical_data(
                config.SILVER_SYMBOL,
                config.HISTORICAL_DAYS
            )
            
            # Получаем данные Polymarket
            logger.info("Получение данных Polymarket...")
            polymarket_info = self.polymarket.get_formatted_data()
            
            return {
                "trading_pairs_data": trading_pairs_data,
                "gold_data": gold_data,
                "silver_data": silver_data,
                "polymarket_info": polymarket_info
            }
            
        except Exception as e:
            logger.error(f"Ошибка при сборе рыночных данных: {e}")
            return None
    
    def get_trading_decisions(self, market_data: Dict) -> Dict[str, Optional[Dict]]:
        """
        Получение торговых решений от DeepSeek для всех инструментов.
        
        Args:
            market_data: Собранные рыночные данные
        
        Returns:
            Словарь с рекомендациями для каждого инструмента
        """
        logger.info("Подготовка данных для DeepSeek...")
        
        # Создаем графики для всех доступных инструментов (исключая SOL)
        logger.info("Создание графиков для анализа...")
        chart_images = {}
        
        # Создаем графики для всех инструментов кроме SOLUSDT
        for symbol in market_data["trading_pairs_data"].keys():
            if symbol != "SOLUSDT":  # Исключаем основную торговую пару
                pair_data = market_data["trading_pairs_data"][symbol]["data"]
                if not pair_data.empty:
                    logger.info(f"Создание графика для {symbol}...")
                    try:
                        chart_base64 = self.deepseek.create_chart(
                            pair_data,
                            symbol,
                            symbol.replace("USDT", "/USDT")
                        )
                        if chart_base64:
                            chart_images[symbol] = chart_base64
                            logger.info(f"График для {symbol} успешно создан")
                    except Exception as e:
                        logger.error(f"Ошибка при создании графика для {symbol}: {e}")
        
        logger.info(f"Создано {len(chart_images)} графиков")
        
        recommendations = {}
        
        # Получаем рекомендации для каждой торговой пары
        for pair, pair_info in market_data["trading_pairs_data"].items():
            logger.info(f"Получение рекомендации для {pair}...")
            
            # Подготавливаем данные
            formatted_data = self.deepseek.prepare_market_data(
                pair_info["data"],
                pair,
                market_data["gold_data"],
                market_data["silver_data"],
                market_data["polymarket_info"]
            )
            
            # Получаем рекомендацию (с графиками если есть)
            recommendation = self.deepseek.get_trading_recommendation(
                formatted_data,
                pair,
                pair_info["current_price"],
                self.max_trade_amount,
                chart_images=chart_images
            )
            
            if recommendation:
                logger.info(f"Получена рекомендация для {pair}: {recommendation.get('action')}")
                recommendations[pair] = recommendation
            else:
                logger.error(f"Не удалось получить рекомендацию для {pair}")
                recommendations[pair] = None
        
        return recommendations
    
    def execute_trade(self, symbol: str, recommendation: Dict, current_price: float) -> bool:
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
        
        # Рассчитываем количество криптовалюты
        base_currency = symbol.replace("USDT", "")  # Извлекаем базовую валюту
        if action == "buy":
            qty_crypto = quantity_usdt / current_price
            side = "Buy"
            logger.info(
                f"Выполнение покупки: {qty_crypto:.6f} {base_currency} на сумму {quantity_usdt} USDT "
                f"по цене ~{current_price} USDT"
            )
        elif action == "sell":
            qty_crypto = quantity_usdt / current_price
            side = "Sell"
            logger.info(
                f"Выполнение продажи: {qty_crypto:.6f} {base_currency} на сумму {quantity_usdt} USDT "
                f"по цене ~{current_price} USDT"
            )
        else:
            logger.error(f"Неизвестное действие: {action}")
            return False
        
        logger.info(f"Причина: {recommendation.get('reasoning', 'Не указана')}")
        
        # Размещаем рыночный ордер
        result = self.bybit.place_order(
            symbol=symbol,
            side=side,
            qty=qty_crypto,
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
        
        # 2. Получаем торговые решения для всех инструментов
        recommendations = self.get_trading_decisions(market_data)
        if not recommendations:
            logger.error("Не удалось получить торговые рекомендации. Пропускаем итерацию.")
            return
        
        # 3. Выполняем торговые операции для каждого инструмента
        results = {}
        for symbol, recommendation in recommendations.items():
            if recommendation is None:
                logger.warning(f"Пропускаем {symbol} - нет рекомендации")
                results[symbol] = False
                continue
            
            logger.info(f"\n{'='*40}")
            logger.info(f"Обработка {symbol}")
            logger.info(f"{'='*40}")
            
            current_price = market_data["trading_pairs_data"][symbol]["current_price"]
            success = self.execute_trade(symbol, recommendation, current_price)
            results[symbol] = success
        
        # Итоговая статистика
        successful = sum(1 for s in results.values() if s)
        total = len(results)
        logger.info(f"\n{'='*60}")
        logger.info(f"Итерация завершена: {successful}/{total} успешных операций")
        logger.info("=" * 60)

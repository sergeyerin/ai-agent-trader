from datetime import datetime
from typing import Optional, Dict
import logging
import pandas as pd

from bybit_client import BybitClient
from polymarket_client import PolymarketClient
from deepseek_client import DeepSeekClient
from goldpricez_client import GoldPricezClient
from portfolio_manager import PortfolioManager
from trade_history import TradeHistory
from indicators import compute_all_indicators, format_indicators_for_prompt
from config import config


# Устанавливаем уровень логирования из config
log_level = getattr(logging, config.LOG_LEVEL, logging.INFO)

# Настройка логирования с несколькими обработчиками
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Вывод в консоль
        logging.FileHandler('trading_agent.log', encoding='utf-8')  # Основной лог-файл
    ]
)

# Добавляем отдельный файл для DEBUG логов
if log_level == logging.DEBUG:
    debug_handler = logging.FileHandler('debug.log', encoding='utf-8')
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(debug_handler)
    logging.info("DEBUG логи будут записываться в debug.log")

# Отключаем DEBUG логи для шумных библиотек
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.INFO)
logging.getLogger('PIL').setLevel(logging.INFO)

logger = logging.getLogger(__name__)


class TradingAgent:
    def __init__(self):
        self.bybit = BybitClient()
        self.polymarket = PolymarketClient()
        self.deepseek = DeepSeekClient()
        self.goldpricez = GoldPricezClient()
        self.portfolio = PortfolioManager(self.bybit)
        self.history = TradeHistory()
        self.trading_pairs = config.TRADING_PAIRS
        self.max_trading_volume = config.MAX_TRADING_VOLUME
        self.max_trade_amount = config.MAX_TRADE_AMOUNT
        
        logger.info(f"Торговый агент инициализирован. Инструменты: {', '.join(self.trading_pairs)}")
        logger.info(f"Максимальный объем торговли: {self.max_trading_volume} USDT")
        logger.info(f"Максимальная сумма сделки: {self.max_trade_amount} USDT")
        logger.info(f"AI анализ: {'ENABLED' if config.ENABLE_AI_ANALYSIS else 'DISABLED'}")
        logger.info(f"Реальная торговля: {'ENABLED' if config.ENABLE_TRADING else 'DISABLED'}")
    
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
            
            # Золото - используем уже полученные данные если есть
            if config.GOLD_SYMBOL in trading_pairs_data:
                logger.info(f"Используются уже полученные данные для {config.GOLD_SYMBOL}")
                gold_data = trading_pairs_data[config.GOLD_SYMBOL]["data"]
            else:
                logger.info("Получение данных золота XAUT/USDT...")
                gold_data = self.bybit.get_historical_data(
                    config.GOLD_SYMBOL,
                    config.HISTORICAL_DAYS
                )
            
            # Серебро - получаем с goldpricez.com
            logger.info("Получение данных по серебру (XAG) с goldpricez.com...")
            silver_data = self.goldpricez.get_silver_historical_data(config.HISTORICAL_DAYS)
            
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
    
    def create_charts(self, market_data: Dict) -> Dict[str, str]:
        """
        Создание графиков для всех инструментов.
        
        Args:
            market_data: Собранные рыночные данные
        
        Returns:
            Словарь с base64-кодированными графиками
        """
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
        return chart_images
    
    def compute_indicators(self, market_data: Dict) -> Dict[str, str]:
        """
        Вычисляет технические индикаторы для всех торговых пар.
        
        Returns:
            Словарь {symbol: форматированный текст индикаторов}
        """
        indicators_texts = {}
        for symbol, pair_info in market_data["trading_pairs_data"].items():
            df = pair_info["data"]
            indicators = compute_all_indicators(df)
            indicators_texts[symbol] = format_indicators_for_prompt(indicators, symbol)
        return indicators_texts
    
    def get_trading_decisions(
        self,
        market_data: Dict,
        chart_images: Optional[Dict[str, str]] = None,
        indicators_texts: Optional[Dict[str, str]] = None,
        portfolio_text: str = "",
        history_text: str = "",
    ) -> Dict[str, Optional[Dict]]:
        """
        Получение торговых решений от DeepSeek для всех инструментов.
        
        Args:
            market_data: Собранные рыночные данные
            chart_images: Словарь с графиками (optional)
            indicators_texts: Словарь с индикаторами по символам
            portfolio_text: Текст портфеля для промпта
            history_text: Текст истории сделок для промпта
        
        Returns:
            Словарь с рекомендациями для каждого инструмента
        """
        if not config.ENABLE_AI_ANALYSIS:
            logger.warning("Анализ AI отключен. Возвращаем пустые рекомендации.")
            return {pair: {"action": "hold", "reasoning": "AI analysis disabled"} 
                    for pair in market_data["trading_pairs_data"].keys()}
        
        logger.info("Подготовка данных для DeepSeek...")
        
        if chart_images is None:
            chart_images = {}
        if indicators_texts is None:
            indicators_texts = {}
        
        recommendations = {}
        
        for pair, pair_info in market_data["trading_pairs_data"].items():
            logger.info(f"Получение рекомендации для {pair}...")
            
            formatted_data = self.deepseek.prepare_market_data(
                pair_info["data"],
                pair,
                market_data["gold_data"],
                market_data["silver_data"],
                market_data["polymarket_info"],
                indicators_text=indicators_texts.get(pair, ""),
                portfolio_text=portfolio_text,
                history_text=history_text,
            )
            
            recommendation = self.deepseek.get_trading_recommendation(
                formatted_data,
                pair,
                pair_info["current_price"],
                self.max_trade_amount,
                chart_images=chart_images
            )
            
            if recommendation:
                confidence = recommendation.get("confidence", 0)
                logger.info(
                    f"Рекомендация для {pair}: {recommendation.get('action')} "
                    f"(уверенность: {confidence}%)"
                )
                recommendations[pair] = recommendation
            else:
                logger.error(f"Не удалось получить рекомендацию для {pair}")
                recommendations[pair] = None
        
        return recommendations
    
    def execute_trade(self, symbol: str, recommendation: Dict, current_price: float) -> bool:
        """
        Выполнение торговой операции на основе рекомендации.
        Проверяет баланс, записывает сделку в историю.
        
        Args:
            symbol: Торговая пара
            recommendation: Рекомендация от DeepSeek
            current_price: Текущая цена
        
        Returns:
            True если операция выполнена успешно, False иначе
        """
        action = recommendation.get("action")
        reasoning = recommendation.get("reasoning", "Не указана")
        
        if action == "hold":
            logger.info("Рекомендация: не проводить сделку")
            logger.info(f"Причина: {reasoning}")
            # Записываем hold в историю
            self.history.record_trade(
                symbol=symbol, action="hold", price=current_price,
                quantity_usdt=0, reasoning=reasoning
            )
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
        
        # Проверяем дневной лимит убытков
        daily_pnl = self.history.get_daily_pnl()
        if daily_pnl < -config.MAX_DAILY_LOSS:
            logger.warning(
                f"Дневной убыток {daily_pnl:.2f} USDT превышает лимит "
                f"{config.MAX_DAILY_LOSS} USDT. Торговля приостановлена."
            )
            return False
        
        # Проверяем баланс
        base_currency = symbol.replace("USDT", "")
        qty_crypto = quantity_usdt / current_price
        
        if action == "buy":
            if not self.portfolio.has_sufficient_funds(quantity_usdt):
                logger.warning(
                    f"Недостаточно USDT для покупки {symbol}. "
                    f"Нужно: {quantity_usdt}, Доступно: {self.portfolio.usdt_balance:.2f}"
                )
                return False
            side = "Buy"
            logger.info(
                f"Выполнение покупки: {qty_crypto:.6f} {base_currency} на сумму {quantity_usdt} USDT "
                f"по цене ~{current_price} USDT"
            )
        elif action == "sell":
            if not self.portfolio.can_sell(symbol, quantity_usdt, current_price):
                logger.warning(
                    f"Недостаточно {base_currency} для продажи. "
                    f"Нужно: {qty_crypto:.6f}, Позиция: {self.portfolio.get_position(symbol)}"
                )
                return False
            side = "Sell"
            logger.info(
                f"Выполнение продажи: {qty_crypto:.6f} {base_currency} на сумму {quantity_usdt} USDT "
                f"по цене ~{current_price} USDT"
            )
        else:
            logger.error(f"Неизвестное действие: {action}")
            return False
        
        logger.info(f"Причина: {reasoning}")
        
        # Проверяем флаг торговли
        if not config.ENABLE_TRADING:
            logger.warning("Реальная торговля отключена. Ордер НЕ был отправлен в Bybit.")
            self.history.record_trade(
                symbol=symbol, action=action, price=current_price,
                quantity_usdt=quantity_usdt, quantity_crypto=qty_crypto,
                reasoning=reasoning, order_result="TRADING_DISABLED", success=True
            )
            return True
        
        # Размещаем рыночный ордер
        result = self.bybit.place_order(
            symbol=symbol,
            side=side,
            qty=qty_crypto,
            order_type="Market"
        )
        
        success = result.get("retCode") == 0
        self.history.record_trade(
            symbol=symbol, action=action, price=current_price,
            quantity_usdt=quantity_usdt, quantity_crypto=qty_crypto,
            reasoning=reasoning, order_result=str(result), success=success
        )
        
        if success:
            logger.info(f"Ордер успешно размещен: {result}")
        else:
            logger.error(f"Ошибка при размещении ордера: {result}")
        
        return success
    
    def run(self):
        """
        Основной цикл работы агента: сбор данных, анализ, торговля.
        """
        logger.info("=" * 60)
        logger.info(f"Запуск торгового агента: {datetime.now()}")
        logger.info("=" * 60)
        
        # 1. Обновляем портфель
        logger.info("Обновление портфеля...")
        self.portfolio.refresh()
        portfolio_text = self.portfolio.format_portfolio_for_prompt()
        logger.info(f"Баланс USDT: {self.portfolio.usdt_balance:.2f}")
        
        # 2. Получаем историю сделок
        history_text = self.history.format_history_for_prompt()
        
        # 3. Проверяем дневной лимит убытков
        daily_pnl = self.history.get_daily_pnl()
        if daily_pnl < -config.MAX_DAILY_LOSS:
            logger.warning(
                f"Дневной убыток {daily_pnl:.2f} USDT превышает лимит "
                f"{config.MAX_DAILY_LOSS} USDT. Пропускаем итерацию."
            )
            return
        
        # 4. Собираем рыночные данные
        market_data = self.collect_market_data()
        if not market_data:
            logger.error("Не удалось собрать рыночные данные. Пропускаем итерацию.")
            return
        
        # 5. Вычисляем технические индикаторы
        indicators_texts = self.compute_indicators(market_data)
        
        # 6. Создаём графики
        chart_images = self.create_charts(market_data)
        
        # 7. Получаем торговые решения с учётом индикаторов, портфеля и истории
        recommendations = self.get_trading_decisions(
            market_data, chart_images,
            indicators_texts=indicators_texts,
            portfolio_text=portfolio_text,
            history_text=history_text,
        )
        if not recommendations:
            logger.error("Не удалось получить торговые рекомендации. Пропускаем итерацию.")
            return
        
        # 8. Выполняем торговые операции
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
        
        # 9. Итоговая статистика
        successful = sum(1 for s in results.values() if s)
        total = len(results)
        logger.info(f"\n{'='*60}")
        logger.info(f"Итерация завершена: {successful}/{total} успешных операций")
        logger.info(f"Баланс USDT: {self.portfolio.usdt_balance:.2f}")
        logger.info(f"Дневной P&L: {daily_pnl:+.2f} USDT")
        logger.info("=" * 60)

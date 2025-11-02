from pybit.unified_trading import HTTP
from datetime import datetime, timedelta
import time
import pandas as pd
import logging
from typing import List, Dict, Optional
from config import config

logger = logging.getLogger(__name__)


class BybitClient:
    def __init__(self):
        self.client = HTTP(
            testnet=config.BYBIT_TESTNET,
            api_key=config.BYBIT_API_KEY,
            api_secret=config.BYBIT_API_SECRET,
            timeout=30,  # Добавляем таймаут 30 секунд
        )
    
    def get_klines(self, symbol: str, interval: str, start_time: int, end_time: int, max_requests: int = 50) -> List[Dict]:
        """
        Получение исторических данных (свечей) для указанного символа.
        
        Args:
            symbol: Торговая пара (например, BTCUSDT)
            interval: Интервал свечи (5 = 5 минут)
            start_time: Время начала в миллисекундах
            end_time: Время окончания в миллисекундах
            max_requests: Максимальное количество запросов (защита от бесконечного цикла)
        
        Returns:
            Список свечей
        """
        all_klines = []
        current_start = start_time
        request_count = 0
        seen_timestamps = set()  # Для отслеживания дубликатов
        
        logger.info(f"Начало загрузки данных для {symbol}...")
        
        # Bybit ограничивает количество записей за запрос, поэтому делаем пагинацию
        while current_start < end_time and request_count < max_requests:
            try:
                request_count += 1
                logger.debug(f"Запрос {request_count}/{max_requests} для {symbol}")
                
                response = self.client.get_kline(
                    category="spot",
                    symbol=symbol,
                    interval=interval,
                    start=current_start,
                    end=end_time,
                    limit=200  # Уменьшаем лимит для более быстрых запросов
                )
                
                if response["retCode"] == 0 and response["result"]["list"]:
                    klines = response["result"]["list"]
                    
                    # Bybit возвращает данные в обратном порядке (от новых к старым)
                    # Поэтому первый элемент - самый новый, последний - самый старый
                    
                    # Фильтруем дубликаты
                    new_klines = []
                    for kline in klines:
                        timestamp = int(kline[0])
                        if timestamp not in seen_timestamps:
                            seen_timestamps.add(timestamp)
                            new_klines.append(kline)
                    
                    # Если нет новых данных, выходим из цикла
                    if not new_klines:
                        logger.info(f"Получены только дубликаты. Загрузка завершена. Всего свечей: {len(all_klines)}")
                        break
                    
                    all_klines.extend(new_klines)
                    logger.debug(f"Получено {len(new_klines)} новых свечей для {symbol}, всего: {len(all_klines)}")
                    
                    # Получаем самую старую временную метку (последний элемент)
                    oldest_timestamp = int(klines[-1][0])
                    
                    # Если получили меньше лимита, значит данных больше нет
                    if len(klines) < 200:
                        logger.info(f"Загрузка данных для {symbol} завершена. Всего свечей: {len(all_klines)}")
                        break
                    
                    # Проверяем, что самая старая свеча старше текущего start
                    if oldest_timestamp <= current_start:
                        logger.info(f"Достигнут начальный timestamp ({oldest_timestamp}), загрузка завершена")
                        break
                    
                    # Следующий запрос - до самой старой полученной свечи
                    # Обновляем end_time вместо current_start
                    end_time = oldest_timestamp - 1
                    time.sleep(0.2)  # Увеличиваем задержку для избежания rate limit
                else:
                    logger.warning(f"Пустой ответ от Bybit для {symbol}: {response.get('retMsg', 'Unknown error')}")
                    break
                    
            except Exception as e:
                logger.error(f"Ошибка при получении данных для {symbol}: {e}")
                break
        
        if request_count >= max_requests:
            logger.warning(f"Достигнут лимит запросов ({max_requests}) для {symbol}")
        
        return all_klines
    
    def get_historical_data(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """
        Получение исторических данных за указанное количество дней.
        
        Args:
            symbol: Торговая пара
            days: Количество дней истории (по умолчанию 7 для быстрой работы)
        
        Returns:
            DataFrame с историческими данными
        """
        logger.info(f"Запрос данных для {symbol} за последние {days} дней")
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        klines = self.get_klines(symbol, config.KLINE_INTERVAL, start_time, end_time)
        
        if not klines:
            logger.warning(f"Не получено данных для {symbol}")
            return pd.DataFrame()
        
        logger.info(f"Получено {len(klines)} свечей для {symbol}")
        
        # Преобразуем в DataFrame
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        
        # Конвертируем типы данных
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = df[col].astype(float)
        
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Проверяем полноту данных
        if not df.empty:
            # Вычисляем ожидаемое количество свечей
            interval_minutes = int(config.KLINE_INTERVAL)
            expected_candles = (days * 24 * 60) // interval_minutes
            
            actual_candles = len(df)
            coverage_percent = (actual_candles / expected_candles) * 100
            
            # Проверяем временной диапазон
            actual_start = df["timestamp"].iloc[0]
            actual_end = df["timestamp"].iloc[-1]
            requested_start = datetime.fromtimestamp(start_time / 1000)
            requested_end = datetime.fromtimestamp(end_time / 1000)
            
            time_diff_start = (requested_start - actual_start).total_seconds() / 3600  # в часах
            
            if coverage_percent < 90:
                logger.warning(
                    f"Неполные данные для {symbol}: получено {actual_candles}/{expected_candles} свечей ({coverage_percent:.1f}%)"
                )
            
            if abs(time_diff_start) > 1:  # Больше 1 часа разницы
                logger.warning(
                    f"Данные для {symbol} начинаются с {actual_start.strftime('%Y-%m-%d %H:%M')}, "
                    f"запрошено с {requested_start.strftime('%Y-%m-%d %H:%M')} "
                    f"(разница: {abs(time_diff_start):.1f} часов)"
                )
            
            logger.info(
                f"Период данных для {symbol}: {actual_start.strftime('%Y-%m-%d %H:%M')} - {actual_end.strftime('%Y-%m-%d %H:%M')}"
            )
        
        return df
    
    def get_account_balance(self) -> Dict:
        """
        Получение баланса аккаунта.
        
        Returns:
            Словарь с информацией о балансе
        """
        try:
            response = self.client.get_wallet_balance(accountType="UNIFIED")
            return response
        except Exception as e:
            print(f"Ошибка при получении баланса: {e}")
            return {}
    
    def place_order(
        self, 
        symbol: str, 
        side: str, 
        qty: float, 
        price: Optional[float] = None,
        order_type: str = "Market"
    ) -> Dict:
        """
        Размещение ордера на покупку или продажу.
        
        Args:
            symbol: Торговая пара
            side: Buy или Sell
            qty: Количество
            price: Цена (для лимитного ордера)
            order_type: Тип ордера (Market или Limit)
        
        Returns:
            Результат размещения ордера
        """
        try:
            params = {
                "category": "spot",
                "symbol": symbol,
                "side": side,
                "orderType": order_type,
                "qty": str(qty),
            }
            
            if order_type == "Limit" and price:
                params["price"] = str(price)
            
            response = self.client.place_order(**params)
            return response
        except Exception as e:
            print(f"Ошибка при размещении ордера: {e}")
            return {"retCode": -1, "retMsg": str(e)}
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Получение текущей цены для символа.
        
        Args:
            symbol: Торговая пара
        
        Returns:
            Текущая цена или None
        """
        try:
            response = self.client.get_tickers(category="spot", symbol=symbol)
            if response["retCode"] == 0 and response["result"]["list"]:
                return float(response["result"]["list"][0]["lastPrice"])
        except Exception as e:
            print(f"Ошибка при получении цены для {symbol}: {e}")
        return None

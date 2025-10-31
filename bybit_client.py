from pybit.unified_trading import HTTP
from datetime import datetime, timedelta
import time
import pandas as pd
from typing import List, Dict, Optional
from config import config


class BybitClient:
    def __init__(self):
        self.client = HTTP(
            testnet=config.BYBIT_TESTNET,
            api_key=config.BYBIT_API_KEY,
            api_secret=config.BYBIT_API_SECRET,
        )
    
    def get_klines(self, symbol: str, interval: str, start_time: int, end_time: int) -> List[Dict]:
        """
        Получение исторических данных (свечей) для указанного символа.
        
        Args:
            symbol: Торговая пара (например, BTCUSDT)
            interval: Интервал свечи (5 = 5 минут)
            start_time: Время начала в миллисекундах
            end_time: Время окончания в миллисекундах
        
        Returns:
            Список свечей
        """
        all_klines = []
        current_start = start_time
        
        # Bybit ограничивает количество записей за запрос, поэтому делаем пагинацию
        while current_start < end_time:
            try:
                response = self.client.get_kline(
                    category="spot",
                    symbol=symbol,
                    interval=interval,
                    start=current_start,
                    end=end_time,
                    limit=1000
                )
                
                if response["retCode"] == 0 and response["result"]["list"]:
                    klines = response["result"]["list"]
                    all_klines.extend(klines)
                    
                    # Получаем последнюю временную метку для следующего запроса
                    last_timestamp = int(klines[-1][0])
                    
                    # Если получили меньше лимита, значит данных больше нет
                    if len(klines) < 1000:
                        break
                    
                    current_start = last_timestamp + 1
                    time.sleep(0.1)  # Избегаем rate limit
                else:
                    break
                    
            except Exception as e:
                print(f"Ошибка при получении данных для {symbol}: {e}")
                break
        
        return all_klines
    
    def get_historical_data(self, symbol: str, months: int = 3) -> pd.DataFrame:
        """
        Получение исторических данных за указанное количество месяцев.
        
        Args:
            symbol: Торговая пара
            months: Количество месяцев истории
        
        Returns:
            DataFrame с историческими данными
        """
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=months * 30)).timestamp() * 1000)
        
        klines = self.get_klines(symbol, config.KLINE_INTERVAL, start_time, end_time)
        
        if not klines:
            return pd.DataFrame()
        
        # Преобразуем в DataFrame
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        
        # Конвертируем типы данных
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = df[col].astype(float)
        
        df = df.sort_values("timestamp").reset_index(drop=True)
        
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

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Bybit
    BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
    BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
    BYBIT_TESTNET = os.getenv("BYBIT_TESTNET", "false").lower() == "true"
    
    # DeepSeek
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    
    # Trading parameters
    MAX_TRADING_VOLUME = float(os.getenv("MAX_TRADING_VOLUME", "100"))
    TRADING_PAIR = os.getenv("TRADING_PAIR", "BTCUSDT")
    MAX_TRADE_AMOUNT = float(os.getenv("MAX_TRADE_AMOUNT", "10"))
    INTERVAL_MINUTES = int(os.getenv("INTERVAL_MINUTES", "5"))
    
    # Data collection parameters
    HISTORICAL_DAYS = 7  # Изменено с месяцев на дни для быстрой работы
    KLINE_INTERVAL = "5"  # 5 minutes
    
    # Market data symbols
    BTC_SYMBOL = "BTCUSDT"
    GOLD_SYMBOL = "XAUTUSDT"
    SILVER_SYMBOL = "XAGUSDT"
    
    # Polymarket
    POLYMARKET_URL = "https://polymarket.com/event/russia-x-ukraine-ceasefire-in-2025?tid=1761927992822"


config = Config()

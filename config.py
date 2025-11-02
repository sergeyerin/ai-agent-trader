import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Bybit
    BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
    BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
    BYBIT_TESTNET = os.getenv("BYBIT_TESTNET", "false").lower() == "true"
    
    # AI Provider (DeepSeek or OpenRouter)
    AI_PROVIDER = os.getenv("AI_PROVIDER", "deepseek")  # "deepseek" или "openrouter"
    
    # DeepSeek
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    
    # OpenRouter
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")
    
    # Trading parameters
    MAX_TRADING_VOLUME = float(os.getenv("MAX_TRADING_VOLUME", "100"))
    # Список торговых инструментов для анализа и торговли
    TRADING_PAIRS = os.getenv(
        "TRADING_PAIRS", 
        "SOLUSDT,BTCUSDT,ETHUSDT,TWTUSDT,MNTUSDT,XAUUSDT"
    ).split(",")
    MAX_TRADE_AMOUNT = float(os.getenv("MAX_TRADE_AMOUNT", "10"))
    INTERVAL_MINUTES = int(os.getenv("INTERVAL_MINUTES", "5"))
    
    # Debug/Testing flags
    ENABLE_AI_ANALYSIS = os.getenv("ENABLE_AI_ANALYSIS", "true").lower() == "true"
    ENABLE_TRADING = os.getenv("ENABLE_TRADING", "true").lower() == "true"
    
    # Data collection parameters
    HISTORICAL_DAYS = 7  # Изменено с месяцев на дни для быстрой работы
    KLINE_INTERVAL = "5"  # 5 minutes
    
    # Market data symbols (для анализа корреляций)
    GOLD_SYMBOL = "XAUTUSDT"
    SILVER_SYMBOL = "XAGUSDT"
    
    # Polymarket
    POLYMARKET_URL = "https://polymarket.com/event/russia-x-ukraine-ceasefire-in-2025?tid=1761927992822"


config = Config()

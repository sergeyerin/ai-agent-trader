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
        "BTCUSDT,ETHUSDT,SOLUSDT,XAUTUSDT"
    ).split(",")
    MAX_TRADE_AMOUNT = float(os.getenv("MAX_TRADE_AMOUNT", "10"))
    INTERVAL_MINUTES = int(os.getenv("INTERVAL_MINUTES", "15"))
    
    # Risk management
    STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "5.0"))  # Макс. убыток на позицию %
    TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "3.0"))  # Цель прибыли %
    MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "20.0"))  # Макс. дневной убыток USDT
    MIN_CONFIDENCE = int(os.getenv("MIN_CONFIDENCE", "60"))  # Мин. уверенность AI для сделки (0-100)
    POSITION_SIZE_PCT = float(os.getenv("POSITION_SIZE_PCT", "5.0"))  # Размер позиции % от портфеля
    
    # Trade history
    TRADES_DB_PATH = os.getenv("TRADES_DB_PATH", "trades.db")
    
    # Debug/Testing flags
    ENABLE_AI_ANALYSIS = os.getenv("ENABLE_AI_ANALYSIS", "true").lower() == "true"
    ENABLE_TRADING = os.getenv("ENABLE_TRADING", "true").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()  # DEBUG, INFO, WARNING, ERROR
    
    # Data collection parameters
    HISTORICAL_DAYS = 7  # Изменено с месяцев на дни для быстрой работы
    KLINE_INTERVAL = "5"  # 5 minutes
    
    # Market data symbols (для анализа корреляций)
    GOLD_SYMBOL = "XAUTUSDT"  # Золото с Bybit
    
    # Limit order settings
    USE_LIMIT_ORDERS = os.getenv("USE_LIMIT_ORDERS", "true").lower() == "true"
    LIMIT_ORDER_OFFSET_PCT = float(os.getenv("LIMIT_ORDER_OFFSET_PCT", "0.05"))  # Offset от рыночной цены %
    
    # Polymarket
    POLYMARKET_CONDITION_ID = os.getenv(
        "POLYMARKET_CONDITION_ID",
        "0x7a61644000000000000000000000000000000000000000000000000000000000"  # Placeholder — нужно обновить на актуальный
    )


config = Config()

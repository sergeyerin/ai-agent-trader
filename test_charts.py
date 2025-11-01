#!/usr/bin/env python3
"""
Тестовый скрипт для проверки генерации графиков.
"""

import logging
from bybit_client import BybitClient
from deepseek_client import DeepSeekClient
from config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_chart_generation():
    """Тест генерации графиков для различных инструментов."""
    logger.info("=" * 60)
    logger.info("Запуск теста генерации графиков")
    logger.info("=" * 60)
    
    bybit = BybitClient()
    deepseek = DeepSeekClient()
    
    # Тестовые символы
    test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    for symbol in test_symbols:
        logger.info(f"\n{'='*40}")
        logger.info(f"Тестирование {symbol}")
        logger.info(f"{'='*40}")
        
        # Получаем данные
        logger.info(f"Получение данных для {symbol}...")
        data = bybit.get_historical_data(symbol, days=3)
        
        if data.empty:
            logger.error(f"Не удалось получить данные для {symbol}")
            continue
        
        logger.info(f"Получено {len(data)} свечей для {symbol}")
        
        # Создаем график
        logger.info(f"Создание графика для {symbol}...")
        chart_base64 = deepseek.create_chart(
            data,
            symbol,
            f"{symbol.replace('USDT', '/USDT')}"
        )
        
        if chart_base64:
            logger.info(f"✓ График для {symbol} успешно создан (размер: {len(chart_base64)} символов)")
        else:
            logger.error(f"✗ Не удалось создать график для {symbol}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Тест завершен")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        test_chart_generation()
    except Exception as e:
        logger.error(f"Ошибка при выполнении теста: {e}", exc_info=True)

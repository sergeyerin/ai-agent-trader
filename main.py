#!/usr/bin/env python3
"""
Торговый агент для Bybit с использованием DeepSeek AI для анализа рынка.
"""

import schedule
import time
import logging
from trading_agent import TradingAgent
from config import config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def job():
    """Задача для выполнения агентом."""
    try:
        agent = TradingAgent()
        agent.run()
    except Exception as e:
        logger.error(f"Ошибка при выполнении торговой итерации: {e}", exc_info=True)


def main():
    """Основная функция запуска агента."""
    logger.info("=" * 80)
    logger.info("Запуск торгового агента")
    logger.info(f"Интервал запуска: каждые {config.INTERVAL_MINUTES} минут")
    logger.info(f"Торговая пара: {config.TRADING_PAIR}")
    logger.info(f"Максимальный объем торговли: {config.MAX_TRADING_VOLUME} USDT")
    logger.info(f"Максимальная сумма сделки: {config.MAX_TRADE_AMOUNT} USDT")
    logger.info("=" * 80)
    
    # Запускаем первую итерацию сразу
    logger.info("Выполнение первой итерации...")
    job()
    
    # Планируем выполнение каждые N минут
    schedule.every(config.INTERVAL_MINUTES).minutes.do(job)
    
    logger.info(f"Агент запущен. Следующая итерация через {config.INTERVAL_MINUTES} минут.")
    
    # Бесконечный цикл выполнения задач по расписанию
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nТорговый агент остановлен пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)

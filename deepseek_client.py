from openai import OpenAI
import pandas as pd
import json
import logging
from typing import Dict, Optional, Tuple
from config import config

logger = logging.getLogger(__name__)


class DeepSeekClient:
    def __init__(self):
        self.provider = config.AI_PROVIDER.lower()
        
        if self.provider == "openrouter":
            logger.info("Использование OpenRouter для доступа к AI моделям")
            self.client = OpenAI(
                api_key=config.OPENROUTER_API_KEY,
                base_url=config.OPENROUTER_API_BASE
            )
            self.model = config.OPENROUTER_MODEL
            self.extra_headers = {
                "HTTP-Referer": "https://github.com/sergeyerin/ai-agent-trader",
                "X-Title": "AI Trading Agent"
            }
        else:  # deepseek
            logger.info("Использование DeepSeek напрямую")
            self.client = OpenAI(
                api_key=config.DEEPSEEK_API_KEY,
                base_url=config.DEEPSEEK_API_BASE
            )
            self.model = config.DEEPSEEK_MODEL
            self.extra_headers = {}
    
    def prepare_market_data(
        self,
        trading_pair_data: pd.DataFrame,
        trading_pair_symbol: str,
        gold_data: pd.DataFrame,
        silver_data: pd.DataFrame,
        polymarket_info: str
    ) -> str:
        """
        Подготовка данных рынка для отправки в DeepSeek.
        
        Args:
            trading_pair_data: Данные по торгуемой паре
            trading_pair_symbol: Символ торгуемой пары
            gold_data: Данные по золоту
            silver_data: Данные по серебру
            polymarket_info: Информация с Polymarket
        
        Returns:
            Форматированная строка с данными
        """
        data_summary = []
        
        # Торгуемая пара
        if not trading_pair_data.empty:
            recent_data = trading_pair_data.tail(min(288, len(trading_pair_data)))  # Последние 288 свечей (24 часа) или все доступные
            data_summary.append(f"=== {trading_pair_symbol} (последние {len(recent_data)} свечей) ===")
            data_summary.append(f"Текущая цена: {recent_data['close'].iloc[-1]:.2f} USDT")
            data_summary.append(f"Минимум за период: {recent_data['low'].min():.2f} USDT")
            data_summary.append(f"Максимум за период: {recent_data['high'].max():.2f} USDT")
            data_summary.append(f"Средний объем: {recent_data['volume'].mean():.2f}")
            data_summary.append(f"Изменение за период: {((recent_data['close'].iloc[-1] / recent_data['open'].iloc[0] - 1) * 100):.2f}%")
            data_summary.append("")
        
        # Золото
        if not gold_data.empty:
            gold_recent = gold_data.tail(min(288, len(gold_data)))
            data_summary.append(f"=== Золото XAUT/USDT (последние {len(gold_recent)} свечей) ===")
            data_summary.append(f"Текущая цена: {gold_recent['close'].iloc[-1]:.2f} USDT")
            data_summary.append(f"Изменение за период: {((gold_recent['close'].iloc[-1] / gold_recent['open'].iloc[0] - 1) * 100):.2f}%")
            data_summary.append("")
        
        # Серебро
        if not silver_data.empty:
            silver_recent = silver_data.tail(min(288, len(silver_data)))
            data_summary.append(f"=== Серебро XAGUSD (последние {len(silver_recent)} свечей) ===")
            data_summary.append(f"Текущая цена: {silver_recent['close'].iloc[-1]:.2f} USD")
            data_summary.append(f"Изменение за период: {((silver_recent['close'].iloc[-1] / silver_recent['open'].iloc[0] - 1) * 100):.2f}%")
            data_summary.append("")
        
        # Polymarket
        data_summary.append("=== Геополитический фактор ===")
        data_summary.append(polymarket_info)
        
        return "\n".join(data_summary)
    
    def get_trading_recommendation(
        self,
        market_data: str,
        trading_pair_symbol: str,
        current_price: float,
        max_trade_amount: float
    ) -> Optional[Dict]:
        """
        Получение торговой рекомендации от DeepSeek.
        
        Args:
            market_data: Форматированные данные рынка
            trading_pair_symbol: Символ торгуемой пары
            current_price: Текущая цена
            max_trade_amount: Максимальная сумма сделки в USDT
        
        Returns:
            Словарь с рекомендацией или None
        """
        # Извлекаем базовую валюту
        base_currency = trading_pair_symbol.replace("USDT", "")
        
        prompt = f"""Ты успешный трейдер криптовалют с многолетним опытом анализа рынков.

У тебя есть следующие исторические данные для анализа:

{market_data}

ТЕКУЩИЕ ПАРАМЕТРЫ:
- Текущая цена {base_currency}: {current_price:.2f} USDT
- Максимальная сумма сделки: {max_trade_amount} USDT
- Торговая пара: {trading_pair_symbol}

ЗАДАЧА:
На основании предоставленных данных (включая корреляцию с золотом, серебром и геополитической ситуацией) 
необходимо принять решение о покупке, продаже или бездействии с {base_currency}.

Учитывай:
1. Технический анализ (тренды, уровни поддержки/сопротивления)
2. Корреляцию с традиционными активами (золото, серебро)
3. Геополитические риски
4. Объемы торгов
5. Риск-менеджмент

ФОРМАТ ОТВЕТА (строго JSON):
{{
    "action": "buy" | "sell" | "hold",
    "price_from": число или null,
    "price_to": число или null,
    "quantity_usdt": число (сумма в USDT) или null,
    "reasoning": "краткое объяснение решения"
}}

Где:
- action: "buy" (покупать), "sell" (продавать), "hold" (не проводить сделку)
- price_from: нижняя граница цены для сделки (null если hold)
- price_to: верхняя граница цены для сделки (null если hold)
- quantity_usdt: рекомендуемая сумма сделки в USDT (не более {max_trade_amount}, null если hold)
- reasoning: краткое обоснование решения

ВАЖНО: Ответь ТОЛЬКО JSON, без дополнительного текста."""

        try:
            # Подготовка параметров запроса
            request_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "Ты профессиональный трейдер криптовалют. Отвечай только в формате JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            # Добавляем extra_headers для OpenRouter
            if self.extra_headers:
                request_params["extra_headers"] = self.extra_headers
            
            logger.info(f"Отправка запроса в {self.provider} (модель: {self.model})")
            response = self.client.chat.completions.create(**request_params)
            
            response_text = response.choices[0].message.content.strip()
            
            # Попытка извлечь JSON из ответа
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            recommendation = json.loads(response_text)
            
            # Валидация ответа
            if "action" not in recommendation:
                print("Ошибка: отсутствует поле 'action' в ответе")
                return None
            
            return recommendation
            
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON от DeepSeek: {e}")
            print(f"Ответ: {response_text if 'response_text' in locals() else 'N/A'}")
            return None
        except Exception as e:
            print(f"Ошибка при получении рекомендации от DeepSeek: {e}")
            return None

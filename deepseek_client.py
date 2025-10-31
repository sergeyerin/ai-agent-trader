from openai import OpenAI
import pandas as pd
import json
from typing import Dict, Optional, Tuple
from config import config


class DeepSeekClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_API_BASE
        )
    
    def prepare_market_data(
        self,
        btc_data: pd.DataFrame,
        gold_data: pd.DataFrame,
        silver_data: pd.DataFrame,
        polymarket_info: str
    ) -> str:
        """
        Подготовка данных рынка для отправки в DeepSeek.
        
        Args:
            btc_data: Данные по BTC/USDT
            gold_data: Данные по золоту
            silver_data: Данные по серебру
            polymarket_info: Информация с Polymarket
        
        Returns:
            Форматированная строка с данными
        """
        data_summary = []
        
        # BTC/USDT
        if not btc_data.empty:
            btc_recent = btc_data.tail(100)  # Последние 100 свечей (примерно 8 часов)
            data_summary.append("=== BTC/USDT (последние 100 свечей) ===")
            data_summary.append(f"Текущая цена: {btc_recent['close'].iloc[-1]:.2f} USDT")
            data_summary.append(f"Минимум за период: {btc_recent['low'].min():.2f} USDT")
            data_summary.append(f"Максимум за период: {btc_recent['high'].max():.2f} USDT")
            data_summary.append(f"Средний объем: {btc_recent['volume'].mean():.2f}")
            data_summary.append(f"Изменение за период: {((btc_recent['close'].iloc[-1] / btc_recent['open'].iloc[0] - 1) * 100):.2f}%")
            data_summary.append("")
        
        # Золото
        if not gold_data.empty:
            gold_recent = gold_data.tail(100)
            data_summary.append("=== Золото XAUT/USDT (последние 100 свечей) ===")
            data_summary.append(f"Текущая цена: {gold_recent['close'].iloc[-1]:.2f} USDT")
            data_summary.append(f"Изменение за период: {((gold_recent['close'].iloc[-1] / gold_recent['open'].iloc[0] - 1) * 100):.2f}%")
            data_summary.append("")
        
        # Серебро
        if not silver_data.empty:
            silver_recent = silver_data.tail(100)
            data_summary.append("=== Серебро XAGUSD (последние 100 свечей) ===")
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
        current_price: float,
        max_trade_amount: float
    ) -> Optional[Dict]:
        """
        Получение торговой рекомендации от DeepSeek.
        
        Args:
            market_data: Форматированные данные рынка
            current_price: Текущая цена BTC
            max_trade_amount: Максимальная сумма сделки в USDT
        
        Returns:
            Словарь с рекомендацией или None
        """
        prompt = f"""Ты успешный трейдер криптовалют с многолетним опытом анализа рынков.

У тебя есть следующие исторические данные для анализа:

{market_data}

ТЕКУЩИЕ ПАРАМЕТРЫ:
- Текущая цена BTC: {current_price:.2f} USDT
- Максимальная сумма сделки: {max_trade_amount} USDT
- Торговая пара: BTC/USDT

ЗАДАЧА:
На основании предоставленных данных (включая корреляцию с золотом, серебром и геополитической ситуацией) 
необходимо принять решение о покупке, продаже или бездействии с BTC.

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
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "Ты профессиональный трейдер криптовалют. Отвечай только в формате JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
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

from openai import OpenAI
import pandas as pd
import json
import logging
import httpx
import base64
import io
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
from typing import Dict, Optional, Tuple, List
from config import config

logger = logging.getLogger(__name__)


class DeepSeekClient:
    def __init__(self):
        self.provider = config.AI_PROVIDER.lower()
        
        # Создаем http_client без прокси
        http_client = httpx.Client(
            timeout=30.0
        )
        
        if self.provider == "openrouter":
            logger.info("Использование OpenRouter для доступа к AI моделям")
            self.client = OpenAI(
                api_key=config.OPENROUTER_API_KEY,
                base_url=config.OPENROUTER_API_BASE,
                http_client=http_client
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
                base_url=config.DEEPSEEK_API_BASE,
                http_client=http_client
            )
            self.model = config.DEEPSEEK_MODEL
            self.extra_headers = {}
    
    def create_chart(self, df: pd.DataFrame, symbol: str, title: str) -> str:
        """
        Создание графика цен и кодирование в base64.
        
        Args:
            df: DataFrame с данными OHLCV
            symbol: Символ инструмента
            title: Заголовок графика
        
        Returns:
            Base64-кодированное изображение графика
        """
        if df.empty:
            return ""
        
        try:
            # Берем последние 7 дней (2016 свечей по 5 минут)
            # 7 дней * 24 часа * 60 минут / 5 минут = 2016 свечей
            plot_data = df.tail(min(2016, len(df))).copy()
            
            # Подготавливаем данные для mplfinance
            plot_data_mpf = plot_data.set_index('timestamp')
            plot_data_mpf = plot_data_mpf[['open', 'high', 'low', 'close', 'volume']]
            plot_data_mpf.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Настройка стиля
            mc = mpf.make_marketcolors(
                up='#26a69a',
                down='#ef5350',
                edge='inherit',
                wick='inherit',
                volume='in'
            )
            
            s = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='-',
                gridcolor='#e0e0e0',
                facecolor='white',
                figcolor='white'
            )
            
            # Создаем график
            fig, axes = mpf.plot(
                plot_data_mpf,
                type='candle',
                style=s,
                title=f'{title} (7 дней)',
                ylabel='Цена (USDT)',
                ylabel_lower='Объем',
                volume=True,
                figsize=(24, 10),  # Очень широкий график для отображения всех 2016 свечей
                returnfig=True,
                datetime_format='%m-%d',
                xrotation=45,
                warn_too_much_data=2500  # Повышаем порог предупреждения выше 2016
            )
            
            # Создаем директорию для графиков если её нет
            charts_dir = "charts"
            if not os.path.exists(charts_dir):
                os.makedirs(charts_dir)
            
            # Генерируем имя файла с временной меткой
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_filename = f"{charts_dir}/{symbol}_{timestamp}.png"
            
            # Сохраняем график на диск
            plt.savefig(chart_filename, format='png', dpi=100, bbox_inches='tight')
            logger.info(f"График сохранен: {chart_filename}")
            
            # Сохраняем в буфер и кодируем в base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close(fig)
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Ошибка при создании графика для {symbol}: {e}")
            return ""
    
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
            
            # Добавляем детальные временные ряды (последние 50 свечей для детального анализа)
            data_summary.append("Детальные данные (последние 50 свечей):")
            data_summary.append("timestamp,open,high,low,close,volume")
            detailed_data = recent_data.tail(50)
            for _, row in detailed_data.iterrows():
                data_summary.append(
                    f"{row['timestamp']},{row['open']:.2f},{row['high']:.2f},{row['low']:.2f},{row['close']:.2f},{row['volume']:.2f}"
                )
            data_summary.append("")
        
        # Золото (для анализа корреляции)
        if not gold_data.empty:
            gold_recent = gold_data.tail(min(288, len(gold_data)))
            data_summary.append(f"=== Золото XAUT/USDT - для анализа корреляции (последние {len(gold_recent)} свечей) ===")
            data_summary.append(f"Текущая цена: {gold_recent['close'].iloc[-1]:.2f} USDT")
            data_summary.append(f"Изменение за период: {((gold_recent['close'].iloc[-1] / gold_recent['open'].iloc[0] - 1) * 100):.2f}%")
            data_summary.append("")
            
            # Детальные временные ряды для золота
            data_summary.append("Детальные данные (последние 50 свечей):")
            data_summary.append("timestamp,open,high,low,close,volume")
            detailed_gold = gold_recent.tail(50)
            for _, row in detailed_gold.iterrows():
                data_summary.append(
                    f"{row['timestamp']},{row['open']:.2f},{row['high']:.2f},{row['low']:.2f},{row['close']:.2f},{row['volume']:.2f}"
                )
            data_summary.append("")
        
        # Серебро (для анализа корреляции)
        if not silver_data.empty:
            silver_recent = silver_data.tail(min(288, len(silver_data)))
            data_summary.append(f"=== Серебро XAGUSD - для анализа корреляции (последние {len(silver_recent)} свечей) ===")
            data_summary.append(f"Текущая цена: {silver_recent['close'].iloc[-1]:.2f} USD")
            data_summary.append(f"Изменение за период: {((silver_recent['close'].iloc[-1] / silver_recent['open'].iloc[0] - 1) * 100):.2f}%")
            data_summary.append("")
            
            # Детальные временные ряды для серебра
            data_summary.append("Детальные данные (последние 50 свечей):")
            data_summary.append("timestamp,open,high,low,close,volume")
            detailed_silver = silver_recent.tail(50)
            for _, row in detailed_silver.iterrows():
                data_summary.append(
                    f"{row['timestamp']},{row['open']:.2f},{row['high']:.2f},{row['low']:.2f},{row['close']:.2f},{row['volume']:.2f}"
                )
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
        max_trade_amount: float,
        chart_images: Optional[Dict[str, str]] = None
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
        
        # Добавляем упоминание о графиках если они есть
        chart_note = ""
        if chart_images and trading_pair_symbol in chart_images:
            chart_note = "\n\nВНИМАНИЕ: К этому запросу прикреплен график с историческими данными цены и объема. Используй визуальный анализ графика для выявления трендов, уровней поддержки/сопротивления и паттернов."
        
        prompt = f"""Ты успешный трейдер криптовалют с многолетним опытом анализа рынков.

У тебя есть следующие исторические данные для анализа:

{market_data}{chart_note}

ТЕКУЩИЕ ПАРАМЕТРЫ:
- Текущая цена {base_currency}: {current_price:.2f} USDT
- Максимальная сумма сделки: {max_trade_amount} USDT
- Торговая пара: {trading_pair_symbol}

ЗАДАЧА:
На основании предоставленных данных необходимо принять решение о покупке, продаже или бездействии с {base_currency}.

ВАЖНО: Данные по золоту и серебру предоставлены ТОЛЬКО для анализа корреляции с криптовалютной парой.
Мы НЕ торгуем золотом или серебром - они используются как индикаторы настроений рынка драгоценных металлов,
которые могут коррелировать с криптовалютами в периоды экономической нестабильности.

Учитывай:
1. Технический анализ криптовалютной пары (тренды, уровни поддержки/сопротивления, объемы)
2. Корреляцию движения цены криптовалюты с золотом/серебром (синхронность трендов, расхождения)
3. Геополитические риски и их влияние на рынок
4. Риск-менеджмент

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
            # Подготовка сообщений
            messages = [
                {"role": "system", "content": "Ты профессиональный трейдер криптовалют. Отвечай только в формате JSON."}
            ]
            
            # Если есть графики, добавляем их в запрос
            if chart_images and trading_pair_symbol in chart_images:
                # Проверяем, поддерживает ли модель vision
                if "vision" in self.model.lower() or "gpt-4" in self.model.lower():
                    logger.info(f"Добавление графика для {trading_pair_symbol} в запрос")
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{chart_images[trading_pair_symbol]}"
                                }
                            }
                        ]
                    })
                else:
                    # Для моделей без vision просто отправляем текст
                    logger.info(f"Модель {self.model} не поддерживает изображения, отправка только текста")
                    messages.append({"role": "user", "content": prompt})
            else:
                messages.append({"role": "user", "content": prompt})
            
            # Подготовка параметров запроса
            request_params = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            logger.info(f"Отправка запроса в {self.provider} (модель: {self.model})")
            
            # Отладочный вывод запроса (ПОЛНЫЙ)
            logger.debug("=" * 80)
            logger.debug("AI REQUEST (FULL):")
            logger.debug(f"Model: {request_params['model']}")
            logger.debug(f"Temperature: {request_params['temperature']}")
            logger.debug(f"Max tokens: {request_params['max_tokens']}")
            logger.debug("-" * 80)
            for idx, msg in enumerate(messages):
                logger.debug(f"\nMessage {idx + 1} - Role: {msg['role']}")
                if isinstance(msg.get('content'), str):
                    # Выводим ПОЛНЫЙ контент (включая все временные ряды)
                    logger.debug(f"Content (length: {len(msg['content'])} chars):\n{msg['content']}")
                elif isinstance(msg.get('content'), list):
                    logger.debug(f"Content: [multipart message with {len(msg['content'])} parts]")
                    for part_idx, part in enumerate(msg['content']):
                        if part.get('type') == 'text':
                            # Выводим ПОЛНЫЙ текст
                            logger.debug(f"  Part {part_idx + 1} - Text (length: {len(part['text'])} chars):\n{part['text']}")
                        elif part.get('type') == 'image_url':
                            # Для изображений выводим только инфо, не base64
                            img_url = part['image_url']['url']
                            logger.debug(f"  Part {part_idx + 1} - Image: base64 data, length: {len(img_url)} chars")
            logger.debug("=" * 80)
            
            # Для OpenRouter передаем extra_headers отдельно
            if self.extra_headers:
                response = self.client.chat.completions.create(
                    **request_params,
                    extra_headers=self.extra_headers
                )
            else:
                response = self.client.chat.completions.create(**request_params)
            
            response_text = response.choices[0].message.content.strip()
            
            # Отладочный вывод ответа
            logger.debug("=" * 80)
            logger.debug("AI RESPONSE:")
            logger.debug(f"Response text:\n{response_text}")
            logger.debug("=" * 80)
            
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

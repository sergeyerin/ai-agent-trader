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
        silver_data: pd.DataFrame = None,
        polymarket_info: str = "",
        indicators_text: str = "",
        portfolio_text: str = "",
        history_text: str = "",
    ) -> str:
        """
        Подготовка данных рынка для отправки в DeepSeek.
        
        Args:
            trading_pair_data: Данные по торгуемой паре
            trading_pair_symbol: Символ торгуемой пары
            gold_data: Данные по золоту
            silver_data: Данные по серебру
            polymarket_info: Информация с Polymarket
            indicators_text: Технические индикаторы (форматированный текст)
            portfolio_text: Состояние портфеля (форматированный текст)
            history_text: История сделок (форматированный текст)
        
        Returns:
            Форматированная строка с данными
        """
        data_summary = []
        
        # Портфель (первым — чтобы AI знал контекст)
        if portfolio_text:
            data_summary.append(portfolio_text)
            data_summary.append("")
        
        # История сделок
        if history_text:
            data_summary.append(history_text)
            data_summary.append("")
        
        # Технические индикаторы
        if indicators_text:
            data_summary.append(indicators_text)
            data_summary.append("")
        
        # Торгуемая пара
        if not trading_pair_data.empty:
            recent_data = trading_pair_data.tail(min(288, len(trading_pair_data)))
            data_summary.append(f"=== {trading_pair_symbol} (последние {len(recent_data)} свечей) ===")
            data_summary.append(f"Текущая цена: {recent_data['close'].iloc[-1]:.2f} USDT")
            data_summary.append(f"Минимум за период: {recent_data['low'].min():.2f} USDT")
            data_summary.append(f"Максимум за период: {recent_data['high'].max():.2f} USDT")
            data_summary.append(f"Средний объем: {recent_data['volume'].mean():.2f}")
            data_summary.append(f"Изменение за период: {((recent_data['close'].iloc[-1] / recent_data['open'].iloc[0] - 1) * 100):.2f}%")
            data_summary.append("")
            
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
            
            data_summary.append("Детальные данные (последние 50 свечей):")
            data_summary.append("timestamp,open,high,low,close,volume")
            detailed_gold = gold_recent.tail(50)
            for _, row in detailed_gold.iterrows():
                data_summary.append(
                    f"{row['timestamp']},{row['open']:.2f},{row['high']:.2f},{row['low']:.2f},{row['close']:.2f},{row['volume']:.2f}"
                )
            data_summary.append("")
        
        # Серебро (для анализа корреляции)
        if silver_data is not None and not silver_data.empty:
            silver_recent = silver_data.tail(min(288, len(silver_data)))
            data_summary.append(f"=== Серебро XAGUSD - для анализа корреляции (последние {len(silver_recent)} свечей) ===")
            data_summary.append(f"Текущая цена: {silver_recent['close'].iloc[-1]:.2f} USD")
            data_summary.append(f"Изменение за период: {((silver_recent['close'].iloc[-1] / silver_recent['open'].iloc[0] - 1) * 100):.2f}%")
            data_summary.append("")
            
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
        
        prompt = f"""Ты профессиональный количественный трейдер. Принимай решения на основе технических индикаторов и данных, а не интуиции.

ДАННЫЕ ДЛЯ АНАЛИЗА:
{market_data}{chart_note}

ТЕКУЩИЕ ПАРАМЕТРЫ:
- Текущая цена {base_currency}: {current_price:.2f} USDT
- Максимальная сумма сделки: {max_trade_amount} USDT
- Торговая пара: {trading_pair_symbol}
- Комиссия Bybit: 0.1% за сделку (учитывай при расчёте целесообразности)

ПРАВИЛА ПРИНЯТИЯ РЕШЕНИЙ:
1. Используй предоставленные технические индикаторы (RSI, EMA, MACD, Bollinger Bands, ATR) как основу решения.
2. НЕ покупай при RSI > 70 (перекуплен). НЕ продавай при RSI < 30 (перепродан).
3. Торгуй по тренду: покупай когда EMA20 > EMA50 > EMA200, продавай при обратном.
4. Учитывай ширину Bollinger Bands — узкие полосы = скорый сильный ход.
5. Если нет чёткого сигнала — выбирай hold. Лучше пропустить сделку, чем потерять на комиссиях.
6. Учитывай текущий портфель: не покупай то, чего и так много. Продавай с прибылью.
7. Анализируй свои прошлые сделки: если последние сделки убыточные, будь осторожнее.
8. Данные по золоту/серебру — ТОЛЬКО для анализа корреляции, мы ими НЕ торгуем.

ФОРМАТ ОТВЕТА (строго JSON):
{{
    "action": "buy" | "sell" | "hold",
    "price_from": число или null,
    "price_to": число или null,
    "quantity_usdt": число (сумма в USDT, не более {max_trade_amount}) или null,
    "confidence": число от 0 до 100,
    "reasoning": "краткое объяснение с указанием конкретных индикаторов"
}}

ВАЖНО: Ответь ТОЛЬКО JSON, без дополнительного текста. Поле confidence — твоя уверенность в рекомендации от 0 до 100."""

        try:
            # Подготовка сообщений
            system_prompt = (
                "Ты количественный трейдер. Отвечай только в формате JSON.\n\n"
                "СТРАТЕГИЯ:\n"
                "1. Momentum: покупай при восходящем тренде (EMA20>EMA50>EMA200) и растущем MACD.\n"
                "2. Mean Reversion: покупай при отскоке от нижней Bollinger Band + RSI<30.\n"
                "3. НЕ торгуй если ожидаемый профит < 0.3% (комиссия round-trip = 0.2%).\n"
                "4. В сомнениях — всегда hold."
            )
            messages = [
                {"role": "system", "content": system_prompt}
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
                "temperature": 0.2,
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

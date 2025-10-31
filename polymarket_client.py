import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict
from config import config


class PolymarketClient:
    def __init__(self):
        self.url = config.POLYMARKET_URL
    
    def get_ceasefire_probability(self) -> Optional[Dict]:
        """
        Получение вероятности прекращения огня между Россией и Украиной.
        
        Returns:
            Словарь с данными о голосовании или None
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
            response = requests.get(self.url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Ищем процент вероятности на странице
            # Polymarket обычно отображает процент в определенных элементах
            probability_text = None
            
            # Попытка найти процент в различных элементах
            for elem in soup.find_all(text=True):
                text = elem.strip()
                if "%" in text and text.replace("%", "").replace(" ", "").isdigit():
                    try:
                        prob = float(text.replace("%", "").strip())
                        if 0 <= prob <= 100:
                            probability_text = prob
                            break
                    except ValueError:
                        continue
            
            if probability_text is not None:
                return {
                    "event": "Russia x Ukraine ceasefire in 2025",
                    "probability": probability_text,
                    "source": "Polymarket"
                }
            else:
                # Если не нашли процент, возвращаем базовые данные
                return {
                    "event": "Russia x Ukraine ceasefire in 2025",
                    "probability": None,
                    "source": "Polymarket",
                    "note": "Unable to parse probability from page"
                }
                
        except Exception as e:
            print(f"Ошибка при получении данных с Polymarket: {e}")
            return None
    
    def get_formatted_data(self) -> str:
        """
        Получение отформатированных данных для использования в промпте.
        
        Returns:
            Строка с данными о голосовании
        """
        data = self.get_ceasefire_probability()
        
        if not data:
            return "Данные Polymarket недоступны"
        
        if data.get("probability") is not None:
            return (
                f"Polymarket - {data['event']}: "
                f"вероятность {data['probability']}%"
            )
        else:
            return f"Polymarket - {data['event']}: данные недоступны"

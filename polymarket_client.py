import requests
import json
import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

# Polymarket Gamma API (публичный, без ключа)
GAMMA_API_BASE = "https://gamma-api.polymarket.com"


class PolymarketClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "AI-Trading-Agent/1.0",
            "Accept": "application/json",
        })

    def search_events(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Поиск событий на Polymarket через Gamma API.

        Args:
            query: Поисковый запрос
            limit: Максимум результатов
        """
        try:
            response = self.session.get(
                f"{GAMMA_API_BASE}/events",
                params={"title": query, "limit": limit, "active": True},
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Ошибка поиска событий Polymarket: {e}")
            return []

    def get_geopolitical_data(self) -> Optional[Dict]:
        """
        Получает геополитические данные, релевантные для крипто-торговли.
        """
        keywords = ["ceasefire", "russia ukraine", "war"]

        for keyword in keywords:
            events = self.search_events(keyword, limit=3)
            if not events:
                continue

            results = []
            for event in events:
                title = event.get("title", "Unknown")
                markets = event.get("markets", [])
                for market in markets:
                    outcome = market.get("outcomePrices")
                    question = market.get("question", title)
                    if outcome:
                        try:
                            prices = json.loads(outcome) if isinstance(outcome, str) else outcome
                            yes_price = float(prices[0]) if prices else None
                            if yes_price is not None:
                                results.append({
                                    "question": question,
                                    "probability_yes": yes_price * 100,
                                })
                        except (json.JSONDecodeError, IndexError, TypeError, ValueError):
                            continue

            if results:
                return {"source": "Polymarket Gamma API", "events": results}

        return None

    def get_formatted_data(self) -> str:
        """
        Получение отформатированных данных для промпта AI.
        """
        data = self.get_geopolitical_data()

        if not data or not data.get("events"):
            return "Данные Polymarket недоступны"

        lines = [f"Источник: {data['source']}"]
        for event in data["events"][:5]:
            prob = event["probability_yes"]
            question = event["question"]
            lines.append(f"  - {question}: вероятность {prob:.1f}%")

        return "\n".join(lines)

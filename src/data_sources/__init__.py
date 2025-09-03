"""
Data sources module for Portfolio Intelligence Platform.
Handles external data integration for news, market data, and sentiment analysis.
"""

from .news_api import NewsProvider
from .market_data import MacroDataProvider

__all__ = [
    'NewsProvider',
    'MacroDataProvider'
]
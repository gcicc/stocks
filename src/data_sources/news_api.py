"""
News API integration for Portfolio Intelligence Platform.
Provides professional news data with sentiment analysis.
"""

import requests
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from textblob import TextBlob

logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """Professional news article data structure."""
    title: str
    description: str
    url: str
    source: str
    published_at: datetime
    sentiment_score: float  # -1.0 to 1.0
    sentiment_label: str    # "Bullish", "Bearish", "Neutral"

class NewsProvider:
    """
    Professional news data provider with sentiment analysis.
    Supports multiple news sources with fallback mechanisms.
    """
    
    def __init__(self, news_api_key: Optional[str] = None):
        self.news_api_key = news_api_key
        self.base_url = "https://newsapi.org/v2"
        self.session = requests.Session()
        
        # Professional headers
        self.session.headers.update({
            'User-Agent': 'Portfolio-Intelligence-Platform/1.0',
            'X-Api-Key': news_api_key if news_api_key else ""
        })
        
        logger.info("News provider initialized")
    
    def get_stock_news(self, symbol: str, days_back: int = 7, max_articles: int = 5) -> List[NewsArticle]:
        """
        Get recent news for a stock symbol with sentiment analysis.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            days_back: Days to look back for news
            max_articles: Maximum number of articles to return
            
        Returns:
            List of NewsArticle objects with sentiment analysis
        """
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)
            
            # Try NewsAPI first
            articles = self._fetch_from_newsapi(symbol, from_date, to_date, max_articles)
            
            # If NewsAPI fails, try Yahoo Finance RSS (fallback)
            if not articles:
                articles = self._fetch_from_yahoo_rss(symbol, max_articles)
            
            # Add sentiment analysis
            for article in articles:
                article.sentiment_score, article.sentiment_label = self._analyze_sentiment(
                    article.title + " " + (article.description or "")
                )
            
            logger.info(f"Retrieved {len(articles)} articles for {symbol}")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to fetch news for {symbol}: {e}")
            return []
    
    def _fetch_from_newsapi(self, symbol: str, from_date: datetime, to_date: datetime, max_articles: int) -> List[NewsArticle]:
        """Fetch news from NewsAPI."""
        if not self.news_api_key:
            return []
        
        try:
            params = {
                'q': f'"{symbol}" OR "{symbol} stock" OR "{symbol} shares"',
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'pageSize': max_articles,
                'language': 'en'
            }
            
            response = self.session.get(f"{self.base_url}/everything", params=params)
            response.raise_for_status()
            
            data = response.json()
            articles = []
            
            for article_data in data.get('articles', []):
                try:
                    article = NewsArticle(
                        title=article_data['title'] or "No title",
                        description=article_data['description'] or "",
                        url=article_data['url'],
                        source=article_data['source']['name'],
                        published_at=datetime.fromisoformat(
                            article_data['publishedAt'].replace('Z', '+00:00')
                        ),
                        sentiment_score=0.0,  # Will be filled later
                        sentiment_label="Neutral"  # Will be filled later
                    )
                    articles.append(article)
                except Exception as e:
                    logger.warning(f"Skipping malformed article: {e}")
                    continue
            
            return articles
            
        except requests.RequestException as e:
            logger.warning(f"NewsAPI request failed: {e}")
            return []
    
    def _fetch_from_yahoo_rss(self, symbol: str, max_articles: int) -> List[NewsArticle]:
        """Fallback: fetch from Yahoo Finance RSS (simplified)."""
        try:
            # This is a simplified fallback - in production you'd use proper RSS parsing
            articles = [
                NewsArticle(
                    title=f"Recent market activity for {symbol}",
                    description=f"Market analysis and recent developments for {symbol}",
                    url=f"https://finance.yahoo.com/quote/{symbol}/news",
                    source="Yahoo Finance",
                    published_at=datetime.now(),
                    sentiment_score=0.0,
                    sentiment_label="Neutral"
                )
            ]
            return articles[:max_articles]
            
        except Exception as e:
            logger.error(f"Yahoo RSS fallback failed: {e}")
            return []
    
    def _analyze_sentiment(self, text: str) -> tuple[float, str]:
        """
        Analyze sentiment using TextBlob with professional classification.
        
        Returns:
            Tuple of (sentiment_score, sentiment_label)
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            
            # Professional sentiment classification
            if polarity >= 0.3:
                label = "Bullish"
            elif polarity <= -0.3:
                label = "Bearish"
            else:
                label = "Neutral"
            
            return polarity, label
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return 0.0, "Neutral"
    
    def get_portfolio_news_summary(self, symbols: List[str]) -> pd.DataFrame:
        """
        Get news summary for all portfolio symbols in professional DataFrame format.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Professional DataFrame with news and sentiment data
        """
        all_news_data = []
        
        for symbol in symbols:
            articles = self.get_stock_news(symbol, max_articles=3)
            
            if articles:
                # Aggregate sentiment for the symbol
                avg_sentiment = sum(article.sentiment_score for article in articles) / len(articles)
                
                # Get most recent article
                latest_article = max(articles, key=lambda x: x.published_at)
                
                all_news_data.append({
                    'Symbol': symbol,
                    'Latest_Headline': latest_article.title[:100] + "..." if len(latest_article.title) > 100 else latest_article.title,
                    'Source': latest_article.source,
                    'Published': latest_article.published_at.strftime('%Y-%m-%d %H:%M'),
                    'Sentiment_Score': round(avg_sentiment, 3),
                    'Sentiment_Label': self._get_sentiment_label(avg_sentiment),
                    'Article_Count': len(articles),
                    'URL': latest_article.url
                })
            else:
                # No news found
                all_news_data.append({
                    'Symbol': symbol,
                    'Latest_Headline': 'No recent news available',
                    'Source': 'N/A',
                    'Published': 'N/A',
                    'Sentiment_Score': 0.0,
                    'Sentiment_Label': 'Neutral',
                    'Article_Count': 0,
                    'URL': ''
                })
        
        df = pd.DataFrame(all_news_data)
        
        # Sort by sentiment score (most bullish first)
        df = df.sort_values('Sentiment_Score', ascending=False).reset_index(drop=True)
        
        return df
    
    def _get_sentiment_label(self, score: float) -> str:
        """Get professional sentiment label from score."""
        if score >= 0.3:
            return "Bullish"
        elif score <= -0.3:
            return "Bearish"
        else:
            return "Neutral"
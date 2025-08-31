"""
Async data manager with multiple data sources and fallback mechanisms.
Optimized for fast parallel data fetching with caching.
"""

import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from joblib import Memory
from concurrent.futures import ThreadPoolExecutor
import time

from utils.config import config

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Container for market data for a single symbol."""
    symbol: str
    prices: pd.DataFrame  # OHLCV data
    current_price: float
    previous_close: float
    volume: int
    last_updated: datetime
    data_source: str
    
    @property
    def price_change(self) -> float:
        """Calculate price change from previous close."""
        return self.current_price - self.previous_close
    
    @property
    def price_change_pct(self) -> float:
        """Calculate percentage price change."""
        if self.previous_close == 0:
            return 0.0
        return (self.price_change / self.previous_close) * 100


class DataSourceError(Exception):
    """Custom exception for data source errors."""
    pass


class YFinanceSource:
    """YFinance data source with async wrapper."""
    
    def __init__(self):
        self.name = "yfinance"
        self.executor = ThreadPoolExecutor(max_workers=config.performance.max_concurrent_requests)
    
    async def fetch_data(self, symbol: str, period: str = "1y") -> Optional[MarketData]:
        """Fetch data for a single symbol using yfinance."""
        try:
            # Run yfinance in thread pool since it's synchronous
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(
                self.executor, 
                lambda: yf.Ticker(symbol)
            )
            
            # Get historical data
            hist = await loop.run_in_executor(
                self.executor,
                lambda: ticker.history(period=period)
            )
            
            if hist.empty:
                return None
            
            # Get current info
            info = await loop.run_in_executor(
                self.executor,
                lambda: ticker.info
            )
            
            current_price = info.get('currentPrice', hist['Close'].iloc[-1])
            previous_close = info.get('previousClose', hist['Close'].iloc[-2] if len(hist) > 1 else current_price)
            volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0
            
            return MarketData(
                symbol=symbol,
                prices=hist,
                current_price=float(current_price),
                previous_close=float(previous_close),
                volume=volume,
                last_updated=datetime.now(),
                data_source=self.name
            )
            
        except Exception as e:
            logger.warning(f"YFinance failed for {symbol}: {str(e)}")
            return None


class AlphaVantageSource:
    """Alpha Vantage data source (requires API key)."""
    
    def __init__(self):
        self.name = "alpha_vantage"
        self.api_key = None  # Set from environment
        self.base_url = "https://www.alphavantage.co/query"
    
    async def fetch_data(self, symbol: str, period: str = "1y") -> Optional[MarketData]:
        """Fetch data using Alpha Vantage API."""
        if not self.api_key:
            logger.debug("Alpha Vantage API key not configured, skipping")
            return None
        
        try:
            # For now, return None since we don't have API key
            # This would be implemented with actual API calls
            logger.debug(f"Alpha Vantage source not implemented for {symbol}")
            return None
            
        except Exception as e:
            logger.warning(f"Alpha Vantage failed for {symbol}: {str(e)}")
            return None


class CacheManager:
    """Simple disk cache for market data."""
    
    def __init__(self, cache_dir: str = "./cache", ttl_seconds: int = 300):
        self.memory = Memory(cache_dir, verbose=0)
        self.ttl_seconds = ttl_seconds
        self._cache = {}
    
    def get_cached_data(self, symbol: str) -> Optional[MarketData]:
        """Get cached data if still valid."""
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H')}"  # Hour-level caching
        
        if cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.ttl_seconds:
                return data
            else:
                del self._cache[cache_key]
        
        return None
    
    def cache_data(self, symbol: str, data: MarketData) -> None:
        """Cache market data."""
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H')}"
        self._cache[cache_key] = (data, datetime.now())


class AsyncDataManager:
    """
    Main data manager with fallback sources and caching.
    Optimized for fast parallel fetching of multiple symbols.
    """
    
    def __init__(self):
        self.sources = [
            YFinanceSource(),
            AlphaVantageSource()
        ]
        self.cache = CacheManager(ttl_seconds=config.data_sources.cache_ttl_seconds)
        self.semaphore = asyncio.Semaphore(config.performance.max_concurrent_requests)
    
    async def fetch_symbol_data(self, symbol: str, period: str = "1y") -> Optional[MarketData]:
        """
        Fetch data for a single symbol with fallback sources.
        
        Args:
            symbol: Stock symbol to fetch
            period: Time period for historical data
            
        Returns:
            MarketData object or None if all sources fail
        """
        # Check cache first
        cached_data = self.cache.get_cached_data(symbol)
        if cached_data:
            logger.debug(f"Using cached data for {symbol}")
            return cached_data
        
        async with self.semaphore:  # Limit concurrent requests
            for source in self.sources:
                try:
                    logger.debug(f"Trying {source.name} for {symbol}")
                    data = await source.fetch_data(symbol, period)
                    
                    if data is not None:
                        logger.info(f"Successfully fetched {symbol} from {source.name}")
                        self.cache.cache_data(symbol, data)
                        return data
                        
                except Exception as e:
                    logger.warning(f"Source {source.name} failed for {symbol}: {str(e)}")
                    continue
            
            logger.error(f"All data sources failed for symbol {symbol}")
            return None
    
    async def fetch_portfolio_data(self, symbols: List[str], period: str = "1y") -> Dict[str, MarketData]:
        """
        Fetch data for multiple symbols in parallel.
        
        Args:
            symbols: List of symbols to fetch
            period: Time period for historical data
            
        Returns:
            Dictionary mapping symbols to MarketData objects
        """
        start_time = time.time()
        logger.info(f"Starting parallel fetch for {len(symbols)} symbols")
        
        # Create tasks for parallel execution
        tasks = [
            self.fetch_symbol_data(symbol, period) 
            for symbol in symbols
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        portfolio_data = {}
        successful_fetches = 0
        
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Exception fetching {symbol}: {str(result)}")
            elif result is not None:
                portfolio_data[symbol] = result
                successful_fetches += 1
            else:
                logger.warning(f"No data available for {symbol}")
        
        fetch_time = time.time() - start_time
        logger.info(f"Completed portfolio fetch: {successful_fetches}/{len(symbols)} symbols in {fetch_time:.2f}s")
        
        return portfolio_data
    
    async def get_current_quotes(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current price quotes for symbols (lightweight version).
        
        Args:
            symbols: List of symbols
            
        Returns:
            Dictionary mapping symbols to current prices
        """
        portfolio_data = await self.fetch_portfolio_data(symbols, period="1d")
        
        quotes = {}
        for symbol, data in portfolio_data.items():
            quotes[symbol] = data.current_price
        
        return quotes
    
    def get_data_quality_report(self, portfolio_data: Dict[str, MarketData]) -> Dict[str, Any]:
        """Generate a data quality report for the fetched data."""
        if not portfolio_data:
            return {"status": "no_data", "message": "No data available"}
        
        total_symbols = len(portfolio_data)
        sources_used = {}
        
        for data in portfolio_data.values():
            source = data.data_source
            sources_used[source] = sources_used.get(source, 0) + 1
        
        # Check data freshness
        now = datetime.now()
        stale_data = []
        
        for symbol, data in portfolio_data.items():
            if (now - data.last_updated).seconds > 3600:  # 1 hour
                stale_data.append(symbol)
        
        return {
            "status": "success",
            "total_symbols": total_symbols,
            "sources_used": sources_used,
            "stale_data_symbols": stale_data,
            "data_quality_score": max(0, 100 - len(stale_data) * 10)  # Simple scoring
        }
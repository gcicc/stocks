"""
Historical data provider for backtesting with multiple data sources.
Supports Yahoo Finance, Alpha Vantage, and CSV files.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path
import requests
import time
from dataclasses import dataclass

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    yf = None

from utils.config import config

logger = logging.getLogger(__name__)


@dataclass
class DataRequest:
    """Request specification for historical data."""
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    interval: str = "1d"  # '1d', '1wk', '1mo'
    data_source: str = "yahoo"  # 'yahoo', 'alphavantage', 'csv'
    csv_directory: Optional[str] = None


class DataProvider:
    """
    Historical data provider supporting multiple data sources.
    
    Data Sources:
    - Yahoo Finance (free, reliable)
    - Alpha Vantage (API key required)
    - CSV files (local data)
    """
    
    def __init__(self, 
                 alpha_vantage_api_key: Optional[str] = None,
                 cache_directory: Optional[str] = None):
        self.alpha_vantage_api_key = alpha_vantage_api_key
        self.cache_directory = Path(cache_directory) if cache_directory else None
        
        if self.cache_directory:
            self.cache_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Data provider initialized with sources: "
                   f"yahoo={'available' if HAS_YFINANCE else 'unavailable'}, "
                   f"alphavantage={'configured' if alpha_vantage_api_key else 'not configured'}, "
                   f"cache={'enabled' if cache_directory else 'disabled'}")
    
    def fetch_historical_data(self, request: DataRequest) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols.
        
        Args:
            request: Data request specification
            
        Returns:
            Dictionary mapping symbols to OHLCV DataFrames with DatetimeIndex
        """
        logger.info(f"Fetching historical data for {len(request.symbols)} symbols")
        logger.info(f"Date range: {request.start_date} to {request.end_date}")
        logger.info(f"Data source: {request.data_source}")
        
        if request.data_source == "yahoo":
            return self._fetch_yahoo_finance_data(request)
        elif request.data_source == "alphavantage":
            return self._fetch_alphavantage_data(request)
        elif request.data_source == "csv":
            return self._fetch_csv_data(request)
        else:
            raise ValueError(f"Unsupported data source: {request.data_source}")
    
    def _fetch_yahoo_finance_data(self, request: DataRequest) -> Dict[str, pd.DataFrame]:
        """Fetch data from Yahoo Finance."""
        
        if not HAS_YFINANCE:
            raise ImportError("yfinance not available. Install with: pip install yfinance")
        
        data = {}
        failed_symbols = []
        
        for i, symbol in enumerate(request.symbols):
            try:
                logger.debug(f"Fetching {symbol} ({i+1}/{len(request.symbols)})")
                
                # Check cache first
                cached_data = self._load_from_cache(symbol, request)
                if cached_data is not None:
                    data[symbol] = cached_data
                    logger.debug(f"Loaded {symbol} from cache")
                    continue
                
                # Fetch from Yahoo Finance
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=request.start_date,
                    end=request.end_date,
                    interval=request.interval,
                    auto_adjust=True,
                    prepost=False
                )
                
                if df.empty:
                    logger.warning(f"No data available for {symbol}")
                    failed_symbols.append(symbol)
                    continue
                
                # Standardize column names
                df = self._standardize_dataframe(df)
                
                # Validate data quality
                if self._validate_data_quality(df, symbol):
                    data[symbol] = df
                    
                    # Cache the data
                    self._save_to_cache(symbol, df, request)
                    
                    logger.debug(f"Successfully fetched {len(df)} records for {symbol}")
                else:
                    failed_symbols.append(symbol)
                
                # Rate limiting
                time.sleep(0.1)  # Be respectful to Yahoo Finance
                
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
                failed_symbols.append(symbol)
                continue
        
        if failed_symbols:
            logger.warning(f"Failed to fetch data for {len(failed_symbols)} symbols: {failed_symbols}")
        
        logger.info(f"Successfully fetched data for {len(data)}/{len(request.symbols)} symbols")
        return data
    
    def _fetch_alphavantage_data(self, request: DataRequest) -> Dict[str, pd.DataFrame]:
        """Fetch data from Alpha Vantage API."""
        
        if not self.alpha_vantage_api_key:
            raise ValueError("Alpha Vantage API key not configured")
        
        data = {}
        failed_symbols = []
        
        # Alpha Vantage has strict rate limits
        api_calls_per_minute = 5
        delay_between_calls = 60.0 / api_calls_per_minute
        
        for i, symbol in enumerate(request.symbols):
            try:
                logger.debug(f"Fetching {symbol} from Alpha Vantage ({i+1}/{len(request.symbols)})")
                
                # Check cache first
                cached_data = self._load_from_cache(symbol, request)
                if cached_data is not None:
                    data[symbol] = cached_data
                    logger.debug(f"Loaded {symbol} from cache")
                    continue
                
                # Prepare API request
                function_map = {
                    "1d": "TIME_SERIES_DAILY_ADJUSTED",
                    "1wk": "TIME_SERIES_WEEKLY_ADJUSTED",
                    "1mo": "TIME_SERIES_MONTHLY_ADJUSTED"
                }
                
                function = function_map.get(request.interval, "TIME_SERIES_DAILY_ADJUSTED")
                
                url = "https://www.alphavantage.co/query"
                params = {
                    'function': function,
                    'symbol': symbol,
                    'outputsize': 'full',
                    'apikey': self.alpha_vantage_api_key
                }
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                json_data = response.json()
                
                # Parse response
                df = self._parse_alphavantage_response(json_data, symbol)
                
                if df is not None and not df.empty:
                    # Filter date range
                    df = df[(df.index >= request.start_date) & (df.index <= request.end_date)]
                    
                    if self._validate_data_quality(df, symbol):
                        data[symbol] = df
                        self._save_to_cache(symbol, df, request)
                        logger.debug(f"Successfully fetched {len(df)} records for {symbol}")
                    else:
                        failed_symbols.append(symbol)
                else:
                    failed_symbols.append(symbol)
                
                # Rate limiting for Alpha Vantage
                if i < len(request.symbols) - 1:
                    time.sleep(delay_between_calls)
                
            except Exception as e:
                logger.error(f"Failed to fetch {symbol} from Alpha Vantage: {str(e)}")
                failed_symbols.append(symbol)
                continue
        
        if failed_symbols:
            logger.warning(f"Failed to fetch data for {len(failed_symbols)} symbols: {failed_symbols}")
        
        logger.info(f"Successfully fetched data for {len(data)}/{len(request.symbols)} symbols")
        return data
    
    def _fetch_csv_data(self, request: DataRequest) -> Dict[str, pd.DataFrame]:
        """Load data from CSV files."""
        
        if not request.csv_directory:
            raise ValueError("CSV directory not specified in request")
        
        csv_dir = Path(request.csv_directory)
        if not csv_dir.exists():
            raise FileNotFoundError(f"CSV directory not found: {csv_dir}")
        
        data = {}
        failed_symbols = []
        
        for symbol in request.symbols:
            try:
                # Look for CSV file (try multiple naming conventions)
                possible_files = [
                    csv_dir / f"{symbol}.csv",
                    csv_dir / f"{symbol.upper()}.csv",
                    csv_dir / f"{symbol.lower()}.csv",
                    csv_dir / f"{symbol}_historical.csv"
                ]
                
                csv_file = None
                for file_path in possible_files:
                    if file_path.exists():
                        csv_file = file_path
                        break
                
                if not csv_file:
                    logger.warning(f"CSV file not found for {symbol}")
                    failed_symbols.append(symbol)
                    continue
                
                # Load CSV
                df = pd.read_csv(csv_file)
                
                # Try to parse date column
                date_columns = ['Date', 'date', 'DATE', 'Timestamp', 'timestamp']
                date_col = None
                for col in date_columns:
                    if col in df.columns:
                        date_col = col
                        break
                
                if date_col is None:
                    logger.error(f"No date column found in {csv_file}")
                    failed_symbols.append(symbol)
                    continue
                
                # Set date index
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
                df.sort_index(inplace=True)
                
                # Standardize columns
                df = self._standardize_dataframe(df)
                
                # Filter date range
                df = df[(df.index >= request.start_date) & (df.index <= request.end_date)]
                
                if self._validate_data_quality(df, symbol):
                    data[symbol] = df
                    logger.debug(f"Successfully loaded {len(df)} records for {symbol}")
                else:
                    failed_symbols.append(symbol)
                
            except Exception as e:
                logger.error(f"Failed to load CSV data for {symbol}: {str(e)}")
                failed_symbols.append(symbol)
                continue
        
        if failed_symbols:
            logger.warning(f"Failed to load data for {len(failed_symbols)} symbols: {failed_symbols}")
        
        logger.info(f"Successfully loaded data for {len(data)}/{len(request.symbols)} symbols")
        return data
    
    def _parse_alphavantage_response(self, json_data: dict, symbol: str) -> Optional[pd.DataFrame]:
        """Parse Alpha Vantage API response into DataFrame."""
        
        try:
            # Find the time series data key
            time_series_key = None
            for key in json_data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break
            
            if not time_series_key:
                logger.error(f"No time series data found for {symbol}")
                return None
            
            time_series = json_data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index', dtype=float)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            # Rename columns (Alpha Vantage uses numbered keys)
            column_mapping = {
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. adjusted close': 'Adj Close',
                '6. volume': 'Volume',
                '5. volume': 'Volume'
            }
            
            df.rename(columns=column_mapping, inplace=True)
            
            # Standardize
            df = self._standardize_dataframe(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse Alpha Vantage response for {symbol}: {str(e)}")
            return None
    
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame column names and data types."""
        
        # Column name mapping
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'adj close': 'Adj Close',
            'adjusted close': 'Adj Close',
            'volume': 'Volume',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Adj Close': 'Adj Close',
            'Volume': 'Volume'
        }
        
        # Apply mapping
        df.rename(columns=column_mapping, inplace=True)
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found")
        
        # Add Adj Close if not present (use Close as fallback)
        if 'Adj Close' not in df.columns:
            df['Adj Close'] = df['Close']
        
        # Add Volume if not present
        if 'Volume' not in df.columns:
            df['Volume'] = 0
        
        # Convert to numeric
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN in OHLC data
        df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
        
        return df
    
    def _validate_data_quality(self, df: pd.DataFrame, symbol: str) -> bool:
        """Validate data quality and completeness."""
        
        if df.empty:
            logger.warning(f"Empty dataset for {symbol}")
            return False
        
        # Check for minimum data points
        if len(df) < 30:
            logger.warning(f"Insufficient data points for {symbol} ({len(df)} records)")
            return False
        
        # Check for data integrity
        ohlc_columns = ['Open', 'High', 'Low', 'Close']
        
        # Check for negative prices
        for col in ohlc_columns:
            if (df[col] <= 0).any():
                logger.warning(f"Invalid prices (<=0) found in {col} for {symbol}")
                return False
        
        # Check High >= Low
        if (df['High'] < df['Low']).any():
            logger.warning(f"High < Low anomaly found for {symbol}")
            return False
        
        # Check for extreme price movements (>50% in one day)
        daily_returns = df['Close'].pct_change()
        extreme_moves = abs(daily_returns) > 0.5
        if extreme_moves.any():
            logger.warning(f"Extreme price movements detected for {symbol}")
            # Don't reject, just warn
        
        logger.debug(f"Data quality validation passed for {symbol}")
        return True
    
    def _load_from_cache(self, symbol: str, request: DataRequest) -> Optional[pd.DataFrame]:
        """Load data from cache if available and fresh."""
        
        if not self.cache_directory:
            return None
        
        cache_file = self.cache_directory / f"{symbol}_{request.data_source}_{request.interval}.parquet"
        
        if not cache_file.exists():
            return None
        
        try:
            # Check cache age
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            max_cache_age = timedelta(hours=24)  # Cache valid for 24 hours
            
            if cache_age > max_cache_age:
                logger.debug(f"Cache expired for {symbol}")
                return None
            
            # Load cached data
            df = pd.read_parquet(cache_file)
            
            # Check if cached data covers the requested date range
            if df.index.min() <= request.start_date and df.index.max() >= request.end_date:
                # Filter to requested range
                filtered_df = df[(df.index >= request.start_date) & (df.index <= request.end_date)]
                return filtered_df
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to load cache for {symbol}: {str(e)}")
            return None
    
    def _save_to_cache(self, symbol: str, df: pd.DataFrame, request: DataRequest):
        """Save data to cache."""
        
        if not self.cache_directory:
            return
        
        try:
            cache_file = self.cache_directory / f"{symbol}_{request.data_source}_{request.interval}.parquet"
            df.to_parquet(cache_file)
            logger.debug(f"Cached data for {symbol}")
            
        except Exception as e:
            logger.debug(f"Failed to cache data for {symbol}: {str(e)}")
    
    def get_available_symbols(self, data_source: str = "yahoo") -> List[str]:
        """Get list of available symbols for backtesting."""
        
        if data_source == "yahoo":
            # Common US stocks for backtesting
            return [
                # Tech
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX',
                # Finance
                'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C',
                # Healthcare
                'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'MDT',
                # Consumer
                'KO', 'PEP', 'PG', 'WMT', 'HD', 'MCD',
                # Industrial
                'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS',
                # Energy
                'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD',
                # ETFs
                'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA'
            ]
        else:
            return []
    
    def create_sample_dataset(self, 
                            symbols: Optional[List[str]] = None,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """Create a sample dataset for backtesting demonstrations."""
        
        if symbols is None:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365*2)  # 2 years
        
        if end_date is None:
            end_date = datetime.now() - timedelta(days=1)
        
        request = DataRequest(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            data_source="yahoo"
        )
        
        return self.fetch_historical_data(request)
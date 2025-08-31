"""
Natural Momentum Indicator calculator with TEMA smoothing and natural logarithm transformations.
Optimized for fast computation across multiple symbols.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    numba = None

from utils.config import config
from core.data_manager import MarketData

logger = logging.getLogger(__name__)


@dataclass
class MomentumResult:
    """Results from momentum calculation for a single symbol."""
    symbol: str
    momentum_values: pd.Series
    tema_values: pd.Series
    signal_line: pd.Series
    current_momentum: float
    momentum_direction: str  # 'bullish', 'bearish', 'neutral'
    strength: float  # 0.0 to 1.0
    
    @property
    def latest_signal(self) -> float:
        """Get the latest signal value."""
        return self.signal_line.iloc[-1] if not self.signal_line.empty else 0.0
    
    @property
    def is_above_zero(self) -> bool:
        """Check if current momentum is above zero line."""
        return self.current_momentum > 0


class NaturalMomentumCalculator:
    """
    Calculate Natural Momentum Indicator using:
    1. Natural logarithm price transformations
    2. TEMA (Triple Exponential Moving Average) smoothing
    3. Zero-line crossover signals
    """
    
    def __init__(self, tema_period: int = None, momentum_lookback: int = None):
        self.tema_period = tema_period or config.algorithm.tema_period
        self.momentum_lookback = momentum_lookback or config.algorithm.momentum_lookback
        
        # Enable Numba acceleration if available
        self.use_numba = HAS_NUMBA and config.performance.enable_numba_acceleration
        
        if self.use_numba:
            logger.info("Numba acceleration enabled for momentum calculations")
        else:
            logger.info("Using standard NumPy calculations (Numba not available)")
    
    def calculate_tema(self, prices: Union[pd.Series, np.ndarray], period: int = None) -> np.ndarray:
        """
        Calculate Triple Exponential Moving Average (TEMA).
        
        TEMA = 3*EMA1 - 3*EMA2 + EMA3
        where:
        - EMA1 = EMA of prices
        - EMA2 = EMA of EMA1  
        - EMA3 = EMA of EMA2
        
        Args:
            prices: Price series
            period: TEMA period (default from config)
            
        Returns:
            TEMA values as numpy array
        """
        if period is None:
            period = self.tema_period
        
        prices_array = np.array(prices) if not isinstance(prices, np.ndarray) else prices
        
        if self.use_numba:
            return self._calculate_tema_numba(prices_array, period)
        else:
            return self._calculate_tema_numpy(prices_array, period)
    
    def _calculate_tema_numpy(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate TEMA using NumPy (fallback when Numba not available)."""
        alpha = 2.0 / (period + 1)
        
        # First EMA
        ema1 = np.zeros_like(prices)
        ema1[0] = prices[0]
        for i in range(1, len(prices)):
            ema1[i] = alpha * prices[i] + (1 - alpha) * ema1[i-1]
        
        # Second EMA (EMA of EMA1)
        ema2 = np.zeros_like(ema1)
        ema2[0] = ema1[0]
        for i in range(1, len(ema1)):
            ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i-1]
        
        # Third EMA (EMA of EMA2)
        ema3 = np.zeros_like(ema2)
        ema3[0] = ema2[0]
        for i in range(1, len(ema2)):
            ema3[i] = alpha * ema2[i] + (1 - alpha) * ema3[i-1]
        
        # TEMA formula
        tema = 3 * ema1 - 3 * ema2 + ema3
        return tema
    
    def calculate_natural_momentum(self, market_data: MarketData) -> MomentumResult:
        """
        Calculate Natural Momentum for a single symbol.
        
        Algorithm:
        1. Apply natural log transformation to prices
        2. Calculate momentum using log price differences
        3. Smooth with TEMA
        4. Generate signal line
        
        Args:
            market_data: MarketData object with OHLCV data
            
        Returns:
            MomentumResult object with all calculations
        """
        try:
            prices = market_data.prices
            if prices.empty or len(prices) < self.momentum_lookback * 2:
                logger.warning(f"Insufficient data for {market_data.symbol}")
                return self._create_empty_result(market_data.symbol)
            
            # Use closing prices for calculations
            close_prices = prices['Close'].values
            
            # Step 1: Natural log transformation
            log_prices = np.log(close_prices)
            
            # Step 2: Calculate momentum using log differences
            momentum_raw = self._calculate_log_momentum(log_prices, self.momentum_lookback)
            
            # Step 3: Apply TEMA smoothing
            tema_smoothed = self.calculate_tema(momentum_raw, self.tema_period)
            
            # Step 4: Generate signal line (simpler smoothing for crossover detection)
            signal_period = max(3, self.tema_period // 3)  # Faster signal line
            signal_line = self.calculate_tema(tema_smoothed, signal_period)
            
            # Step 5: Calculate current momentum metrics
            current_momentum = tema_smoothed[-1] if len(tema_smoothed) > 0 else 0.0
            direction = self._determine_direction(tema_smoothed, signal_line)
            strength = self._calculate_strength(tema_smoothed, signal_line)
            
            # Create pandas Series with proper index
            index = prices.index
            
            return MomentumResult(
                symbol=market_data.symbol,
                momentum_values=pd.Series(momentum_raw, index=index[-len(momentum_raw):]),
                tema_values=pd.Series(tema_smoothed, index=index[-len(tema_smoothed):]),
                signal_line=pd.Series(signal_line, index=index[-len(signal_line):]),
                current_momentum=float(current_momentum),
                momentum_direction=direction,
                strength=float(strength)
            )
            
        except Exception as e:
            logger.error(f"Error calculating momentum for {market_data.symbol}: {str(e)}")
            return self._create_empty_result(market_data.symbol)
    
    def _calculate_log_momentum(self, log_prices: np.ndarray, lookback: int) -> np.ndarray:
        """Calculate momentum using natural log price differences."""
        if len(log_prices) < lookback + 1:
            return np.array([0.0])
        
        momentum = np.zeros(len(log_prices) - lookback)
        
        for i in range(lookback, len(log_prices)):
            # Momentum = current log price - log price N periods ago
            momentum[i - lookback] = log_prices[i] - log_prices[i - lookback]
        
        # Scale momentum for better visualization (multiply by 100)
        return momentum * 100
    
    def _determine_direction(self, tema_values: np.ndarray, signal_values: np.ndarray) -> str:
        """Determine momentum direction based on TEMA and signal line."""
        if len(tema_values) < 2 or len(signal_values) < 2:
            return 'neutral'
        
        current_tema = tema_values[-1]
        current_signal = signal_values[-1]
        
        # Bullish: TEMA above signal line and TEMA trending up
        if current_tema > current_signal and current_tema > tema_values[-2]:
            return 'bullish'
        # Bearish: TEMA below signal line and TEMA trending down
        elif current_tema < current_signal and current_tema < tema_values[-2]:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_strength(self, tema_values: np.ndarray, signal_values: np.ndarray) -> float:
        """Calculate momentum strength as a value between 0 and 1."""
        if len(tema_values) < 5:
            return 0.0
        
        # Calculate the distance between TEMA and signal line
        distance = abs(tema_values[-1] - signal_values[-1])
        
        # Calculate recent volatility for normalization
        recent_tema = tema_values[-10:] if len(tema_values) >= 10 else tema_values
        volatility = np.std(recent_tema)
        
        if volatility == 0:
            return 0.0
        
        # Normalize strength (0 to 1, with 1 being very strong)
        strength = min(1.0, distance / (2 * volatility))
        return strength
    
    def _create_empty_result(self, symbol: str) -> MomentumResult:
        """Create empty result for symbols with insufficient data."""
        empty_series = pd.Series([], dtype=float)
        
        return MomentumResult(
            symbol=symbol,
            momentum_values=empty_series,
            tema_values=empty_series,
            signal_line=empty_series,
            current_momentum=0.0,
            momentum_direction='neutral',
            strength=0.0
        )
    
    def calculate_portfolio_momentum(self, portfolio_data: Dict[str, MarketData]) -> Dict[str, MomentumResult]:
        """
        Calculate momentum for all symbols in a portfolio.
        
        Args:
            portfolio_data: Dictionary mapping symbols to MarketData
            
        Returns:
            Dictionary mapping symbols to MomentumResult objects
        """
        results = {}
        
        logger.info(f"Calculating momentum for {len(portfolio_data)} symbols")
        
        for symbol, market_data in portfolio_data.items():
            try:
                result = self.calculate_natural_momentum(market_data)
                results[symbol] = result
                logger.debug(f"Calculated momentum for {symbol}: {result.momentum_direction}")
                
            except Exception as e:
                logger.error(f"Failed to calculate momentum for {symbol}: {str(e)}")
                results[symbol] = self._create_empty_result(symbol)
        
        logger.info(f"Completed momentum calculations for {len(results)} symbols")
        return results


# Add Numba-optimized functions if available
if HAS_NUMBA:
    @numba.jit(nopython=True, parallel=False)
    def _calculate_tema_numba(prices: np.ndarray, period: int) -> np.ndarray:
        """Numba-optimized TEMA calculation."""
        alpha = 2.0 / (period + 1)
        
        # First EMA
        ema1 = np.zeros_like(prices)
        ema1[0] = prices[0]
        for i in range(1, len(prices)):
            ema1[i] = alpha * prices[i] + (1 - alpha) * ema1[i-1]
        
        # Second EMA
        ema2 = np.zeros_like(ema1)
        ema2[0] = ema1[0]
        for i in range(1, len(ema1)):
            ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i-1]
        
        # Third EMA
        ema3 = np.zeros_like(ema2)
        ema3[0] = ema2[0]
        for i in range(1, len(ema2)):
            ema3[i] = alpha * ema2[i] + (1 - alpha) * ema3[i-1]
        
        # TEMA
        tema = 3 * ema1 - 3 * ema2 + ema3
        return tema
    
    # Bind the numba function to the class
    NaturalMomentumCalculator._calculate_tema_numba = staticmethod(_calculate_tema_numba)
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
    
    # Multi-timeframe enhancements
    tema_short: Optional[pd.Series] = None  # Short-term TEMA (5-8 periods)
    tema_medium: Optional[pd.Series] = None  # Medium-term TEMA (14 periods)
    tema_long: Optional[pd.Series] = None  # Long-term TEMA (21-34 periods)
    timeframe_consensus: Optional[float] = None  # -1 to +1, agreement across timeframes
    trend_strength: Optional[str] = None  # 'weak', 'moderate', 'strong'
    
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
    
    def __init__(self, tema_period: int = None, momentum_lookback: int = None, enable_multi_timeframe: bool = True):
        self.tema_period = tema_period or config.algorithm.tema_period
        self.momentum_lookback = momentum_lookback or config.algorithm.momentum_lookback
        
        # Multi-timeframe configuration
        self.enable_multi_timeframe = enable_multi_timeframe
        self.tema_short = max(5, self.tema_period // 2)  # Short-term: half of main period
        self.tema_medium = self.tema_period  # Medium-term: main period (14)
        self.tema_long = self.tema_period * 2  # Long-term: double main period
        
        # Enable Numba acceleration if available
        self.use_numba = HAS_NUMBA and config.performance.enable_numba_acceleration
        
        if self.use_numba:
            logger.info("Numba acceleration enabled for momentum calculations")
        else:
            logger.info("Using standard NumPy calculations (Numba not available)")
        
        logger.info(f"Multi-timeframe TEMA periods: short={self.tema_short}, medium={self.tema_medium}, long={self.tema_long}")
    
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
            
            # Step 3: Apply TEMA smoothing (multi-timeframe)
            if self.enable_multi_timeframe:
                tema_short = self.calculate_tema(momentum_raw, self.tema_short)
                tema_medium = self.calculate_tema(momentum_raw, self.tema_medium)
                tema_long = self.calculate_tema(momentum_raw, self.tema_long)
                
                # Use medium-term TEMA as primary for backward compatibility
                tema_smoothed = tema_medium
            else:
                tema_smoothed = self.calculate_tema(momentum_raw, self.tema_period)
                tema_short = tema_medium = tema_long = None
            
            # Step 4: Generate signal line (simpler smoothing for crossover detection)
            signal_period = max(3, self.tema_period // 3)  # Faster signal line
            signal_line = self.calculate_tema(tema_smoothed, signal_period)
            
            # Step 5: Calculate multi-timeframe consensus and enhanced metrics
            if self.enable_multi_timeframe and all(t is not None for t in [tema_short, tema_medium, tema_long]):
                consensus = self._calculate_timeframe_consensus(tema_short, tema_medium, tema_long)
                trend_strength = self._calculate_trend_strength(tema_short, tema_medium, tema_long)
            else:
                consensus = None
                trend_strength = None
            
            # Step 6: Calculate current momentum metrics (enhanced with consensus)
            current_momentum = tema_smoothed[-1] if len(tema_smoothed) > 0 else 0.0
            direction = self._determine_direction_enhanced(tema_smoothed, signal_line, consensus)
            strength = self._calculate_strength_enhanced(tema_smoothed, signal_line, consensus)
            
            # Create pandas Series with proper index
            index = prices.index
            
            # Create result with multi-timeframe data
            return MomentumResult(
                symbol=market_data.symbol,
                momentum_values=pd.Series(momentum_raw, index=index[-len(momentum_raw):]),
                tema_values=pd.Series(tema_smoothed, index=index[-len(tema_smoothed):]),
                signal_line=pd.Series(signal_line, index=index[-len(signal_line):]),
                current_momentum=float(current_momentum),
                momentum_direction=direction,
                strength=float(strength),
                # Multi-timeframe enhancements
                tema_short=pd.Series(tema_short, index=index[-len(tema_short):]) if tema_short is not None else None,
                tema_medium=pd.Series(tema_medium, index=index[-len(tema_medium):]) if tema_medium is not None else None,
                tema_long=pd.Series(tema_long, index=index[-len(tema_long):]) if tema_long is not None else None,
                timeframe_consensus=consensus,
                trend_strength=trend_strength
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
    
    def _calculate_timeframe_consensus(self, tema_short: np.ndarray, tema_medium: np.ndarray, tema_long: np.ndarray) -> float:
        """Calculate consensus across multiple timeframes (-1 to +1).
        
        Returns:
            -1.0: All timeframes bearish
            0.0: Mixed signals
            +1.0: All timeframes bullish
        """
        if len(tema_short) < 2 or len(tema_medium) < 2 or len(tema_long) < 2:
            return 0.0
        
        # Check direction for each timeframe (positive = uptrend, negative = downtrend)
        short_direction = 1 if tema_short[-1] > tema_short[-2] else -1
        medium_direction = 1 if tema_medium[-1] > tema_medium[-2] else -1
        long_direction = 1 if tema_long[-1] > tema_long[-2] else -1
        
        # Check zero-line position for each timeframe
        short_position = 1 if tema_short[-1] > 0 else -1
        medium_position = 1 if tema_medium[-1] > 0 else -1
        long_position = 1 if tema_long[-1] > 0 else -1
        
        # Calculate weighted consensus (direction + position)
        direction_consensus = (short_direction + medium_direction + long_direction) / 3.0
        position_consensus = (short_position + medium_position + long_position) / 3.0
        
        # Combine direction and position (50/50 weighting)
        total_consensus = (direction_consensus + position_consensus) / 2.0
        
        return float(np.clip(total_consensus, -1.0, 1.0))
    
    def _calculate_trend_strength(self, tema_short: np.ndarray, tema_medium: np.ndarray, tema_long: np.ndarray) -> str:
        """Calculate overall trend strength based on timeframe alignment.
        
        Returns:
            'weak': Conflicting signals across timeframes
            'moderate': Some alignment but mixed signals
            'strong': Strong alignment across all timeframes
        """
        consensus = self._calculate_timeframe_consensus(tema_short, tema_medium, tema_long)
        
        if abs(consensus) >= 0.8:
            return 'strong'
        elif abs(consensus) >= 0.4:
            return 'moderate'
        else:
            return 'weak'
    
    def _determine_direction_enhanced(self, tema_values: np.ndarray, signal_values: np.ndarray, consensus: Optional[float]) -> str:
        """Enhanced direction determination using multi-timeframe consensus."""
        # Start with basic direction logic
        basic_direction = self._determine_direction(tema_values, signal_values)
        
        # If no consensus available, return basic direction
        if consensus is None:
            return basic_direction
        
        # Use consensus to refine direction
        if consensus > 0.5:
            return 'bullish'
        elif consensus < -0.5:
            return 'bearish'
        elif abs(consensus) < 0.2:
            return 'neutral'
        else:
            # Moderate consensus - stick with basic direction but add nuance
            if basic_direction == 'neutral':
                return 'bullish' if consensus > 0 else 'bearish'
            return basic_direction
    
    def _calculate_strength_enhanced(self, tema_values: np.ndarray, signal_values: np.ndarray, consensus: Optional[float]) -> float:
        """Enhanced strength calculation using multi-timeframe consensus."""
        # Start with basic strength calculation
        basic_strength = self._calculate_strength(tema_values, signal_values)
        
        # If no consensus available, return basic strength
        if consensus is None:
            return basic_strength
        
        # Enhance strength with consensus factor
        consensus_factor = abs(consensus)  # 0 to 1
        
        # Weighted combination: 70% basic strength + 30% consensus
        enhanced_strength = (0.7 * basic_strength) + (0.3 * consensus_factor)
        
        return min(1.0, enhanced_strength)
    
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
            strength=0.0,
            # Multi-timeframe fields for empty result
            tema_short=None,
            tema_medium=None,
            tema_long=None,
            timeframe_consensus=None,
            trend_strength=None
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
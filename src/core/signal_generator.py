"""
Signal generator for Buy/Hold/Sell recommendations based on Natural Momentum indicators.
Includes confidence scoring and risk assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from utils.config import config
from core.momentum_calculator import MomentumResult
from core.data_manager import MarketData

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal types for trading recommendations."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradingSignal:
    """Trading signal with confidence and risk metrics."""
    symbol: str
    signal: SignalType
    confidence: float  # 0.0 to 1.0
    strength: float   # 0.0 to 1.0 (momentum strength)
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_level: str = "MEDIUM"  # LOW, MEDIUM, HIGH
    reason: str = ""  # Human-readable explanation
    
    @property
    def signal_color(self) -> str:
        """Get color code for UI display."""
        if self.signal == SignalType.BUY:
            return "green"
        elif self.signal == SignalType.SELL:
            return "red"
        else:
            return "gray"
    
    @property
    def confidence_text(self) -> str:
        """Get confidence as text description."""
        if self.confidence >= 0.8:
            return "Very High"
        elif self.confidence >= 0.6:
            return "High"
        elif self.confidence >= 0.4:
            return "Medium"
        elif self.confidence >= 0.2:
            return "Low"
        else:
            return "Very Low"


class SignalGenerator:
    """
    Generate trading signals based on Natural Momentum calculations.
    
    Signal Logic:
    - BUY: Momentum crosses above zero with good strength and confidence
    - SELL: Momentum crosses below zero with good strength and confidence  
    - HOLD: Unclear signals or low confidence
    """
    
    def __init__(self, 
                 signal_threshold: float = None,
                 confidence_threshold: float = None,
                 strength_threshold: float = 0.15,  # Lowered from 0.3 to 0.15
                 enable_adaptive_thresholds: bool = True,
                 volatility_multiplier: float = 2.0,
                 enable_signal_confirmation: bool = True,
                 confirmation_periods: int = 3,
                 min_confirmations: int = 2,
                 enable_divergence_detection: bool = True,
                 divergence_lookback: int = 10,
                 backtesting_mode: bool = False):
        
        self.base_signal_threshold = signal_threshold or config.algorithm.signal_threshold
        self.confidence_threshold = confidence_threshold or config.algorithm.confidence_threshold
        self.strength_threshold = strength_threshold
        
        # Adaptive threshold configuration
        self.enable_adaptive_thresholds = enable_adaptive_thresholds
        self.volatility_multiplier = volatility_multiplier  # How much volatility affects thresholds
        
        # Signal confirmation configuration
        self.enable_signal_confirmation = enable_signal_confirmation
        self.confirmation_periods = confirmation_periods  # Look back N periods for confirmation
        self.min_confirmations = min_confirmations  # Minimum confirmations required
        
        # Divergence detection configuration
        self.enable_divergence_detection = enable_divergence_detection
        self.divergence_lookback = divergence_lookback  # Periods to analyze for divergences
        
        # Backtesting mode configuration
        self.backtesting_mode = backtesting_mode
        
        logger.info(f"Signal generator initialized with thresholds: "
                   f"base_signal={self.base_signal_threshold}, confidence={self.confidence_threshold}")
        logger.info(f"Adaptive thresholds: {'enabled' if enable_adaptive_thresholds else 'disabled'}")
        logger.info(f"Signal confirmation: {'enabled' if enable_signal_confirmation else 'disabled'} "
                   f"(periods={confirmation_periods}, min={min_confirmations})")
        logger.info(f"Divergence detection: {'enabled' if enable_divergence_detection else 'disabled'} "
                   f"(lookback={divergence_lookback})")
        logger.info(f"Backtesting mode: {'enabled' if backtesting_mode else 'disabled'}")
    
    def generate_signal(self, 
                       momentum_result: MomentumResult, 
                       market_data: MarketData) -> TradingSignal:
        """
        Generate trading signal for a single symbol.
        
        Args:
            momentum_result: Calculated momentum indicators
            market_data: Current market data
            
        Returns:
            TradingSignal with recommendation
        """
        try:
            # Check if we have enough data
            if (momentum_result.tema_values.empty or 
                len(momentum_result.tema_values) < 5):
                return self._create_hold_signal(
                    momentum_result.symbol, 
                    "Insufficient data for signal generation"
                )
            
            # Get current values
            current_momentum = momentum_result.current_momentum
            momentum_strength = momentum_result.strength
            direction = momentum_result.momentum_direction
            
            # Calculate signal confidence
            confidence = self._calculate_confidence(momentum_result, market_data)
            
            # Calculate adaptive threshold based on market conditions
            adaptive_threshold = self._calculate_adaptive_threshold(
                momentum_result, market_data
            ) if self.enable_adaptive_thresholds else self.base_signal_threshold
            
            # Generate base signal from momentum crossover with adaptive threshold
            base_signal = self._determine_base_signal(
                current_momentum, 
                momentum_result.tema_values,
                momentum_result.signal_line,
                adaptive_threshold
            )
            
            # Apply signal confirmation if enabled
            if self.enable_signal_confirmation:
                confirmed_signal = self._apply_signal_confirmation(
                    base_signal, momentum_result, market_data, adaptive_threshold
                )
            else:
                confirmed_signal = base_signal
            
            # Apply filters and confidence checks
            final_signal = self._apply_signal_filters(
                confirmed_signal, confidence, momentum_strength, direction
            )
            
            # Calculate price targets and risk metrics
            price_target, stop_loss = self._calculate_targets(
                market_data.current_price, 
                final_signal,
                momentum_result
            )
            
            # Detect momentum divergences if enabled
            divergence_info = None
            if self.enable_divergence_detection:
                divergence_info = self._detect_momentum_divergence(
                    momentum_result, market_data
                )
                
                # Adjust signal based on divergence
                if divergence_info:
                    final_signal = self._apply_divergence_adjustment(
                        final_signal, divergence_info
                    )
            
            # Determine risk level (consider divergence)
            risk_level = self._assess_risk_level(momentum_result, market_data, divergence_info)
            
            # Create explanation with all enhancement info
            if self.enable_signal_confirmation and base_signal != SignalType.HOLD:
                if confirmed_signal == base_signal:
                    confirmation_info = "confirmed"
                elif confirmed_signal == SignalType.HOLD:
                    confirmation_info = "downgraded"
                else:
                    confirmation_info = None
            else:
                confirmation_info = None
            reason = self._generate_signal_reason(
                final_signal, current_momentum, direction, confidence, adaptive_threshold, confirmation_info, divergence_info
            )
            
            return TradingSignal(
                symbol=momentum_result.symbol,
                signal=final_signal,
                confidence=confidence,
                strength=momentum_strength,
                price_target=price_target,
                stop_loss=stop_loss,
                risk_level=risk_level,
                reason=reason
            )
            
        except Exception as e:
            logger.error(f"Error generating signal for {momentum_result.symbol}: {str(e)}")
            return self._create_hold_signal(
                momentum_result.symbol, 
                f"Error in signal calculation: {str(e)}"
            )
    
    def _determine_base_signal(self, 
                              current_momentum: float,
                              tema_values: pd.Series,
                              signal_line: pd.Series,
                              threshold: float = None) -> SignalType:
        """Determine base signal from momentum indicators with adaptive threshold."""
        
        # Use provided threshold or fallback to base threshold
        signal_threshold = threshold if threshold is not None else self.base_signal_threshold
        
        # Check adaptive threshold crossover
        if current_momentum > signal_threshold:
            # Above zero line - potential buy
            if len(tema_values) >= 2:
                # Check if momentum is strengthening
                if tema_values.iloc[-1] > tema_values.iloc[-2]:
                    return SignalType.BUY
            else:
                return SignalType.BUY
                
        elif current_momentum < -signal_threshold:
            # Below zero line - potential sell
            if len(tema_values) >= 2:
                # Check if momentum is weakening
                if tema_values.iloc[-1] < tema_values.iloc[-2]:
                    return SignalType.SELL
            else:
                return SignalType.SELL
        
        return SignalType.HOLD
    
    def _calculate_adaptive_threshold(self, 
                                    momentum_result: MomentumResult, 
                                    market_data: MarketData) -> float:
        """Calculate adaptive signal threshold based on market volatility.
        
        Higher volatility = higher threshold (fewer signals, higher quality)
        Lower volatility = lower threshold (more signals, earlier entry)
        
        Args:
            momentum_result: Momentum calculation results
            market_data: Current market data
            
        Returns:
            Adaptive threshold value
        """
        try:
            # Calculate market volatility using multiple methods
            volatility_factors = []
            
            # Factor 1: Recent price volatility (20-day standard deviation)
            if len(market_data.prices) >= 20:
                recent_prices = market_data.prices['Close'].iloc[-20:]
                price_returns = recent_prices.pct_change().dropna()
                if len(price_returns) > 0:
                    price_volatility = price_returns.std()
                    # Normalize to 0-1 scale (typical daily volatility 0-5%)
                    normalized_price_vol = min(1.0, price_volatility * 20)  # Scale up
                    volatility_factors.append(normalized_price_vol)
            
            # Factor 2: Momentum volatility (stability of TEMA values)
            if len(momentum_result.tema_values) >= 10:
                recent_tema = momentum_result.tema_values.iloc[-10:]
                tema_volatility = recent_tema.std()
                # Normalize momentum volatility
                if tema_volatility > 0:
                    normalized_tema_vol = min(1.0, abs(tema_volatility) / 10.0)
                    volatility_factors.append(normalized_tema_vol)
            
            # Factor 3: Multi-timeframe consensus volatility (if available)
            if (hasattr(momentum_result, 'timeframe_consensus') and 
                momentum_result.timeframe_consensus is not None):
                # Lower consensus = higher uncertainty = higher volatility
                consensus = abs(momentum_result.timeframe_consensus)
                consensus_volatility = 1.0 - consensus  # Invert: low consensus = high volatility
                volatility_factors.append(consensus_volatility)
            
            # Calculate overall volatility score
            if volatility_factors:
                avg_volatility = sum(volatility_factors) / len(volatility_factors)
            else:
                avg_volatility = 0.5  # Default medium volatility
            
            # Apply volatility to threshold
            # High volatility -> higher threshold (more conservative)
            # Low volatility -> lower threshold (more sensitive)
            volatility_adjustment = avg_volatility * self.volatility_multiplier
            adaptive_threshold = self.base_signal_threshold + volatility_adjustment
            
            logger.debug(f"Adaptive threshold calculation: "
                        f"base={self.base_signal_threshold:.3f}, "
                        f"volatility={avg_volatility:.3f}, "
                        f"adjustment={volatility_adjustment:.3f}, "
                        f"adaptive={adaptive_threshold:.3f}")
            
            return adaptive_threshold
            
        except Exception as e:
            logger.warning(f"Error calculating adaptive threshold: {str(e)}")
            return self.base_signal_threshold  # Fallback to base threshold
    
    def _apply_signal_confirmation(self, 
                                 base_signal: SignalType,
                                 momentum_result: MomentumResult, 
                                 market_data: MarketData,
                                 threshold: float) -> SignalType:
        """Apply signal confirmation logic using multiple validation factors.
        
        Confirmation factors:
        1. Consecutive momentum direction consistency
        2. Multi-timeframe agreement (if available)
        3. Volume confirmation
        4. Price-momentum alignment
        
        Args:
            base_signal: Initial signal from momentum crossover
            momentum_result: Momentum calculation results  
            market_data: Market data for additional validation
            threshold: Current adaptive threshold
            
        Returns:
            Confirmed or downgraded signal
        """
        try:
            if base_signal == SignalType.HOLD:
                return base_signal  # No confirmation needed for HOLD
            
            confirmation_score = 0
            max_confirmations = 4  # Number of confirmation factors
            
            # Factor 1: Consecutive momentum direction consistency
            momentum_consistency = self._check_momentum_consistency(
                momentum_result.tema_values, base_signal
            )
            if momentum_consistency:
                confirmation_score += 1
                logger.debug(f"Confirmation +1: Momentum consistency")
            
            # Factor 2: Multi-timeframe agreement (if available)
            timeframe_agreement = self._check_timeframe_agreement(
                momentum_result, base_signal
            )
            if timeframe_agreement:
                confirmation_score += 1
                logger.debug(f"Confirmation +1: Timeframe agreement")
            
            # Factor 3: Volume confirmation
            volume_confirmation = self._check_volume_confirmation(
                market_data, base_signal
            )
            if volume_confirmation:
                confirmation_score += 1
                logger.debug(f"Confirmation +1: Volume confirmation")
            
            # Factor 4: Price-momentum alignment
            alignment_confirmation = self._check_price_momentum_alignment(
                momentum_result, market_data, base_signal
            )
            if alignment_confirmation:
                confirmation_score += 1
                logger.debug(f"Confirmation +1: Price-momentum alignment")
            
            # Determine final signal based on confirmation score
            confirmation_ratio = confirmation_score / max_confirmations
            min_ratio = self.min_confirmations / max_confirmations
            
            logger.debug(f"Signal confirmation: {confirmation_score}/{max_confirmations} "
                        f"({confirmation_ratio:.1%}), minimum required: {min_ratio:.1%}")
            
            if confirmation_ratio >= min_ratio:
                logger.debug(f"Signal CONFIRMED: {base_signal.value}")
                return base_signal
            else:
                logger.debug(f"Signal DOWNGRADED: {base_signal.value} -> HOLD (insufficient confirmation)")
                return SignalType.HOLD
                
        except Exception as e:
            logger.warning(f"Error in signal confirmation: {str(e)}")
            return base_signal  # Fallback to base signal if confirmation fails
    
    def _check_momentum_consistency(self, tema_values: pd.Series, signal: SignalType) -> bool:
        """Check if momentum direction is consistent over recent periods."""
        if len(tema_values) < self.confirmation_periods:
            return False
        
        recent_tema = tema_values.iloc[-self.confirmation_periods:]
        
        if signal == SignalType.BUY:
            # For BUY signal, check if momentum is generally increasing
            increasing_count = sum(1 for i in range(1, len(recent_tema)) 
                                 if recent_tema.iloc[i] > recent_tema.iloc[i-1])
            return increasing_count >= (self.confirmation_periods - 1) * 0.6
            
        elif signal == SignalType.SELL:
            # For SELL signal, check if momentum is generally decreasing
            decreasing_count = sum(1 for i in range(1, len(recent_tema)) 
                                 if recent_tema.iloc[i] < recent_tema.iloc[i-1])
            return decreasing_count >= (self.confirmation_periods - 1) * 0.6
        
        return False
    
    def _check_timeframe_agreement(self, momentum_result: MomentumResult, signal: SignalType) -> bool:
        """Check if multiple timeframes agree with the signal direction."""
        # Only available if multi-timeframe analysis is enabled
        if not hasattr(momentum_result, 'timeframe_consensus') or momentum_result.timeframe_consensus is None:
            return True  # Skip if not available (don't penalize)
        
        consensus = momentum_result.timeframe_consensus
        
        if signal == SignalType.BUY:
            return consensus > 0.3  # Bullish consensus
        elif signal == SignalType.SELL:
            return consensus < -0.3  # Bearish consensus
        
        return False
    
    def _check_volume_confirmation(self, market_data: MarketData, signal: SignalType) -> bool:
        """Check if volume supports the signal direction."""
        try:
            if len(market_data.prices) < 5:
                return True  # Skip if insufficient data
            
            # Calculate average volume over recent periods
            recent_volume = market_data.prices['Volume'].iloc[-5:]
            avg_volume = recent_volume.mean()
            current_volume = market_data.volume
            
            # Volume should be above average for strong signals
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Require at least 80% of average volume for confirmation
            return volume_ratio >= 0.8
            
        except Exception as e:
            logger.debug(f"Volume confirmation failed: {str(e)}")
            return True  # Don't penalize if volume data unavailable
    
    def _check_price_momentum_alignment(self, momentum_result: MomentumResult, 
                                       market_data: MarketData, signal: SignalType) -> bool:
        """Check if price movement aligns with momentum signal."""
        try:
            if len(market_data.prices) < 3:
                return True  # Skip if insufficient data
            
            # Calculate recent price change
            recent_prices = market_data.prices['Close'].iloc[-3:]
            price_change = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
            
            # Get momentum direction
            current_momentum = momentum_result.current_momentum
            
            if signal == SignalType.BUY:
                # For BUY signal, both price and momentum should be positive or momentum strongly positive
                return (price_change > -0.01 and current_momentum > 0.5) or current_momentum > 1.0
                
            elif signal == SignalType.SELL:
                # For SELL signal, both price and momentum should be negative or momentum strongly negative
                return (price_change < 0.01 and current_momentum < -0.5) or current_momentum < -1.0
            
            return False
            
        except Exception as e:
            logger.debug(f"Price-momentum alignment check failed: {str(e)}")
            return True  # Don't penalize if check fails
    
    def _detect_momentum_divergence(self, momentum_result: MomentumResult, 
                                   market_data: MarketData) -> Optional[dict]:
        """Detect momentum divergences that may signal trend reversals.
        
        Types of divergences:
        - Bullish divergence: Price makes lower lows, momentum makes higher lows
        - Bearish divergence: Price makes higher highs, momentum makes lower highs
        
        Args:
            momentum_result: Momentum calculation results
            market_data: Market data with price information
            
        Returns:
            Dictionary with divergence info or None if no divergence detected
        """
        try:
            if len(market_data.prices) < self.divergence_lookback:
                return None
            
            # Get recent price and momentum data
            recent_prices = market_data.prices['Close'].iloc[-self.divergence_lookback:]
            recent_momentum = momentum_result.tema_values.iloc[-self.divergence_lookback:] if len(momentum_result.tema_values) >= self.divergence_lookback else None
            
            if recent_momentum is None:
                return None
            
            # Find peaks and troughs in both price and momentum
            price_peaks = self._find_peaks_and_troughs(recent_prices.values)
            momentum_peaks = self._find_peaks_and_troughs(recent_momentum.values)
            
            # Analyze for divergences
            bullish_div = self._check_bullish_divergence(price_peaks, momentum_peaks)
            bearish_div = self._check_bearish_divergence(price_peaks, momentum_peaks)
            
            if bullish_div:
                strength = self._calculate_divergence_strength(bullish_div)
                logger.debug(f"Bullish divergence detected: strength={strength}")
                return {
                    'type': 'bullish',
                    'strength': strength,
                    'price_points': bullish_div['price_points'],
                    'momentum_points': bullish_div['momentum_points']
                }
            elif bearish_div:
                strength = self._calculate_divergence_strength(bearish_div)
                logger.debug(f"Bearish divergence detected: strength={strength}")
                return {
                    'type': 'bearish',
                    'strength': strength,
                    'price_points': bearish_div['price_points'],
                    'momentum_points': bearish_div['momentum_points']
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Error in divergence detection: {str(e)}")
            return None
    
    def _find_peaks_and_troughs(self, data: np.ndarray) -> dict:
        """Find peaks and troughs in a data series.
        
        Args:
            data: Array of values
            
        Returns:
            Dictionary with peaks and troughs information
        """
        if len(data) < 5:
            return {'peaks': [], 'troughs': []}
        
        peaks = []
        troughs = []
        
        # Simple peak/trough detection (local maxima/minima)
        for i in range(2, len(data) - 2):
            # Check for peak (local maximum)
            if (data[i] > data[i-1] and data[i] > data[i+1] and
                data[i] > data[i-2] and data[i] > data[i+2]):
                peaks.append({'index': i, 'value': data[i]})
            
            # Check for trough (local minimum)
            elif (data[i] < data[i-1] and data[i] < data[i+1] and
                  data[i] < data[i-2] and data[i] < data[i+2]):
                troughs.append({'index': i, 'value': data[i]})
        
        return {'peaks': peaks, 'troughs': troughs}
    
    def _check_bullish_divergence(self, price_peaks: dict, momentum_peaks: dict) -> Optional[dict]:
        """Check for bullish divergence (price lower lows, momentum higher lows)."""
        price_troughs = price_peaks['troughs']
        momentum_troughs = momentum_peaks['troughs']
        
        if len(price_troughs) < 2 or len(momentum_troughs) < 2:
            return None
        
        # Get the two most recent troughs
        recent_price_troughs = sorted(price_troughs, key=lambda x: x['index'])[-2:]
        recent_momentum_troughs = sorted(momentum_troughs, key=lambda x: x['index'])[-2:]
        
        if len(recent_price_troughs) < 2 or len(recent_momentum_troughs) < 2:
            return None
        
        # Check for bullish divergence pattern
        price_lower_low = recent_price_troughs[1]['value'] < recent_price_troughs[0]['value']
        momentum_higher_low = recent_momentum_troughs[1]['value'] > recent_momentum_troughs[0]['value']
        
        if price_lower_low and momentum_higher_low:
            return {
                'price_points': recent_price_troughs,
                'momentum_points': recent_momentum_troughs
            }
        
        return None
    
    def _check_bearish_divergence(self, price_peaks: dict, momentum_peaks: dict) -> Optional[dict]:
        """Check for bearish divergence (price higher highs, momentum lower highs)."""
        price_peaks_list = price_peaks['peaks']
        momentum_peaks_list = momentum_peaks['peaks']
        
        if len(price_peaks_list) < 2 or len(momentum_peaks_list) < 2:
            return None
        
        # Get the two most recent peaks
        recent_price_peaks = sorted(price_peaks_list, key=lambda x: x['index'])[-2:]
        recent_momentum_peaks = sorted(momentum_peaks_list, key=lambda x: x['index'])[-2:]
        
        if len(recent_price_peaks) < 2 or len(recent_momentum_peaks) < 2:
            return None
        
        # Check for bearish divergence pattern
        price_higher_high = recent_price_peaks[1]['value'] > recent_price_peaks[0]['value']
        momentum_lower_high = recent_momentum_peaks[1]['value'] < recent_momentum_peaks[0]['value']
        
        if price_higher_high and momentum_lower_high:
            return {
                'price_points': recent_price_peaks,
                'momentum_points': recent_momentum_peaks
            }
        
        return None
    
    def _calculate_divergence_strength(self, divergence_data: dict) -> str:
        """Calculate the strength of a divergence."""
        try:
            price_points = divergence_data['price_points']
            momentum_points = divergence_data['momentum_points']
            
            if len(price_points) < 2 or len(momentum_points) < 2:
                return 'weak'
            
            # Calculate the magnitude of price vs momentum changes
            price_change = abs(price_points[1]['value'] - price_points[0]['value']) / price_points[0]['value']
            momentum_change = abs(momentum_points[1]['value'] - momentum_points[0]['value'])
            
            # Strong divergence: significant price change with opposite momentum change
            if price_change > 0.05 and momentum_change > 0.5:  # 5% price change, significant momentum change
                return 'strong'
            elif price_change > 0.02 or momentum_change > 0.2:
                return 'moderate'
            else:
                return 'weak'
                
        except Exception:
            return 'weak'
    
    def _apply_divergence_adjustment(self, signal: SignalType, divergence_info: dict) -> SignalType:
        """Adjust signal based on detected divergence."""
        divergence_type = divergence_info.get('type')
        divergence_strength = divergence_info.get('strength')
        
        # Only apply strong adjustments for strong divergences
        if divergence_strength != 'strong':
            return signal
        
        # Strong bullish divergence suggests upward reversal
        if divergence_type == 'bullish':
            if signal == SignalType.SELL:
                logger.debug("Strong bullish divergence - downgrading SELL to HOLD")
                return SignalType.HOLD
        
        # Strong bearish divergence suggests downward reversal
        elif divergence_type == 'bearish':
            if signal == SignalType.BUY:
                logger.debug("Strong bearish divergence - downgrading BUY to HOLD")
                return SignalType.HOLD
        
        return signal
    
    def _calculate_confidence(self, 
                            momentum_result: MomentumResult, 
                            market_data: MarketData) -> float:
        """Calculate confidence score for the signal."""
        
        confidence_factors = []
        
        # Factor 1: Momentum strength (higher is better)
        confidence_factors.append(min(1.0, momentum_result.strength * 2))
        
        # Factor 2: Trend consistency (check last 5 periods)
        if len(momentum_result.tema_values) >= 5:
            recent_tema = momentum_result.tema_values.iloc[-5:].values
            trend_consistency = self._calculate_trend_consistency(recent_tema)
            confidence_factors.append(trend_consistency)
        
        # Factor 3: Volume confirmation (if available)
        if hasattr(market_data, 'volume') and market_data.volume > 0:
            # Higher volume generally increases confidence
            # This is a simplified approach - in practice you'd compare to average volume
            volume_factor = min(1.0, 0.7)  # Simplified for MVP
            confidence_factors.append(volume_factor)
        
        # Factor 4: Distance from threshold (higher distance = higher confidence)
        # Use adaptive threshold if available
        reference_threshold = 0.0  # Default zero line
        distance_factor = min(1.0, abs(momentum_result.current_momentum) / 2.0)
        confidence_factors.append(distance_factor)
        
        # Factor 5: Adaptive threshold factor (if enabled)
        if self.enable_adaptive_thresholds:
            # Higher threshold means more conservative, so crossing it should increase confidence
            try:
                adaptive_threshold = self._calculate_adaptive_threshold(momentum_result, market_data)
                if adaptive_threshold > self.base_signal_threshold:
                    # In high volatility, crossing threshold is more significant
                    threshold_boost = min(0.3, (adaptive_threshold - self.base_signal_threshold) * 0.1)
                    confidence_factors.append(0.7 + threshold_boost)  # Base + boost
            except Exception:
                pass  # Skip if calculation fails
        
        # Calculate weighted average
        if confidence_factors:
            return np.mean(confidence_factors)
        else:
            return 0.5  # Default moderate confidence
    
    def _calculate_trend_consistency(self, values: np.ndarray) -> float:
        """Calculate how consistent the trend is (0-1 scale)."""
        if len(values) < 2:
            return 0.5
        
        # Calculate consecutive same-direction moves
        differences = np.diff(values)
        if len(differences) == 0:
            return 0.5
        
        # Count how many consecutive moves are in the same direction
        positive_moves = np.sum(differences > 0)
        negative_moves = np.sum(differences < 0)
        total_moves = len(differences)
        
        # Consistency is the ratio of moves in the dominant direction
        dominant_direction_moves = max(positive_moves, negative_moves)
        consistency = dominant_direction_moves / total_moves if total_moves > 0 else 0.5
        
        return consistency
    
    def _apply_signal_filters(self, 
                            base_signal: SignalType,
                            confidence: float,
                            strength: float,
                            direction: str) -> SignalType:
        """Apply filters to refine the signal."""
        
        # Filter 1: Confidence threshold
        if confidence < self.confidence_threshold:
            return SignalType.HOLD
        
        # Filter 2: Strength threshold  
        if strength < self.strength_threshold:
            return SignalType.HOLD
        
        # Filter 3: Direction confirmation (skip in backtesting mode for more trades)
        if not self.backtesting_mode:
            if base_signal == SignalType.BUY and direction != 'bullish':
                return SignalType.HOLD
            elif base_signal == SignalType.SELL and direction != 'bearish':
                return SignalType.HOLD
        
        return base_signal
    
    def _calculate_targets(self, 
                          current_price: float,
                          signal: SignalType,
                          momentum_result: MomentumResult) -> Tuple[Optional[float], Optional[float]]:
        """Calculate price target and stop loss levels."""
        
        if signal == SignalType.HOLD:
            return None, None
        
        # Simple target calculation based on momentum strength
        # In practice, this would be more sophisticated
        momentum_strength = momentum_result.strength
        
        if signal == SignalType.BUY:
            # Target: 2-5% above current price based on strength
            target_pct = 0.02 + (momentum_strength * 0.03)  # 2-5%
            price_target = current_price * (1 + target_pct)
            
            # Stop loss: 1-3% below current price
            stop_pct = 0.01 + (momentum_strength * 0.02)  # 1-3%
            stop_loss = current_price * (1 - stop_pct)
            
        else:  # SELL
            # Target: 2-5% below current price
            target_pct = 0.02 + (momentum_strength * 0.03)
            price_target = current_price * (1 - target_pct)
            
            # Stop loss: 1-3% above current price
            stop_pct = 0.01 + (momentum_strength * 0.02)
            stop_loss = current_price * (1 + stop_pct)
        
        return round(price_target, 2), round(stop_loss, 2)
    
    def _assess_risk_level(self, 
                          momentum_result: MomentumResult, 
                          market_data: MarketData,
                          divergence_info: Optional[dict] = None) -> str:
        """Assess risk level for the trade."""
        
        risk_factors = []
        
        # Factor 1: Momentum strength (higher strength = lower risk)
        strength_risk = 1.0 - momentum_result.strength
        risk_factors.append(strength_risk)
        
        # Factor 2: Volatility (calculate from recent price data)
        if len(market_data.prices) >= 20:
            recent_prices = market_data.prices['Close'].iloc[-20:].values
            volatility = np.std(np.diff(recent_prices) / recent_prices[:-1])
            volatility_risk = min(1.0, volatility * 10)  # Scale volatility
            risk_factors.append(volatility_risk)
        
        # Factor 3: Divergence risk (if detected)
        if divergence_info:
            divergence_type = divergence_info.get('type')
            divergence_strength = divergence_info.get('strength')
            
            # Divergences increase uncertainty and risk
            if divergence_strength == 'strong':
                risk_factors.append(0.8)  # High risk for strong divergences
            elif divergence_strength == 'moderate':
                risk_factors.append(0.6)  # Medium risk for moderate divergences
            else:
                risk_factors.append(0.4)  # Low additional risk for weak divergences
        
        # Calculate overall risk score
        avg_risk = np.mean(risk_factors)
        
        if avg_risk < 0.3:
            return "LOW"
        elif avg_risk < 0.7:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _generate_signal_reason(self, 
                               signal: SignalType,
                               momentum: float,
                               direction: str,
                               confidence: float,
                               threshold: float = None,
                               confirmation_info: str = None,
                               divergence_info: dict = None) -> str:
        """Generate human-readable explanation for the signal with all enhancement info."""
        
        threshold_info = ""
        if threshold is not None and self.enable_adaptive_thresholds:
            if threshold != self.base_signal_threshold:
                volatility_level = "high" if threshold > self.base_signal_threshold else "low"
                threshold_info = f" (adaptive threshold: {threshold:.3f} for {volatility_level} volatility)"
        
        confirmation_text = ""
        if confirmation_info:
            confirmation_text = f" - {confirmation_info}"
        
        divergence_text = ""
        if divergence_info:
            div_type = divergence_info.get('type', '')
            div_strength = divergence_info.get('strength', '')
            if div_type:
                divergence_text = f" - {div_strength} {div_type} divergence detected"
        
        if signal == SignalType.BUY:
            return (f"Bullish momentum ({momentum:.3f}) with {direction} trend. "
                   f"Confidence: {confidence:.0%}{threshold_info}{confirmation_text}{divergence_text}")
        elif signal == SignalType.SELL:
            return (f"Bearish momentum ({momentum:.3f}) with {direction} trend. "
                   f"Confidence: {confidence:.0%}{threshold_info}{confirmation_text}{divergence_text}")
        else:
            reason_base = f"Neutral momentum ({momentum:.3f}) or low confidence ({confidence:.0%}). Hold current position"
            if confirmation_info == "downgraded":
                return f"{reason_base} (signal downgraded due to insufficient confirmation){threshold_info}{divergence_text}."
            else:
                return f"{reason_base}{threshold_info}{divergence_text}."
    
    def _create_hold_signal(self, symbol: str, reason: str) -> TradingSignal:
        """Create a HOLD signal with explanation."""
        return TradingSignal(
            symbol=symbol,
            signal=SignalType.HOLD,
            confidence=0.0,
            strength=0.0,
            reason=reason,
            risk_level="MEDIUM"
        )
    
    def generate_portfolio_signals(self, 
                                 momentum_results: Dict[str, MomentumResult],
                                 market_data: Dict[str, MarketData]) -> Dict[str, TradingSignal]:
        """
        Generate trading signals for an entire portfolio.
        
        Args:
            momentum_results: Dictionary of momentum calculations
            market_data: Dictionary of market data
            
        Returns:
            Dictionary mapping symbols to TradingSignal objects
        """
        signals = {}
        
        logger.info(f"Generating signals for {len(momentum_results)} symbols")
        
        for symbol, momentum_result in momentum_results.items():
            try:
                # Get corresponding market data
                symbol_market_data = market_data.get(symbol)
                if symbol_market_data is None:
                    logger.warning(f"No market data available for {symbol}")
                    signals[symbol] = self._create_hold_signal(
                        symbol, "No market data available"
                    )
                    continue
                
                # Generate signal
                signal = self.generate_signal(momentum_result, symbol_market_data)
                signals[symbol] = signal
                
                logger.debug(f"Generated {signal.signal.value} signal for {symbol} "
                           f"(confidence: {signal.confidence:.0%})")
                
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {str(e)}")
                signals[symbol] = self._create_hold_signal(
                    symbol, f"Signal generation error: {str(e)}"
                )
        
        # Log summary
        signal_counts = {}
        for signal in signals.values():
            signal_type = signal.signal.value
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
        
        logger.info(f"Signal summary: {signal_counts}")
        
        return signals
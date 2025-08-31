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
                 strength_threshold: float = 0.3):
        
        self.signal_threshold = signal_threshold or config.algorithm.signal_threshold
        self.confidence_threshold = confidence_threshold or config.algorithm.confidence_threshold
        self.strength_threshold = strength_threshold
        
        logger.info(f"Signal generator initialized with thresholds: "
                   f"signal={self.signal_threshold}, confidence={self.confidence_threshold}")
    
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
            
            # Generate base signal from momentum crossover
            base_signal = self._determine_base_signal(
                current_momentum, 
                momentum_result.tema_values,
                momentum_result.signal_line
            )
            
            # Apply filters and confidence checks
            final_signal = self._apply_signal_filters(
                base_signal, confidence, momentum_strength, direction
            )
            
            # Calculate price targets and risk metrics
            price_target, stop_loss = self._calculate_targets(
                market_data.current_price, 
                final_signal,
                momentum_result
            )
            
            # Determine risk level
            risk_level = self._assess_risk_level(momentum_result, market_data)
            
            # Create explanation
            reason = self._generate_signal_reason(
                final_signal, current_momentum, direction, confidence
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
                              signal_line: pd.Series) -> SignalType:
        """Determine base signal from momentum indicators."""
        
        # Check zero-line crossover
        if current_momentum > self.signal_threshold:
            # Above zero line - potential buy
            if len(tema_values) >= 2:
                # Check if momentum is strengthening
                if tema_values.iloc[-1] > tema_values.iloc[-2]:
                    return SignalType.BUY
            else:
                return SignalType.BUY
                
        elif current_momentum < -self.signal_threshold:
            # Below zero line - potential sell
            if len(tema_values) >= 2:
                # Check if momentum is weakening
                if tema_values.iloc[-1] < tema_values.iloc[-2]:
                    return SignalType.SELL
            else:
                return SignalType.SELL
        
        return SignalType.HOLD
    
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
        
        # Factor 4: Distance from zero line (higher distance = higher confidence)
        distance_factor = min(1.0, abs(momentum_result.current_momentum) / 2.0)
        confidence_factors.append(distance_factor)
        
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
        
        # Filter 3: Direction confirmation
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
                          market_data: MarketData) -> str:
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
                               confidence: float) -> str:
        """Generate human-readable explanation for the signal."""
        
        if signal == SignalType.BUY:
            return (f"Bullish momentum ({momentum:.3f}) with {direction} trend. "
                   f"Confidence: {confidence:.0%}")
        elif signal == SignalType.SELL:
            return (f"Bearish momentum ({momentum:.3f}) with {direction} trend. "
                   f"Confidence: {confidence:.0%}")
        else:
            return (f"Neutral momentum ({momentum:.3f}) or low confidence ({confidence:.0%}). "
                   f"Hold current position.")
    
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
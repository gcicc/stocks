"""
Advanced Risk Management System for Portfolio Intelligence Platform.
Implements Kelly Criterion, Risk Parity, dynamic stop-losses, and correlation analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from core.signal_generator import TradingSignal, SignalType
from core.data_manager import MarketData

logger = logging.getLogger(__name__)


class PositionSizingMethod(Enum):
    """Position sizing methods available."""
    EQUAL_WEIGHT = "equal_weight"
    KELLY_CRITERION = "kelly_criterion"
    RISK_PARITY = "risk_parity" 
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    SIGNAL_STRENGTH = "signal_strength"


@dataclass
class RiskParameters:
    """Risk management parameters."""
    max_position_size: float = 0.2  # Maximum 20% per position
    max_portfolio_risk: float = 0.02  # Maximum 2% portfolio risk per trade
    stop_loss_method: str = "atr"  # 'fixed', 'atr', 'dynamic'
    stop_loss_multiplier: float = 2.0  # Multiplier for ATR-based stops
    correlation_threshold: float = 0.7  # Maximum correlation between positions
    lookback_period: int = 60  # Days for volatility/correlation calculations
    kelly_fraction: float = 0.25  # Fraction of Kelly position to use
    min_position_size: float = 0.01  # Minimum 1% position size
    max_drawdown_limit: float = 0.15  # Maximum 15% portfolio drawdown


@dataclass
class PositionSize:
    """Position sizing recommendation."""
    symbol: str
    recommended_size: float  # As fraction of portfolio
    recommended_shares: int
    risk_amount: float  # Dollar amount at risk
    stop_loss_price: Optional[float] = None
    reasoning: str = ""
    confidence: float = 0.0


@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics."""
    total_risk: float  # Total portfolio risk as % of capital
    correlation_matrix: Optional[pd.DataFrame] = None
    diversification_ratio: float = 0.0  # Higher = more diversified
    concentration_risk: float = 0.0  # Higher = more concentrated
    expected_volatility: float = 0.0  # Expected portfolio volatility
    var_95: float = 0.0  # 95% Value at Risk
    max_correlated_exposure: float = 0.0  # Largest correlated position group


class RiskManager:
    """
    Advanced risk management system implementing multiple position sizing
    algorithms and dynamic stop-loss optimization.
    """
    
    def __init__(self, risk_params: Optional[RiskParameters] = None):
        self.risk_params = risk_params or RiskParameters()
        logger.info(f"Risk manager initialized with {self.risk_params.max_position_size:.1%} max position size")
    
    def calculate_position_sizes(self,
                               signals: Dict[str, TradingSignal],
                               market_data: Dict[str, MarketData],
                               current_portfolio_value: float,
                               method: PositionSizingMethod = PositionSizingMethod.KELLY_CRITERION) -> Dict[str, PositionSize]:
        """
        Calculate optimal position sizes using specified method.
        
        Args:
            signals: Trading signals for each symbol
            market_data: Current market data for each symbol
            current_portfolio_value: Total portfolio value
            method: Position sizing method to use
            
        Returns:
            Dictionary mapping symbols to position sizing recommendations
        """
        try:
            # Filter to BUY/SELL signals only
            active_signals = {
                symbol: signal for symbol, signal in signals.items()
                if signal.signal in [SignalType.BUY, SignalType.SELL]
            }
            
            if not active_signals:
                return {}
            
            logger.info(f"Calculating position sizes for {len(active_signals)} signals using {method.value}")
            
            # Calculate position sizes based on method
            if method == PositionSizingMethod.KELLY_CRITERION:
                return self._kelly_criterion_sizing(active_signals, market_data, current_portfolio_value)
            elif method == PositionSizingMethod.RISK_PARITY:
                return self._risk_parity_sizing(active_signals, market_data, current_portfolio_value)
            elif method == PositionSizingMethod.VOLATILITY_ADJUSTED:
                return self._volatility_adjusted_sizing(active_signals, market_data, current_portfolio_value)
            elif method == PositionSizingMethod.SIGNAL_STRENGTH:
                return self._signal_strength_sizing(active_signals, market_data, current_portfolio_value)
            else:  # EQUAL_WEIGHT
                return self._equal_weight_sizing(active_signals, market_data, current_portfolio_value)
        
        except Exception as e:
            logger.error(f"Error calculating position sizes: {str(e)}")
            return {}
    
    def _kelly_criterion_sizing(self,
                              signals: Dict[str, TradingSignal],
                              market_data: Dict[str, MarketData],
                              portfolio_value: float) -> Dict[str, PositionSize]:
        """
        Calculate position sizes using Kelly Criterion.
        Kelly% = (bp - q) / b, where:
        b = odds received (avg_win/avg_loss), p = win_probability, q = 1-p
        """
        position_sizes = {}
        
        for symbol, signal in signals.items():
            try:
                data = market_data.get(symbol)
                if not data or len(data.prices) < self.risk_params.lookback_period:
                    continue
                
                # Estimate win probability from signal confidence
                win_prob = signal.confidence
                
                # Estimate win/loss ratio from historical volatility and signal strength
                returns = data.prices['Close'].pct_change().dropna()
                if len(returns) < 20:
                    continue
                
                avg_return = returns.mean()
                volatility = returns.std()
                
                # Estimate average win/loss based on signal strength and volatility
                expected_win = abs(avg_return) + (signal.strength * volatility)
                expected_loss = volatility * 1.5  # Conservative loss estimate
                
                if expected_loss <= 0:
                    continue
                
                # Kelly Criterion calculation
                b = expected_win / expected_loss  # Odds ratio
                kelly_fraction = (b * win_prob - (1 - win_prob)) / b
                
                # Apply safety factor and constraints
                kelly_fraction = max(0, min(kelly_fraction, 1.0)) * self.risk_params.kelly_fraction
                
                # Ensure within position limits
                position_fraction = min(kelly_fraction, self.risk_params.max_position_size)
                position_fraction = max(position_fraction, self.risk_params.min_position_size)
                
                # Calculate share quantity
                current_price = data.current_price
                position_value = portfolio_value * position_fraction
                shares = int(position_value / current_price)
                
                # Calculate stop loss
                stop_loss = self._calculate_stop_loss(data, signal)
                
                # Calculate risk amount
                if stop_loss and signal.signal == SignalType.BUY:
                    risk_per_share = current_price - stop_loss
                    risk_amount = shares * risk_per_share
                else:
                    risk_amount = position_value * 0.02  # 2% default risk
                
                position_sizes[symbol] = PositionSize(
                    symbol=symbol,
                    recommended_size=position_fraction,
                    recommended_shares=shares,
                    risk_amount=risk_amount,
                    stop_loss_price=stop_loss,
                    reasoning=f"Kelly Criterion: {kelly_fraction:.1%} * {self.risk_params.kelly_fraction:.1%} safety = {position_fraction:.1%}",
                    confidence=signal.confidence
                )
                
            except Exception as e:
                logger.warning(f"Error calculating Kelly sizing for {symbol}: {str(e)}")
                continue
        
        return position_sizes
    
    def _risk_parity_sizing(self,
                          signals: Dict[str, TradingSignal],
                          market_data: Dict[str, MarketData],
                          portfolio_value: float) -> Dict[str, PositionSize]:
        """
        Calculate position sizes using Risk Parity approach.
        Each position contributes equal risk to the portfolio.
        """
        position_sizes = {}
        
        # Calculate volatilities for all symbols
        volatilities = {}
        for symbol, signal in signals.items():
            data = market_data.get(symbol)
            if not data or len(data.prices) < self.risk_params.lookback_period:
                continue
            
            returns = data.prices['Close'].pct_change().dropna()
            if len(returns) >= 20:
                vol = returns.std() * np.sqrt(252)  # Annualized volatility
                volatilities[symbol] = vol
        
        if not volatilities:
            return {}
        
        # Calculate inverse volatility weights
        inv_vol_weights = {symbol: 1/vol for symbol, vol in volatilities.items()}
        total_inv_vol = sum(inv_vol_weights.values())
        
        # Normalize weights
        for symbol, signal in signals.items():
            if symbol not in inv_vol_weights:
                continue
            
            try:
                data = market_data[symbol]
                
                # Risk parity weight
                weight = inv_vol_weights[symbol] / total_inv_vol
                
                # Apply position limits
                position_fraction = min(weight, self.risk_params.max_position_size)
                position_fraction = max(position_fraction, self.risk_params.min_position_size)
                
                # Calculate shares
                current_price = data.current_price
                position_value = portfolio_value * position_fraction
                shares = int(position_value / current_price)
                
                # Calculate stop loss and risk
                stop_loss = self._calculate_stop_loss(data, signal)
                risk_amount = position_value * volatilities[symbol] * 0.1  # Risk estimate
                
                position_sizes[symbol] = PositionSize(
                    symbol=symbol,
                    recommended_size=position_fraction,
                    recommended_shares=shares,
                    risk_amount=risk_amount,
                    stop_loss_price=stop_loss,
                    reasoning=f"Risk Parity: Inverse vol weight {weight:.1%}, adjusted to {position_fraction:.1%}",
                    confidence=signal.confidence
                )
                
            except Exception as e:
                logger.warning(f"Error calculating risk parity sizing for {symbol}: {str(e)}")
                continue
        
        return position_sizes
    
    def _volatility_adjusted_sizing(self,
                                  signals: Dict[str, TradingSignal],
                                  market_data: Dict[str, MarketData],
                                  portfolio_value: float) -> Dict[str, PositionSize]:
        """Size positions based on inverse volatility with signal strength adjustment."""
        position_sizes = {}
        
        for symbol, signal in signals.items():
            try:
                data = market_data.get(symbol)
                if not data or len(data.prices) < 30:
                    continue
                
                returns = data.prices['Close'].pct_change().dropna()
                if len(returns) < 20:
                    continue
                
                # Calculate volatility
                volatility = returns.std() * np.sqrt(252)
                
                # Base size inversely proportional to volatility
                base_size = min(0.1 / volatility, self.risk_params.max_position_size) if volatility > 0 else 0.05
                
                # Adjust by signal strength and confidence
                strength_multiplier = 0.5 + (signal.strength * signal.confidence)
                position_fraction = base_size * strength_multiplier
                
                # Apply constraints
                position_fraction = min(position_fraction, self.risk_params.max_position_size)
                position_fraction = max(position_fraction, self.risk_params.min_position_size)
                
                # Calculate shares and risk
                current_price = data.current_price
                position_value = portfolio_value * position_fraction
                shares = int(position_value / current_price)
                
                stop_loss = self._calculate_stop_loss(data, signal)
                risk_amount = position_value * volatility * 0.1
                
                position_sizes[symbol] = PositionSize(
                    symbol=symbol,
                    recommended_size=position_fraction,
                    recommended_shares=shares,
                    risk_amount=risk_amount,
                    stop_loss_price=stop_loss,
                    reasoning=f"Vol-adjusted: Base {base_size:.1%} * strength {strength_multiplier:.2f} = {position_fraction:.1%}",
                    confidence=signal.confidence
                )
                
            except Exception as e:
                logger.warning(f"Error calculating volatility-adjusted sizing for {symbol}: {str(e)}")
                continue
        
        return position_sizes
    
    def _signal_strength_sizing(self,
                              signals: Dict[str, TradingSignal],
                              market_data: Dict[str, MarketData],
                              portfolio_value: float) -> Dict[str, PositionSize]:
        """Size positions based on signal strength and confidence."""
        position_sizes = {}
        
        # Calculate total signal score for normalization
        total_score = sum(signal.strength * signal.confidence for signal in signals.values())
        
        if total_score <= 0:
            return {}
        
        for symbol, signal in signals.items():
            try:
                data = market_data.get(symbol)
                if not data:
                    continue
                
                # Position size based on relative signal strength
                signal_score = signal.strength * signal.confidence
                base_weight = signal_score / total_score
                
                # Scale to available capital
                position_fraction = base_weight * len(signals) * 0.15  # Conservative scaling
                
                # Apply constraints
                position_fraction = min(position_fraction, self.risk_params.max_position_size)
                position_fraction = max(position_fraction, self.risk_params.min_position_size)
                
                # Calculate shares and risk
                current_price = data.current_price
                position_value = portfolio_value * position_fraction
                shares = int(position_value / current_price)
                
                stop_loss = self._calculate_stop_loss(data, signal)
                risk_amount = position_value * 0.02  # 2% risk
                
                position_sizes[symbol] = PositionSize(
                    symbol=symbol,
                    recommended_size=position_fraction,
                    recommended_shares=shares,
                    risk_amount=risk_amount,
                    stop_loss_price=stop_loss,
                    reasoning=f"Signal strength: Score {signal_score:.3f}/{total_score:.3f} = {position_fraction:.1%}",
                    confidence=signal.confidence
                )
                
            except Exception as e:
                logger.warning(f"Error calculating signal strength sizing for {symbol}: {str(e)}")
                continue
        
        return position_sizes
    
    def _equal_weight_sizing(self,
                           signals: Dict[str, TradingSignal],
                           market_data: Dict[str, MarketData],
                           portfolio_value: float) -> Dict[str, PositionSize]:
        """Equal weight position sizing (fallback method)."""
        position_sizes = {}
        
        if not signals:
            return {}
        
        # Equal weight across all signals
        equal_weight = min(1.0 / len(signals), self.risk_params.max_position_size)
        
        for symbol, signal in signals.items():
            try:
                data = market_data.get(symbol)
                if not data:
                    continue
                
                current_price = data.current_price
                position_value = portfolio_value * equal_weight
                shares = int(position_value / current_price)
                
                stop_loss = self._calculate_stop_loss(data, signal)
                risk_amount = position_value * 0.02
                
                position_sizes[symbol] = PositionSize(
                    symbol=symbol,
                    recommended_size=equal_weight,
                    recommended_shares=shares,
                    risk_amount=risk_amount,
                    stop_loss_price=stop_loss,
                    reasoning=f"Equal weight: {equal_weight:.1%} across {len(signals)} positions",
                    confidence=signal.confidence
                )
                
            except Exception as e:
                logger.warning(f"Error calculating equal weight sizing for {symbol}: {str(e)}")
                continue
        
        return position_sizes
    
    def _calculate_stop_loss(self,
                           market_data: MarketData,
                           signal: TradingSignal) -> Optional[float]:
        """Calculate dynamic stop loss based on market conditions."""
        try:
            current_price = market_data.current_price
            
            if self.risk_params.stop_loss_method == "fixed":
                # Fixed percentage stop loss
                if signal.signal == SignalType.BUY:
                    return current_price * (1 - 0.05)  # 5% stop loss
                else:  # SELL
                    return current_price * (1 + 0.05)
            
            elif self.risk_params.stop_loss_method == "atr":
                # ATR-based stop loss
                atr = self._calculate_atr(market_data.prices)
                if atr > 0:
                    if signal.signal == SignalType.BUY:
                        return current_price - (atr * self.risk_params.stop_loss_multiplier)
                    else:
                        return current_price + (atr * self.risk_params.stop_loss_multiplier)
            
            elif self.risk_params.stop_loss_method == "dynamic":
                # Dynamic stop loss based on volatility and signal strength
                returns = market_data.prices['Close'].pct_change().dropna()
                if len(returns) >= 20:
                    volatility = returns.std()
                    # Stronger signals get tighter stops, weaker signals get looser stops
                    stop_distance = volatility * (2.0 - signal.confidence) * self.risk_params.stop_loss_multiplier
                    
                    if signal.signal == SignalType.BUY:
                        return current_price * (1 - stop_distance)
                    else:
                        return current_price * (1 + stop_distance)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error calculating stop loss for {market_data.symbol}: {str(e)}")
            return None
    
    def _calculate_atr(self, prices: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        try:
            if len(prices) < period + 1:
                return 0.0
            
            high = prices['High']
            low = prices['Low'] 
            close = prices['Close'].shift(1)
            
            tr1 = high - low
            tr2 = (high - close).abs()
            tr3 = (low - close).abs()
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating ATR: {str(e)}")
            return 0.0
    
    def analyze_portfolio_risk(self,
                             position_sizes: Dict[str, PositionSize],
                             market_data: Dict[str, MarketData]) -> PortfolioRisk:
        """Analyze overall portfolio risk including correlations and concentration."""
        try:
            if not position_sizes:
                return PortfolioRisk(total_risk=0.0)
            
            symbols = list(position_sizes.keys())
            weights = [pos.recommended_size for pos in position_sizes.values()]
            
            # Calculate correlation matrix
            correlation_matrix = self._calculate_correlation_matrix(symbols, market_data)
            
            # Calculate portfolio-level metrics
            total_risk = sum(pos.risk_amount for pos in position_sizes.values())
            
            # Diversification ratio (1 = no diversification, higher = more diversified)
            if correlation_matrix is not None and not correlation_matrix.empty:
                avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix, k=1)].mean()
                diversification_ratio = 1 / (1 + avg_correlation) if avg_correlation > 0 else 1.0
            else:
                diversification_ratio = 1.0
            
            # Concentration risk (Herfindahl index)
            concentration_risk = sum(w**2 for w in weights)
            
            # Expected portfolio volatility
            if correlation_matrix is not None and len(weights) == len(correlation_matrix):
                individual_vols = []
                for symbol in symbols:
                    data = market_data.get(symbol)
                    if data and len(data.prices) >= 20:
                        returns = data.prices['Close'].pct_change().dropna()
                        vol = returns.std() * np.sqrt(252)
                        individual_vols.append(vol)
                    else:
                        individual_vols.append(0.2)  # Default 20% volatility
                
                # Portfolio volatility calculation
                portfolio_variance = 0
                for i in range(len(weights)):
                    for j in range(len(weights)):
                        correlation = correlation_matrix.iloc[i, j] if correlation_matrix is not None else 0.5
                        portfolio_variance += weights[i] * weights[j] * individual_vols[i] * individual_vols[j] * correlation
                
                expected_volatility = np.sqrt(portfolio_variance)
            else:
                expected_volatility = 0.15  # Default estimate
            
            # 95% VaR (normal distribution assumption)
            var_95 = expected_volatility * 1.65  # 95th percentile
            
            # Maximum correlated exposure
            max_correlated_exposure = max(weights) if weights else 0.0
            
            return PortfolioRisk(
                total_risk=total_risk,
                correlation_matrix=correlation_matrix,
                diversification_ratio=diversification_ratio,
                concentration_risk=concentration_risk,
                expected_volatility=expected_volatility,
                var_95=var_95,
                max_correlated_exposure=max_correlated_exposure
            )
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio risk: {str(e)}")
            return PortfolioRisk(total_risk=0.0)
    
    def _calculate_correlation_matrix(self,
                                    symbols: List[str],
                                    market_data: Dict[str, MarketData]) -> Optional[pd.DataFrame]:
        """Calculate correlation matrix between symbols."""
        try:
            if len(symbols) < 2:
                return None
            
            # Collect returns for all symbols
            returns_data = {}
            min_length = float('inf')
            
            for symbol in symbols:
                data = market_data.get(symbol)
                if data and len(data.prices) >= 30:
                    returns = data.prices['Close'].pct_change().dropna()
                    if len(returns) >= 20:
                        returns_data[symbol] = returns
                        min_length = min(min_length, len(returns))
            
            if len(returns_data) < 2 or min_length < 20:
                return None
            
            # Align returns to common length
            aligned_returns = {}
            for symbol, returns in returns_data.items():
                aligned_returns[symbol] = returns.tail(min_length)
            
            # Create DataFrame and calculate correlation
            returns_df = pd.DataFrame(aligned_returns)
            correlation_matrix = returns_df.corr()
            
            return correlation_matrix
            
        except Exception as e:
            logger.warning(f"Error calculating correlation matrix: {str(e)}")
            return None
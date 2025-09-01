"""
Comprehensive performance metrics calculator for backtesting results.
Includes risk-adjusted returns, drawdown analysis, and trading statistics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

from backtesting.backtest_engine import BacktestResults, PortfolioSnapshot, Trade

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Risk-related performance metrics."""
    volatility: float  # Annualized volatility
    var_95: float     # 95% Value at Risk (daily)
    var_99: float     # 99% Value at Risk (daily)
    cvar_95: float    # 95% Conditional Value at Risk (Expected Shortfall)
    downside_deviation: float  # Downside deviation (Sortino ratio denominator)
    beta: Optional[float] = None  # Beta vs benchmark
    tracking_error: Optional[float] = None  # Tracking error vs benchmark


@dataclass
class ReturnMetrics:
    """Return-related performance metrics."""
    total_return: float
    annualized_return: float
    compound_annual_growth_rate: float
    geometric_mean: float
    arithmetic_mean: float
    monthly_returns: pd.Series
    annual_returns: pd.Series
    rolling_returns_12m: pd.Series


@dataclass
class RatioMetrics:
    """Risk-adjusted performance ratios."""
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: Optional[float] = None
    treynor_ratio: Optional[float] = None
    omega_ratio: float = 0.0


@dataclass
class DrawdownMetrics:
    """Drawdown analysis metrics."""
    max_drawdown: float
    max_drawdown_duration: int  # Days
    current_drawdown: float
    drawdown_periods: List[Dict[str, Any]]  # All drawdown periods
    recovery_factor: float  # Total return / Max drawdown
    lake_ratio: float  # Average drawdown / Max drawdown
    pain_index: float  # Average drawdown over entire period


@dataclass
class TradingMetrics:
    """Trading activity metrics."""
    total_trades: int
    avg_trades_per_month: float
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    expectancy: float  # Average expected profit per trade
    payoff_ratio: float  # Avg win / Avg loss
    consecutive_wins: int
    consecutive_losses: int
    avg_holding_period: float  # Average days per trade


@dataclass
class ComprehensiveMetrics:
    """Complete set of performance metrics."""
    returns: ReturnMetrics
    risk: RiskMetrics
    ratios: RatioMetrics
    drawdown: DrawdownMetrics
    trading: TradingMetrics
    
    # Overall scores
    risk_adjusted_score: float = 0.0  # Composite score
    consistency_score: float = 0.0    # Return consistency
    efficiency_score: float = 0.0     # Trading efficiency


class PerformanceAnalyzer:
    """
    Comprehensive performance metrics calculator.
    
    Calculates 50+ performance metrics including:
    - Risk-adjusted returns (Sharpe, Sortino, Calmar)
    - Drawdown analysis
    - Value at Risk (VaR) and Expected Shortfall
    - Trading statistics and efficiency metrics
    - Benchmark comparison (optional)
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate for ratio calculations
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf_rate = risk_free_rate / 252  # Convert to daily
        
        logger.info(f"Performance analyzer initialized (risk-free rate: {risk_free_rate:.2%})")
    
    def analyze(self, 
                backtest_results: BacktestResults,
                benchmark_data: Optional[pd.DataFrame] = None) -> ComprehensiveMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            backtest_results: Backtesting results to analyze
            benchmark_data: Optional benchmark data for comparison
            
        Returns:
            Complete set of performance metrics
        """
        try:
            logger.info("Starting comprehensive performance analysis")
            
            # Extract time series data
            portfolio_history = backtest_results.portfolio_history
            trades = backtest_results.trades
            
            if not portfolio_history:
                raise ValueError("No portfolio history available for analysis")
            
            # Calculate returns series
            returns_series = self._calculate_returns_series(portfolio_history)
            
            # Calculate each category of metrics
            return_metrics = self._calculate_return_metrics(portfolio_history, returns_series)
            risk_metrics = self._calculate_risk_metrics(returns_series, benchmark_data)
            ratio_metrics = self._calculate_ratio_metrics(return_metrics, risk_metrics, returns_series)
            drawdown_metrics = self._calculate_drawdown_metrics(portfolio_history)
            trading_metrics = self._calculate_trading_metrics(trades)
            
            # Calculate composite scores
            risk_adjusted_score = self._calculate_risk_adjusted_score(ratio_metrics)
            consistency_score = self._calculate_consistency_score(returns_series)
            efficiency_score = self._calculate_efficiency_score(trading_metrics)
            
            comprehensive = ComprehensiveMetrics(
                returns=return_metrics,
                risk=risk_metrics,
                ratios=ratio_metrics,
                drawdown=drawdown_metrics,
                trading=trading_metrics,
                risk_adjusted_score=risk_adjusted_score,
                consistency_score=consistency_score,
                efficiency_score=efficiency_score
            )
            
            logger.info("Performance analysis completed")
            return comprehensive
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {str(e)}")
            raise
    
    def _calculate_returns_series(self, portfolio_history: List[PortfolioSnapshot]) -> pd.Series:
        """Calculate daily returns series."""
        
        values = [snapshot.total_value for snapshot in portfolio_history]
        dates = [snapshot.date for snapshot in portfolio_history]
        
        # Create DataFrame
        df = pd.DataFrame({'value': values}, index=pd.DatetimeIndex(dates))
        
        # Calculate daily returns
        returns = df['value'].pct_change().dropna()
        
        return returns
    
    def _calculate_return_metrics(self, 
                                 portfolio_history: List[PortfolioSnapshot],
                                 returns_series: pd.Series) -> ReturnMetrics:
        """Calculate return-based performance metrics."""
        
        if not portfolio_history:
            raise ValueError("No portfolio history for return calculations")
        
        initial_value = portfolio_history[0].total_value
        final_value = portfolio_history[-1].total_value
        
        # Basic return calculations
        total_return = (final_value - initial_value) / initial_value
        
        # Time-based calculations
        start_date = portfolio_history[0].date
        end_date = portfolio_history[-1].date
        total_days = (end_date - start_date).days
        years = total_days / 365.25
        
        # Annualized return
        if years > 0:
            annualized_return = (final_value / initial_value) ** (1 / years) - 1
        else:
            annualized_return = total_return
        
        # CAGR (same as annualized return for consistency)
        cagr = annualized_return
        
        # Statistical measures
        geometric_mean = returns_series.add(1).prod() ** (1/len(returns_series)) - 1 if len(returns_series) > 0 else 0.0
        arithmetic_mean = returns_series.mean()
        
        # Monthly and annual aggregations
        monthly_returns = self._aggregate_returns(returns_series, 'M')
        annual_returns = self._aggregate_returns(returns_series, 'A')
        
        # Rolling 12-month returns
        rolling_returns_12m = self._calculate_rolling_returns(returns_series, 252)
        
        return ReturnMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            compound_annual_growth_rate=cagr,
            geometric_mean=geometric_mean,
            arithmetic_mean=arithmetic_mean,
            monthly_returns=monthly_returns,
            annual_returns=annual_returns,
            rolling_returns_12m=rolling_returns_12m
        )
    
    def _calculate_risk_metrics(self, 
                               returns_series: pd.Series,
                               benchmark_data: Optional[pd.DataFrame] = None) -> RiskMetrics:
        """Calculate risk-based performance metrics."""
        
        if len(returns_series) == 0:
            return RiskMetrics(
                volatility=0.0,
                var_95=0.0,
                var_99=0.0,
                cvar_95=0.0,
                downside_deviation=0.0
            )
        
        # Volatility (annualized)
        volatility = returns_series.std() * np.sqrt(252)
        
        # Value at Risk (VaR)
        var_95 = returns_series.quantile(0.05)  # 5th percentile (95% VaR)
        var_99 = returns_series.quantile(0.01)  # 1st percentile (99% VaR)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = returns_series[returns_series <= var_95].mean() if var_95 != 0 else 0.0
        
        # Downside deviation (for Sortino ratio)
        negative_returns = returns_series[returns_series < self.daily_rf_rate]
        downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0.0
        
        # Beta and tracking error (if benchmark provided)
        beta = None
        tracking_error = None
        
        if benchmark_data is not None:
            try:
                # Calculate benchmark returns
                benchmark_returns = benchmark_data['Close'].pct_change().dropna()
                
                # Align dates
                aligned_returns, aligned_benchmark = returns_series.align(benchmark_returns, join='inner')
                
                if len(aligned_returns) > 1:
                    # Beta calculation
                    covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                    benchmark_variance = np.var(aligned_benchmark)
                    beta = covariance / benchmark_variance if benchmark_variance != 0 else 0.0
                    
                    # Tracking error
                    tracking_error = (aligned_returns - aligned_benchmark).std() * np.sqrt(252)
                    
            except Exception as e:
                logger.debug(f"Benchmark calculation failed: {str(e)}")
        
        return RiskMetrics(
            volatility=volatility,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            downside_deviation=downside_deviation,
            beta=beta,
            tracking_error=tracking_error
        )
    
    def _calculate_ratio_metrics(self, 
                               return_metrics: ReturnMetrics,
                               risk_metrics: RiskMetrics,
                               returns_series: pd.Series) -> RatioMetrics:
        """Calculate risk-adjusted performance ratios."""
        
        # Sharpe ratio
        excess_return = return_metrics.annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / risk_metrics.volatility if risk_metrics.volatility > 0 else 0.0
        
        # Sortino ratio (uses downside deviation instead of total volatility)
        sortino_ratio = excess_return / risk_metrics.downside_deviation if risk_metrics.downside_deviation > 0 else 0.0
        
        # Calmar ratio (return / max drawdown will be calculated later)
        calmar_ratio = 0.0  # Placeholder, will be updated after drawdown calculation
        
        # Information ratio (vs benchmark)
        information_ratio = None
        if risk_metrics.tracking_error is not None and risk_metrics.tracking_error > 0:
            information_ratio = excess_return / risk_metrics.tracking_error
        
        # Treynor ratio (vs benchmark)
        treynor_ratio = None
        if risk_metrics.beta is not None and risk_metrics.beta > 0:
            treynor_ratio = excess_return / risk_metrics.beta
        
        # Omega ratio (probability-weighted ratio of gains to losses)
        omega_ratio = self._calculate_omega_ratio(returns_series, self.daily_rf_rate)
        
        return RatioMetrics(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio,
            omega_ratio=omega_ratio
        )
    
    def _calculate_drawdown_metrics(self, portfolio_history: List[PortfolioSnapshot]) -> DrawdownMetrics:
        """Calculate comprehensive drawdown analysis."""
        
        if not portfolio_history:
            return DrawdownMetrics(
                max_drawdown=0.0,
                max_drawdown_duration=0,
                current_drawdown=0.0,
                drawdown_periods=[],
                recovery_factor=0.0,
                lake_ratio=0.0,
                pain_index=0.0
            )
        
        values = [snapshot.total_value for snapshot in portfolio_history]
        dates = [snapshot.date for snapshot in portfolio_history]
        
        # Calculate drawdown series
        running_max = pd.Series(values).expanding().max()
        drawdown_series = (pd.Series(values) - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = abs(drawdown_series.min())
        
        # Current drawdown
        current_drawdown = abs(drawdown_series.iloc[-1])
        
        # Drawdown periods analysis
        drawdown_periods = self._identify_drawdown_periods(drawdown_series, dates)
        
        # Max drawdown duration
        if drawdown_periods:
            max_dd_duration = max(period['duration_days'] for period in drawdown_periods)
        else:
            max_dd_duration = 0
        
        # Recovery factor
        total_return = (values[-1] - values[0]) / values[0]
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0.0
        
        # Lake ratio (average drawdown / max drawdown)
        avg_drawdown = abs(drawdown_series.mean())
        lake_ratio = avg_drawdown / max_drawdown if max_drawdown > 0 else 0.0
        
        # Pain index (average drawdown over entire period)
        pain_index = avg_drawdown
        
        return DrawdownMetrics(
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            current_drawdown=current_drawdown,
            drawdown_periods=drawdown_periods,
            recovery_factor=recovery_factor,
            lake_ratio=lake_ratio,
            pain_index=pain_index
        )
    
    def _calculate_trading_metrics(self, trades: List[Trade]) -> TradingMetrics:
        """Calculate trading activity and efficiency metrics."""
        
        if not trades:
            return TradingMetrics(
                total_trades=0,
                avg_trades_per_month=0.0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                profit_factor=0.0,
                expectancy=0.0,
                payoff_ratio=0.0,
                consecutive_wins=0,
                consecutive_losses=0,
                avg_holding_period=0.0
            )
        
        # Basic counts
        total_trades = len(trades)
        completed_trades = [t for t in trades if t.exit_date is not None]
        winning_trades = [t for t in completed_trades if t.pnl > 0]
        losing_trades = [t for t in completed_trades if t.pnl < 0]
        
        # Win/loss statistics
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = win_count / len(completed_trades) if completed_trades else 0.0
        
        # Profit/loss amounts
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
        largest_win = max([t.pnl for t in winning_trades]) if winning_trades else 0.0
        largest_loss = min([t.pnl for t in losing_trades]) if losing_trades else 0.0
        
        # Profit factor and expectancy
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        expectancy = np.mean([t.pnl for t in completed_trades]) if completed_trades else 0.0
        
        # Payoff ratio
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
        
        # Consecutive wins/losses
        consecutive_wins, consecutive_losses = self._calculate_consecutive_trades(completed_trades)
        
        # Average holding period
        holding_periods = []
        for trade in completed_trades:
            if trade.exit_date and trade.entry_date:
                days = (trade.exit_date - trade.entry_date).days
                holding_periods.append(days)
        
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0.0
        
        # Trading frequency
        if trades:
            first_trade = min(trades, key=lambda t: t.entry_date)
            last_trade = max(trades, key=lambda t: t.entry_date)
            total_months = max(1, (last_trade.entry_date - first_trade.entry_date).days / 30.44)
            avg_trades_per_month = total_trades / total_months
        else:
            avg_trades_per_month = 0.0
        
        return TradingMetrics(
            total_trades=total_trades,
            avg_trades_per_month=avg_trades_per_month,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            payoff_ratio=payoff_ratio,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            avg_holding_period=avg_holding_period
        )
    
    def _aggregate_returns(self, returns_series: pd.Series, frequency: str) -> pd.Series:
        """Aggregate returns to monthly or annual frequency."""
        
        try:
            # Convert daily returns to specified frequency
            return (1 + returns_series).groupby(pd.Grouper(freq=frequency)).prod() - 1
        except Exception:
            return pd.Series(dtype=float)
    
    def _calculate_rolling_returns(self, returns_series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling returns over specified window."""
        
        try:
            return (1 + returns_series).rolling(window=window).apply(lambda x: x.prod() - 1)
        except Exception:
            return pd.Series(dtype=float)
    
    def _calculate_omega_ratio(self, returns_series: pd.Series, threshold: float) -> float:
        """Calculate Omega ratio (probability-weighted ratio of gains to losses)."""
        
        try:
            excess_returns = returns_series - threshold
            gains = excess_returns[excess_returns > 0].sum()
            losses = abs(excess_returns[excess_returns < 0].sum())
            
            return gains / losses if losses > 0 else float('inf')
        except Exception:
            return 0.0
    
    def _identify_drawdown_periods(self, drawdown_series: pd.Series, dates: List[datetime]) -> List[Dict[str, Any]]:
        """Identify individual drawdown periods."""
        
        periods = []
        in_drawdown = False
        start_idx = 0
        
        for i, dd in enumerate(drawdown_series):
            if not in_drawdown and dd < -0.001:  # Start of drawdown (>0.1%)
                in_drawdown = True
                start_idx = i
            elif in_drawdown and dd >= -0.001:  # End of drawdown
                in_drawdown = False
                
                # Record the drawdown period
                period_dd = drawdown_series.iloc[start_idx:i+1]
                max_dd = abs(period_dd.min())
                duration = i - start_idx + 1
                
                periods.append({
                    'start_date': dates[start_idx],
                    'end_date': dates[i],
                    'duration_days': duration,
                    'max_drawdown': max_dd,
                    'recovery_days': duration
                })
        
        # Handle case where we end in a drawdown
        if in_drawdown:
            period_dd = drawdown_series.iloc[start_idx:]
            max_dd = abs(period_dd.min())
            duration = len(drawdown_series) - start_idx
            
            periods.append({
                'start_date': dates[start_idx],
                'end_date': dates[-1],
                'duration_days': duration,
                'max_drawdown': max_dd,
                'recovery_days': -1  # Still in drawdown
            })
        
        return periods
    
    def _calculate_consecutive_trades(self, completed_trades: List[Trade]) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses."""
        
        if not completed_trades:
            return 0, 0
        
        # Sort trades by entry date
        sorted_trades = sorted(completed_trades, key=lambda t: t.entry_date)
        
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in sorted_trades:
            if trade.pnl > 0:  # Winning trade
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:  # Losing trade
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        return max_consecutive_wins, max_consecutive_losses
    
    def _calculate_risk_adjusted_score(self, ratio_metrics: RatioMetrics) -> float:
        """Calculate composite risk-adjusted performance score."""
        
        # Weighted combination of key ratios
        weights = {
            'sharpe': 0.4,
            'sortino': 0.3,
            'calmar': 0.2,
            'omega': 0.1
        }
        
        # Normalize ratios (convert to 0-1 scale)
        normalized_sharpe = max(0, min(1, ratio_metrics.sharpe_ratio / 3.0))  # Cap at 3.0
        normalized_sortino = max(0, min(1, ratio_metrics.sortino_ratio / 3.0))
        normalized_calmar = max(0, min(1, ratio_metrics.calmar_ratio / 2.0))  # Cap at 2.0
        normalized_omega = max(0, min(1, (ratio_metrics.omega_ratio - 1) / 2.0))  # Omega > 1 is good
        
        score = (
            weights['sharpe'] * normalized_sharpe +
            weights['sortino'] * normalized_sortino +
            weights['calmar'] * normalized_calmar +
            weights['omega'] * normalized_omega
        )
        
        return score
    
    def _calculate_consistency_score(self, returns_series: pd.Series) -> float:
        """Calculate return consistency score (0-1 scale)."""
        
        if len(returns_series) == 0:
            return 0.0
        
        # Calculate monthly returns
        monthly_returns = self._aggregate_returns(returns_series, 'M')
        
        if len(monthly_returns) == 0:
            return 0.0
        
        # Consistency based on percentage of positive months
        positive_months = (monthly_returns > 0).sum()
        total_months = len(monthly_returns)
        consistency = positive_months / total_months
        
        return consistency
    
    def _calculate_efficiency_score(self, trading_metrics: TradingMetrics) -> float:
        """Calculate trading efficiency score (0-1 scale)."""
        
        if trading_metrics.total_trades == 0:
            return 0.0
        
        # Combine multiple efficiency factors
        factors = []
        
        # Win rate (target: 50%+)
        win_rate_score = min(1.0, trading_metrics.win_rate * 2)
        factors.append(win_rate_score)
        
        # Profit factor (target: 1.5+)
        profit_factor_score = min(1.0, trading_metrics.profit_factor / 2.0)
        factors.append(profit_factor_score)
        
        # Payoff ratio (target: 1.0+)
        payoff_score = min(1.0, trading_metrics.payoff_ratio)
        factors.append(payoff_score)
        
        # Return average
        return np.mean(factors)
    
    def update_calmar_ratio(self, ratio_metrics: RatioMetrics, 
                          return_metrics: ReturnMetrics,
                          drawdown_metrics: DrawdownMetrics):
        """Update Calmar ratio after drawdown calculation."""
        
        if drawdown_metrics.max_drawdown > 0:
            ratio_metrics.calmar_ratio = return_metrics.annualized_return / drawdown_metrics.max_drawdown
        else:
            ratio_metrics.calmar_ratio = 0.0
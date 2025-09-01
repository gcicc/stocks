"""
Historical Backtesting Engine for Portfolio Intelligence Platform.
Simulates trading strategies with realistic costs and performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum

from core.momentum_calculator import NaturalMomentumCalculator, MomentumResult
from core.signal_generator import SignalGenerator, TradingSignal, SignalType
from core.data_manager import MarketData
from utils.config import config

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types for backtesting simulation."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class BacktestSettings:
    """Configuration settings for backtesting simulation."""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1% per trade
    slippage_rate: float = 0.0005   # 0.05% slippage
    position_size_method: str = "equal_weight"  # 'equal_weight', 'risk_parity', 'signal_strength'
    max_position_size: float = 0.2  # Maximum 20% per position
    rebalance_frequency: str = "daily"  # 'daily', 'weekly', 'monthly'
    enable_short_selling: bool = False
    stop_loss_pct: Optional[float] = None  # Global stop loss if specified
    take_profit_pct: Optional[float] = None  # Global take profit if specified


@dataclass
class Trade:
    """Individual trade record for backtesting."""
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime] = None
    order_type: OrderType = OrderType.BUY
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: int = 0
    commission_paid: float = 0.0
    slippage_cost: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    signal_confidence: float = 0.0
    signal_strength: float = 0.0
    hold_days: int = 0
    exit_reason: str = ""  # 'signal', 'stop_loss', 'take_profit', 'end_of_period'


@dataclass
class PortfolioSnapshot:
    """Portfolio state at a specific point in time."""
    date: datetime
    total_value: float
    cash: float
    positions: Dict[str, int] = field(default_factory=dict)  # symbol -> quantity
    position_values: Dict[str, float] = field(default_factory=dict)  # symbol -> market value
    daily_return: float = 0.0
    cumulative_return: float = 0.0


@dataclass
class BacktestResults:
    """Comprehensive backtesting results and performance metrics."""
    settings: BacktestSettings
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    
    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    
    # Trading statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Time series data
    portfolio_history: List[PortfolioSnapshot] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    
    # Additional metrics
    total_commission_paid: float = 0.0
    total_slippage_cost: float = 0.0
    
    @property
    def summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics as dictionary."""
        return {
            'total_return': f"{self.total_return:.2%}",
            'annualized_return': f"{self.annualized_return:.2%}",
            'volatility': f"{self.volatility:.2%}",
            'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
            'max_drawdown': f"{self.max_drawdown:.2%}",
            'win_rate': f"{self.win_rate:.2%}",
            'total_trades': self.total_trades,
            'profit_factor': f"{self.profit_factor:.2f}"
        }


class BacktestEngine:
    """
    Historical backtesting engine for momentum-based trading strategies.
    
    Features:
    - Realistic trading costs (commission + slippage)
    - Multiple position sizing methods
    - Risk management (stop loss, take profit)
    - Comprehensive performance metrics
    - Trade-level analysis
    """
    
    def __init__(self, 
                 momentum_calculator: NaturalMomentumCalculator,
                 signal_generator: SignalGenerator):
        self.momentum_calculator = momentum_calculator
        self.signal_generator = signal_generator
        
        # Performance tracking
        self.current_capital = 0.0
        self.current_positions = {}  # symbol -> quantity
        self.cash = 0.0
        
        logger.info("Backtesting engine initialized")
    
    def run_backtest(self, 
                    historical_data: Dict[str, pd.DataFrame],
                    settings: BacktestSettings) -> BacktestResults:
        """
        Run complete backtesting simulation.
        
        Args:
            historical_data: Dict mapping symbols to OHLCV DataFrames with DatetimeIndex
            settings: Backtesting configuration
            
        Returns:
            BacktestResults with comprehensive performance analysis
        """
        try:
            logger.info(f"Starting backtest: {settings.start_date} to {settings.end_date}")
            logger.info(f"Initial capital: ${settings.initial_capital:,.2f}")
            logger.info(f"Universe: {len(historical_data)} symbols")
            
            # Initialize tracking variables
            self.current_capital = settings.initial_capital
            self.cash = settings.initial_capital
            self.current_positions = {}
            
            portfolio_history = []
            all_trades = []
            
            # Get trading date range
            trading_dates = self._get_trading_dates(historical_data, settings)
            logger.info(f"Trading over {len(trading_dates)} days")
            
            # Main simulation loop
            for i, current_date in enumerate(trading_dates):
                try:
                    # Get market data for current date
                    current_market_data = self._get_market_data_for_date(
                        historical_data, current_date
                    )
                    
                    if not current_market_data:
                        # Still create portfolio snapshot with current cash value
                        portfolio_value = self.cash
                        daily_return = self._calculate_daily_return(portfolio_history, portfolio_value)
                        cumulative_return = (portfolio_value - settings.initial_capital) / settings.initial_capital
                        
                        snapshot = PortfolioSnapshot(
                            date=current_date,
                            total_value=portfolio_value,
                            cash=self.cash,
                            positions=dict(self.current_positions),
                            position_values={},
                            daily_return=daily_return,
                            cumulative_return=cumulative_return
                        )
                        portfolio_history.append(snapshot)
                        continue
                    
                    # Calculate momentum indicators for all symbols
                    momentum_results = self._calculate_momentum_for_date(
                        historical_data, current_date, settings
                    )
                    
                    # Generate trading signals
                    signals = self._generate_signals(momentum_results, current_market_data)
                    
                    # Execute trades based on signals
                    daily_trades = self._execute_trades(
                        signals, current_market_data, current_date, settings
                    )
                    all_trades.extend(daily_trades)
                    
                    # Update portfolio valuation
                    portfolio_value = self._calculate_portfolio_value(
                        current_market_data, current_date
                    )
                    
                    # Record portfolio snapshot
                    daily_return = self._calculate_daily_return(portfolio_history, portfolio_value)
                    cumulative_return = (portfolio_value - settings.initial_capital) / settings.initial_capital
                    
                    snapshot = PortfolioSnapshot(
                        date=current_date,
                        total_value=portfolio_value,
                        cash=self.cash,
                        positions=dict(self.current_positions),
                        position_values=self._get_position_values(current_market_data),
                        daily_return=daily_return,
                        cumulative_return=cumulative_return
                    )
                    portfolio_history.append(snapshot)
                    
                    # Progress logging
                    if i % 50 == 0 or i == len(trading_dates) - 1:
                        logger.debug(f"Progress: {current_date.strftime('%Y-%m-%d')} "
                                   f"({i+1}/{len(trading_dates)}) - "
                                   f"Portfolio: ${portfolio_value:,.2f}")
                
                except Exception as e:
                    logger.error(f"Error processing {current_date}: {str(e)}")
                    continue
            
            # Calculate final results
            results = self._calculate_backtest_results(
                settings, portfolio_history, all_trades
            )
            
            logger.info("Backtesting completed successfully")
            logger.info(f"Final results: {results.summary_stats}")
            
            return results
            
        except Exception as e:
            logger.error(f"Backtesting failed: {str(e)}")
            raise
    
    def _get_trading_dates(self, 
                          historical_data: Dict[str, pd.DataFrame],
                          settings: BacktestSettings) -> List[datetime]:
        """Get list of valid trading dates within the backtest period."""
        
        # Find common date range across all symbols
        all_dates = set()
        for symbol_data in historical_data.values():
            symbol_dates = symbol_data.index
            filtered_dates = symbol_dates[
                (symbol_dates >= settings.start_date) & 
                (symbol_dates <= settings.end_date)
            ]
            all_dates.update(filtered_dates.to_pydatetime())
        
        # Sort dates
        trading_dates = sorted(list(all_dates))
        
        logger.debug(f"Found {len(trading_dates)} trading dates")
        return trading_dates
    
    def _get_market_data_for_date(self, 
                                 historical_data: Dict[str, pd.DataFrame],
                                 date: datetime) -> Dict[str, MarketData]:
        """Get MarketData objects for all symbols at specific date."""
        
        market_data = {}
        
        for symbol, data in historical_data.items():
            try:
                # Find closest available date
                available_dates = data.index
                if date not in available_dates:
                    # Find closest date (forward fill)
                    closest_dates = available_dates[available_dates <= date]
                    if len(closest_dates) == 0:
                        continue
                    actual_date = closest_dates[-1]
                else:
                    actual_date = date
                
                # Get price data for the date
                row = data.loc[actual_date]
                
                # Get historical data up to this point (for momentum calculation)
                historical_subset = data.loc[:actual_date].copy()
                
                # Create MarketData object
                market_data[symbol] = MarketData(
                    symbol=symbol,
                    prices=historical_subset,
                    current_price=float(row['Close']),
                    previous_close=float(row['Close']),  # Simplified
                    volume=float(row['Volume']) if 'Volume' in row else 0.0,
                    data_source="backtest"
                )
                
            except Exception as e:
                logger.debug(f"Failed to get market data for {symbol} on {date}: {str(e)}")
                continue
        
        return market_data
    
    def _calculate_momentum_for_date(self, 
                                   historical_data: Dict[str, pd.DataFrame],
                                   date: datetime,
                                   settings: BacktestSettings) -> Dict[str, MomentumResult]:
        """Calculate momentum indicators for all symbols at specific date."""
        
        market_data = self._get_market_data_for_date(historical_data, date)
        return self.momentum_calculator.calculate_portfolio_momentum(market_data)
    
    def _generate_signals(self, 
                         momentum_results: Dict[str, MomentumResult],
                         market_data: Dict[str, MarketData]) -> Dict[str, TradingSignal]:
        """Generate trading signals based on momentum analysis."""
        
        return self.signal_generator.generate_portfolio_signals(
            momentum_results, market_data
        )
    
    def _execute_trades(self, 
                       signals: Dict[str, TradingSignal],
                       market_data: Dict[str, MarketData],
                       date: datetime,
                       settings: BacktestSettings) -> List[Trade]:
        """Execute trades based on generated signals."""
        
        executed_trades = []
        
        for symbol, signal in signals.items():
            try:
                current_position = self.current_positions.get(symbol, 0)
                market_price = market_data[symbol].current_price
                
                # Determine target position
                target_position = self._calculate_position_size(
                    signal, market_price, settings
                )
                
                # Calculate trade quantity
                trade_quantity = target_position - current_position
                
                if abs(trade_quantity) < 1:  # Skip tiny trades
                    continue
                
                # Execute the trade
                if trade_quantity > 0:  # Buy
                    trade = self._execute_buy_order(
                        symbol, trade_quantity, market_price, date, signal, settings
                    )
                elif trade_quantity < 0:  # Sell
                    trade = self._execute_sell_order(
                        symbol, abs(trade_quantity), market_price, date, signal, settings
                    )
                else:
                    continue
                
                if trade:
                    executed_trades.append(trade)
                    # Update position
                    self.current_positions[symbol] = self.current_positions.get(symbol, 0) + (
                        trade_quantity if trade.order_type == OrderType.BUY else -trade_quantity
                    )
                
            except Exception as e:
                logger.error(f"Error executing trade for {symbol}: {str(e)}")
                continue
        
        return executed_trades
    
    def _calculate_position_size(self, 
                               signal: TradingSignal,
                               price: float,
                               settings: BacktestSettings) -> int:
        """Calculate position size based on signal and sizing method."""
        
        if signal.signal == SignalType.HOLD:
            return 0
        
        # Calculate base position value
        if settings.position_size_method == "equal_weight":
            position_value = self.current_capital * settings.max_position_size
        elif settings.position_size_method == "signal_strength":
            position_value = self.current_capital * settings.max_position_size * signal.strength
        else:  # Default to equal weight
            position_value = self.current_capital * settings.max_position_size
        
        # Convert to number of shares
        max_shares = int(position_value / price)
        
        # Don't exceed available cash
        max_affordable = int(self.cash / price) if signal.signal == SignalType.BUY else max_shares
        
        return min(max_shares, max_affordable)
    
    def _execute_buy_order(self, 
                          symbol: str,
                          quantity: int,
                          price: float,
                          date: datetime,
                          signal: TradingSignal,
                          settings: BacktestSettings) -> Optional[Trade]:
        """Execute buy order with realistic trading costs."""
        
        if quantity <= 0:
            return None
        
        # Calculate costs
        gross_amount = quantity * price
        slippage_cost = gross_amount * settings.slippage_rate
        commission = gross_amount * settings.commission_rate
        total_cost = gross_amount + slippage_cost + commission
        
        # Check if we have enough cash
        if total_cost > self.cash:
            # Reduce quantity to fit available cash
            affordable_quantity = int((self.cash - commission) / (price * (1 + settings.slippage_rate)))
            if affordable_quantity <= 0:
                return None
            quantity = affordable_quantity
            gross_amount = quantity * price
            slippage_cost = gross_amount * settings.slippage_rate
            total_cost = gross_amount + slippage_cost + commission
        
        # Update cash
        self.cash -= total_cost
        
        # Create trade record
        trade = Trade(
            symbol=symbol,
            entry_date=date,
            order_type=OrderType.BUY,
            entry_price=price * (1 + settings.slippage_rate),  # Include slippage
            quantity=quantity,
            commission_paid=commission,
            slippage_cost=slippage_cost,
            signal_confidence=signal.confidence,
            signal_strength=signal.strength
        )
        
        logger.debug(f"BUY {quantity} {symbol} @ ${price:.2f} (total: ${total_cost:.2f})")
        return trade
    
    def _execute_sell_order(self, 
                           symbol: str,
                           quantity: int,
                           price: float,
                           date: datetime,
                           signal: TradingSignal,
                           settings: BacktestSettings) -> Optional[Trade]:
        """Execute sell order with realistic trading costs."""
        
        current_position = self.current_positions.get(symbol, 0)
        if quantity > current_position:
            quantity = current_position
        
        if quantity <= 0:
            return None
        
        # Calculate proceeds
        gross_proceeds = quantity * price
        slippage_cost = gross_proceeds * settings.slippage_rate
        commission = gross_proceeds * settings.commission_rate
        net_proceeds = gross_proceeds - slippage_cost - commission
        
        # Update cash
        self.cash += net_proceeds
        
        # Create trade record
        trade = Trade(
            symbol=symbol,
            entry_date=date,
            order_type=OrderType.SELL,
            entry_price=price * (1 - settings.slippage_rate),  # Include slippage
            quantity=quantity,
            commission_paid=commission,
            slippage_cost=slippage_cost,
            signal_confidence=signal.confidence,
            signal_strength=signal.strength
        )
        
        logger.debug(f"SELL {quantity} {symbol} @ ${price:.2f} (proceeds: ${net_proceeds:.2f})")
        return trade
    
    def _calculate_portfolio_value(self, 
                                  market_data: Dict[str, MarketData],
                                  date: datetime) -> float:
        """Calculate current total portfolio value."""
        
        total_value = self.cash
        
        for symbol, quantity in self.current_positions.items():
            if quantity != 0 and symbol in market_data:
                position_value = quantity * market_data[symbol].current_price
                total_value += position_value
        
        return total_value
    
    def _get_position_values(self, market_data: Dict[str, MarketData]) -> Dict[str, float]:
        """Get market values for all current positions."""
        
        position_values = {}
        
        for symbol, quantity in self.current_positions.items():
            if quantity != 0 and symbol in market_data:
                position_values[symbol] = quantity * market_data[symbol].current_price
        
        return position_values
    
    def _calculate_daily_return(self, 
                              portfolio_history: List[PortfolioSnapshot],
                              current_value: float) -> float:
        """Calculate daily return."""
        
        if not portfolio_history:
            return 0.0
        
        previous_value = portfolio_history[-1].total_value
        if previous_value <= 0:
            return 0.0
        
        return (current_value - previous_value) / previous_value
    
    def _calculate_backtest_results(self, 
                                   settings: BacktestSettings,
                                   portfolio_history: List[PortfolioSnapshot],
                                   trades: List[Trade]) -> BacktestResults:
        """Calculate comprehensive backtesting results and performance metrics."""
        
        if not portfolio_history:
            return BacktestResults(
                settings=settings,
                start_date=settings.start_date,
                end_date=settings.end_date,
                initial_capital=settings.initial_capital,
                final_capital=settings.initial_capital
            )
        
        # Basic metrics
        initial_capital = settings.initial_capital
        final_capital = portfolio_history[-1].total_value
        total_return = (final_capital - initial_capital) / initial_capital
        
        # Time-based metrics
        days = (settings.end_date - settings.start_date).days
        years = days / 365.25
        annualized_return = (final_capital / initial_capital) ** (1 / years) - 1 if years > 0 else total_return
        
        # Risk metrics
        daily_returns = [snapshot.daily_return for snapshot in portfolio_history[1:]]
        volatility = np.std(daily_returns) * np.sqrt(252) if daily_returns else 0.0  # Annualized
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0
        
        # Drawdown analysis
        max_dd, max_dd_duration = self._calculate_drawdown_metrics(portfolio_history)
        
        # Trading statistics
        trade_stats = self._calculate_trade_statistics(trades)
        
        # Cost analysis
        total_commission = sum(trade.commission_paid for trade in trades)
        total_slippage = sum(trade.slippage_cost for trade in trades)
        
        return BacktestResults(
            settings=settings,
            start_date=settings.start_date,
            end_date=settings.end_date,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            total_trades=len(trades),
            winning_trades=trade_stats['winning_trades'],
            losing_trades=trade_stats['losing_trades'],
            win_rate=trade_stats['win_rate'],
            avg_win=trade_stats['avg_win'],
            avg_loss=trade_stats['avg_loss'],
            profit_factor=trade_stats['profit_factor'],
            portfolio_history=portfolio_history,
            trades=trades,
            total_commission_paid=total_commission,
            total_slippage_cost=total_slippage
        )
    
    def _calculate_drawdown_metrics(self, 
                                   portfolio_history: List[PortfolioSnapshot]) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration."""
        
        values = [snapshot.total_value for snapshot in portfolio_history]
        peak = values[0]
        max_dd = 0.0
        max_dd_duration = 0
        current_dd_duration = 0
        
        for value in values:
            if value > peak:
                peak = value
                current_dd_duration = 0
            else:
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
        
        return max_dd, max_dd_duration
    
    def _calculate_trade_statistics(self, trades: List[Trade]) -> Dict[str, float]:
        """Calculate detailed trading statistics."""
        
        if not trades:
            return {
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }
        
        # Separate winning and losing trades
        completed_trades = [t for t in trades if t.exit_date is not None]
        winning_trades = [t for t in completed_trades if t.pnl > 0]
        losing_trades = [t for t in completed_trades if t.pnl < 0]
        
        # Basic statistics
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = win_count / len(completed_trades) if completed_trades else 0.0
        
        # Average win/loss
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
        
        # Profit factor
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        return {
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
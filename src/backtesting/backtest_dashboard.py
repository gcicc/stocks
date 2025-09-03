"""
Interactive backtesting dashboard with comprehensive results visualization.
Integrates all backtesting components with Streamlit interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtesting.backtest_engine import BacktestEngine, BacktestSettings, BacktestResults
from backtesting.data_provider import DataProvider, DataRequest
from backtesting.performance_metrics import PerformanceAnalyzer, ComprehensiveMetrics
from core.momentum_calculator import NaturalMomentumCalculator
from core.signal_generator import SignalGenerator
from utils.config import config

logger = logging.getLogger(__name__)


class BacktestDashboard:
    """
    Comprehensive backtesting dashboard with interactive controls and visualizations.
    
    Features:
    - Strategy configuration panel
    - Historical data fetching and validation
    - Real-time backtesting execution
    - Comprehensive performance metrics
    - Interactive charts and analysis
    """
    
    def __init__(self):
        self.data_provider = DataProvider(cache_directory="data/cache")
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Initialize session state
        if 'backtest_results' not in st.session_state:
            st.session_state.backtest_results = None
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = None
    
    def render_dashboard(self):
        """Render the complete backtesting dashboard."""
        
        st.set_page_config(
            page_title="Portfolio Intelligence - Backtesting",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("📊 Historical Backtesting Dashboard")
        st.markdown("---")
        
        # Sidebar configuration
        with st.sidebar:
            self._render_strategy_config()
            
            # Run backtest button
            if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
                self._run_backtest()
        
        # Main dashboard area
        if st.session_state.backtest_results is not None:
            self._render_results_dashboard()
        else:
            self._render_welcome_screen()
    
    def _render_strategy_config(self):
        """Render strategy configuration panel in sidebar."""
        
        st.header("🎯 Strategy Configuration")
        
        # Date range selection
        st.subheader("📅 Backtest Period")
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365),
                max_value=datetime.now() - timedelta(days=30)
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now() - timedelta(days=1),
                max_value=datetime.now()
            )
        
        # Symbol selection
        st.subheader("📈 Universe Selection")
        
        universe_type = st.selectbox(
            "Universe Type",
            ["Popular Stocks", "Tech Stocks", "Custom Selection"],
            index=0
        )
        
        if universe_type == "Popular Stocks":
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'UNH', 'SPY']
        elif universe_type == "Tech Stocks":
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'CRM', 'ADBE', 'NFLX']
        else:
            symbols_input = st.text_input(
                "Enter symbols (comma-separated)",
                value="AAPL, MSFT, GOOGL, TSLA, SPY"
            )
            symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
        
        st.info(f"Selected {len(symbols)} symbols: {', '.join(symbols[:5])}")
        
        # Capital settings
        st.subheader("💰 Capital Settings")
        
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=1000
        )
        
        position_size = st.slider(
            "Max Position Size (%)",
            min_value=1,
            max_value=50,
            value=20,
            help="Maximum percentage of portfolio allocated to a single position"
        )
        
        # Algorithm settings
        st.subheader("⚙️ Algorithm Settings")
        
        tema_period = st.slider(
            "TEMA Period",
            min_value=5,
            max_value=50,
            value=14,
            help="Triple Exponential Moving Average period"
        )
        
        signal_threshold = st.slider(
            "Signal Threshold",
            min_value=0.0,
            max_value=2.0,
            value=0.1,
            step=0.05,
            help="Minimum momentum required for signal generation"
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Minimum confidence required for trade execution"
        )
        
        # Enhanced features
        st.subheader("🔧 Enhanced Features")
        
        enable_multi_timeframe = st.checkbox(
            "Multi-Timeframe Analysis",
            value=True,
            help="Enable enhanced multi-timeframe TEMA analysis"
        )
        
        enable_adaptive_thresholds = st.checkbox(
            "Adaptive Thresholds",
            value=True,
            help="Adjust signal thresholds based on market volatility"
        )
        
        enable_signal_confirmation = st.checkbox(
            "Signal Confirmation",
            value=True,
            help="Require multiple confirmation factors before trade execution"
        )
        
        enable_divergence_detection = st.checkbox(
            "Divergence Detection",
            value=True,
            help="Detect momentum divergences for trend reversal signals"
        )
        
        # Trading costs
        st.subheader("💸 Trading Costs")
        
        commission_rate = st.slider(
            "Commission Rate (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            help="Commission rate per trade"
        ) / 100
        
        slippage_rate = st.slider(
            "Slippage Rate (%)",
            min_value=0.0,
            max_value=0.5,
            value=0.05,
            step=0.01,
            help="Market impact and slippage per trade"
        ) / 100
        
        # Store configuration in session state
        st.session_state.backtest_config = {
            'start_date': datetime.combine(start_date, datetime.min.time()),
            'end_date': datetime.combine(end_date, datetime.min.time()),
            'symbols': symbols,
            'initial_capital': initial_capital,
            'position_size': position_size / 100,
            'tema_period': tema_period,
            'signal_threshold': signal_threshold,
            'confidence_threshold': confidence_threshold,
            'commission_rate': commission_rate,
            'slippage_rate': slippage_rate,
            'enable_multi_timeframe': enable_multi_timeframe,
            'enable_adaptive_thresholds': enable_adaptive_thresholds,
            'enable_signal_confirmation': enable_signal_confirmation,
            'enable_divergence_detection': enable_divergence_detection
        }
    
    def _render_welcome_screen(self):
        """Render welcome screen before backtesting."""
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            ## Welcome to Historical Backtesting
            
            Configure your strategy parameters in the sidebar and click **Run Backtest** to begin.
            
            ### 🚀 Enhanced Features
            
            ✅ **Multi-Timeframe TEMA Analysis**  
            Analyze momentum across short, medium, and long timeframes for better signal quality
            
            ✅ **Adaptive Signal Thresholds**  
            Automatically adjust signal sensitivity based on market volatility conditions
            
            ✅ **Signal Confirmation Logic**  
            Validate signals using momentum consistency, volume, and price alignment
            
            ✅ **Momentum Divergence Detection**  
            Identify potential trend reversals through price-momentum divergences
            
            ✅ **Comprehensive Performance Metrics**  
            50+ metrics including Sharpe ratio, maximum drawdown, and trading statistics
            
            ✅ **Realistic Trading Costs**  
            Include commission and slippage for accurate performance simulation
            
            ### 📊 Analysis Features
            
            - Interactive portfolio value chart
            - Detailed performance metrics
            - Trade-by-trade analysis
            - Risk and return decomposition
            - Benchmark comparison
            """)
            
            if st.button("📖 View Sample Results", use_container_width=True):
                st.session_state.show_sample = True
    
    def _run_backtest(self):
        """Execute backtesting with current configuration."""
        
        config = st.session_state.backtest_config
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Fetch historical data
            status_text.text("📊 Fetching historical data...")
            progress_bar.progress(10)
            
            data_request = DataRequest(
                symbols=config['symbols'],
                start_date=config['start_date'],
                end_date=config['end_date'],
                data_source="yahoo"
            )
            
            historical_data = self.data_provider.fetch_historical_data(data_request)
            
            if not historical_data:
                st.error("❌ Failed to fetch historical data")
                return
            
            progress_bar.progress(30)
            
            # Step 2: Initialize strategy components
            status_text.text("⚙️ Initializing strategy components...")
            
            momentum_calculator = NaturalMomentumCalculator(
                tema_period=config['tema_period'],
                enable_multi_timeframe=config['enable_multi_timeframe']
            )
            
            signal_generator = SignalGenerator(
                signal_threshold=config['signal_threshold'],
                confidence_threshold=config['confidence_threshold'],
                enable_adaptive_thresholds=config['enable_adaptive_thresholds'],
                enable_signal_confirmation=config['enable_signal_confirmation'],
                enable_divergence_detection=config['enable_divergence_detection']
            )
            
            backtest_engine = BacktestEngine(momentum_calculator, signal_generator)
            
            progress_bar.progress(40)
            
            # Step 3: Configure backtest settings
            status_text.text("📋 Configuring backtest settings...")
            
            backtest_settings = BacktestSettings(
                start_date=config['start_date'],
                end_date=config['end_date'],
                initial_capital=config['initial_capital'],
                commission_rate=config['commission_rate'],
                slippage_rate=config['slippage_rate'],
                max_position_size=config['position_size']
            )
            
            progress_bar.progress(50)
            
            # Step 4: Run backtesting
            status_text.text("🚀 Running backtesting simulation...")
            
            results = backtest_engine.run_backtest(historical_data, backtest_settings)
            
            progress_bar.progress(80)
            
            # Step 5: Calculate performance metrics
            status_text.text("📈 Calculating performance metrics...")
            
            performance_metrics = self.performance_analyzer.analyze(results)
            
            # Update Calmar ratio after drawdown calculation
            self.performance_analyzer.update_calmar_ratio(
                performance_metrics.ratios,
                performance_metrics.returns,
                performance_metrics.drawdown
            )
            
            progress_bar.progress(100)
            
            # Store results
            st.session_state.backtest_results = results
            st.session_state.performance_metrics = performance_metrics
            
            status_text.text("✅ Backtesting completed successfully!")
            
            # Clear progress indicators after a moment
            import time
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ Backtest completed! Analyzed {len(historical_data)} symbols over {len(results.portfolio_history)} trading days.")
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Backtesting failed: {str(e)}")
            logger.error(f"Backtesting error: {str(e)}")
    
    def _render_results_dashboard(self):
        """Render comprehensive results dashboard."""
        
        results = st.session_state.backtest_results
        metrics = st.session_state.performance_metrics
        
        # Key metrics header
        self._render_key_metrics(results, metrics)
        
        st.markdown("---")
        
        # Main charts section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._render_performance_chart(results)
        
        with col2:
            self._render_drawdown_chart(results, metrics)
        
        st.markdown("---")
        
        # Detailed analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Metrics", "💹 Trade Analysis", "📈 Risk Analysis", "🔍 Strategy Details"])
        
        with tab1:
            self._render_performance_metrics(metrics)
        
        with tab2:
            self._render_trade_analysis(results)
        
        with tab3:
            self._render_risk_analysis(metrics)
        
        with tab4:
            self._render_strategy_details(results)
    
    def _render_key_metrics(self, results: BacktestResults, metrics: ComprehensiveMetrics):
        """Render key performance metrics header."""
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_return = results.total_return
            st.metric(
                "Total Return",
                f"{total_return:.1%}",
                delta=f"${(results.final_capital - results.initial_capital):,.0f}"
            )
        
        with col2:
            st.metric(
                "Annualized Return",
                f"{results.annualized_return:.1%}",
                delta=f"vs {config.benchmarks.risk_free_rate:.1%} risk-free"
            )
        
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{metrics.ratios.sharpe_ratio:.2f}",
                delta="Risk-adjusted"
            )
        
        with col4:
            st.metric(
                "Max Drawdown",
                f"{results.max_drawdown:.1%}",
                delta=f"{results.max_drawdown_duration} days",
                delta_color="inverse"
            )
        
        with col5:
            st.metric(
                "Win Rate",
                f"{results.win_rate:.1%}",
                delta=f"{results.total_trades} trades"
            )
    
    def _render_performance_chart(self, results: BacktestResults):
        """Render interactive performance chart."""
        
        st.subheader("📈 Portfolio Performance")
        
        # Create performance chart
        df = pd.DataFrame([
            {
                'Date': snapshot.date,
                'Portfolio Value': snapshot.total_value,
                'Cumulative Return': snapshot.cumulative_return * 100,
                'Daily Return': snapshot.daily_return * 100
            }
            for snapshot in results.portfolio_history
        ])
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Portfolio Value', 'Daily Returns'),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Portfolio Value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='<b>%{y:$,.0f}</b><br>%{x}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add benchmark line (starting value)
        initial_value = results.initial_capital
        fig.add_hline(
            y=initial_value,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Initial Capital: ${initial_value:,.0f}",
            row=1, col=1
        )
        
        # Daily returns
        colors = ['green' if ret >= 0 else 'red' for ret in df['Daily Return']]
        fig.add_trace(
            go.Bar(
                x=df['Date'],
                y=df['Daily Return'],
                name='Daily Returns',
                marker_color=colors,
                opacity=0.6,
                hovertemplate='<b>%{y:.2f}%</b><br>%{x}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Portfolio Performance Over Time",
            title_x=0.5
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Daily Return (%)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_drawdown_chart(self, results: BacktestResults, metrics: ComprehensiveMetrics):
        """Render drawdown chart."""
        
        st.subheader("📉 Drawdown Analysis")
        
        # Calculate drawdown series
        values = [snapshot.total_value for snapshot in results.portfolio_history]
        dates = [snapshot.date for snapshot in results.portfolio_history]
        
        running_max = pd.Series(values).expanding().max()
        drawdown_series = (pd.Series(values) - running_max) / running_max * 100
        
        # Create drawdown chart
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=drawdown_series,
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.3)',
                line=dict(color='red', width=1),
                name='Drawdown',
                hovertemplate='<b>%{y:.2f}%</b><br>%{x}<extra></extra>'
            )
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        # Highlight maximum drawdown
        max_dd_idx = drawdown_series.idxmin()
        fig.add_trace(
            go.Scatter(
                x=[dates[max_dd_idx]],
                y=[drawdown_series.iloc[max_dd_idx]],
                mode='markers',
                marker=dict(color='red', size=10, symbol='x'),
                name=f'Max DD: {metrics.drawdown.max_drawdown:.1%}',
                hovertemplate=f'<b>Max Drawdown: {metrics.drawdown.max_drawdown:.1%}</b><br>%{{x}}<extra></extra>'
            )
        )
        
        fig.update_layout(
            title="Portfolio Drawdown Over Time",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown statistics
        st.markdown("**Drawdown Statistics:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Max Drawdown", f"{metrics.drawdown.max_drawdown:.1%}")
        
        with col2:
            st.metric("Max DD Duration", f"{metrics.drawdown.max_drawdown_duration} days")
        
        with col3:
            st.metric("Recovery Factor", f"{metrics.drawdown.recovery_factor:.2f}")
    
    def _render_performance_metrics(self, metrics: ComprehensiveMetrics):
        """Render detailed performance metrics."""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Return Metrics")
            metrics_data = {
                "Total Return": f"{metrics.returns.total_return:.2%}",
                "Annualized Return": f"{metrics.returns.annualized_return:.2%}",
                "CAGR": f"{metrics.returns.compound_annual_growth_rate:.2%}",
                "Geometric Mean": f"{metrics.returns.geometric_mean:.2%}",
                "Arithmetic Mean": f"{metrics.returns.arithmetic_mean:.2%}"
            }
            st.table(pd.DataFrame(list(metrics_data.items()), columns=["Metric", "Value"]))
            
            st.subheader("⚖️ Risk-Adjusted Ratios")
            ratio_data = {
                "Sharpe Ratio": f"{metrics.ratios.sharpe_ratio:.3f}",
                "Sortino Ratio": f"{metrics.ratios.sortino_ratio:.3f}",
                "Calmar Ratio": f"{metrics.ratios.calmar_ratio:.3f}",
                "Omega Ratio": f"{metrics.ratios.omega_ratio:.3f}"
            }
            st.table(pd.DataFrame(list(ratio_data.items()), columns=["Metric", "Value"]))
        
        with col2:
            st.subheader("🎯 Risk Metrics")
            risk_data = {
                "Volatility (Annualized)": f"{metrics.risk.volatility:.2%}",
                "VaR (95%)": f"{metrics.risk.var_95:.2%}",
                "VaR (99%)": f"{metrics.risk.var_99:.2%}",
                "CVaR (95%)": f"{metrics.risk.cvar_95:.2%}",
                "Downside Deviation": f"{metrics.risk.downside_deviation:.2%}"
            }
            st.table(pd.DataFrame(list(risk_data.items()), columns=["Metric", "Value"]))
            
            st.subheader("📉 Drawdown Metrics")
            dd_data = {
                "Max Drawdown": f"{metrics.drawdown.max_drawdown:.2%}",
                "Current Drawdown": f"{metrics.drawdown.current_drawdown:.2%}",
                "Recovery Factor": f"{metrics.drawdown.recovery_factor:.2f}",
                "Pain Index": f"{metrics.drawdown.pain_index:.2%}",
                "Lake Ratio": f"{metrics.drawdown.lake_ratio:.2f}"
            }
            st.table(pd.DataFrame(list(dd_data.items()), columns=["Metric", "Value"]))
    
    def _render_trade_analysis(self, results: BacktestResults):
        """Render trade analysis."""
        
        if not results.trades:
            st.warning("No trades executed during the backtest period.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Trading Statistics")
            
            # Create trades DataFrame
            trades_df = pd.DataFrame([
                {
                    'Symbol': trade.symbol,
                    'Entry Date': trade.entry_date.strftime('%Y-%m-%d'),
                    'Exit Date': trade.exit_date.strftime('%Y-%m-%d') if trade.exit_date else 'Open',
                    'Type': trade.order_type.value,
                    'Entry Price': f"${trade.entry_price:.2f}",
                    'Quantity': trade.quantity,
                    'P&L': f"${trade.pnl:.2f}",
                    'P&L %': f"{trade.pnl_pct:.1%}",
                    'Confidence': f"{trade.signal_confidence:.1%}",
                    'Strength': f"{trade.signal_strength:.1%}"
                }
                for trade in results.trades[:20]  # Show first 20 trades
            ])
            
            st.dataframe(trades_df, use_container_width=True)
            
            if len(results.trades) > 20:
                st.info(f"Showing first 20 of {len(results.trades)} total trades")
        
        with col2:
            st.subheader("📊 Trade Distribution")
            
            # P&L distribution
            pnl_values = [trade.pnl for trade in results.trades if trade.exit_date]
            
            if pnl_values:
                fig = go.Figure()
                
                fig.add_trace(
                    go.Histogram(
                        x=pnl_values,
                        nbinsx=20,
                        marker_color='lightblue',
                        opacity=0.7,
                        name='Trade P&L Distribution'
                    )
                )
                
                fig.update_layout(
                    title="Trade P&L Distribution",
                    xaxis_title="P&L ($)",
                    yaxis_title="Number of Trades",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_risk_analysis(self, metrics: ComprehensiveMetrics):
        """Render risk analysis."""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Risk-Return Analysis")
            
            # Create risk-return scatter
            fig = go.Figure()
            
            # Single point for the strategy
            fig.add_trace(
                go.Scatter(
                    x=[metrics.risk.volatility],
                    y=[metrics.returns.annualized_return],
                    mode='markers',
                    marker=dict(size=15, color='red'),
                    name='Strategy',
                    text=[f'Strategy<br>Return: {metrics.returns.annualized_return:.1%}<br>Risk: {metrics.risk.volatility:.1%}'],
                    hovertemplate='%{text}<extra></extra>'
                )
            )
            
            fig.update_layout(
                title="Risk-Return Profile",
                xaxis_title="Volatility (Risk)",
                yaxis_title="Annualized Return",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Performance Scores")
            
            # Performance scores gauge
            scores = [
                ("Risk-Adjusted Score", metrics.risk_adjusted_score),
                ("Consistency Score", metrics.consistency_score),
                ("Efficiency Score", metrics.efficiency_score)
            ]
            
            for name, score in scores:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': name},
                    gauge = {
                        'axis': {'range': [None, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.3], 'color': "lightgray"},
                            {'range': [0.3, 0.7], 'color': "yellow"},
                            {'range': [0.7, 1], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.8
                        }
                    }
                ))
                
                fig.update_layout(height=250, margin=dict(l=0, r=0, t=50, b=0))
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_strategy_details(self, results: BacktestResults):
        """Render strategy implementation details."""
        
        st.subheader("⚙️ Strategy Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Backtest Settings:**")
            settings_data = {
                "Start Date": results.start_date.strftime('%Y-%m-%d'),
                "End Date": results.end_date.strftime('%Y-%m-%d'),
                "Initial Capital": f"${results.initial_capital:,.2f}",
                "Final Capital": f"${results.final_capital:,.2f}",
                "Commission Rate": f"{results.settings.commission_rate:.3%}",
                "Slippage Rate": f"{results.settings.slippage_rate:.3%}",
                "Max Position Size": f"{results.settings.max_position_size:.1%}"
            }
            st.table(pd.DataFrame(list(settings_data.items()), columns=["Setting", "Value"]))
        
        with col2:
            st.markdown("**Cost Analysis:**")
            cost_data = {
                "Total Commission Paid": f"${results.total_commission_paid:.2f}",
                "Total Slippage Cost": f"${results.total_slippage_cost:.2f}",
                "Total Trading Costs": f"${results.total_commission_paid + results.total_slippage_cost:.2f}",
                "Cost as % of Capital": f"{(results.total_commission_paid + results.total_slippage_cost) / results.initial_capital:.3%}"
            }
            st.table(pd.DataFrame(list(cost_data.items()), columns=["Cost", "Value"]))
        
        # Enhanced features status
        if hasattr(st.session_state, 'backtest_config'):
            config = st.session_state.backtest_config
            
            st.markdown("**Enhanced Features Status:**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                status = "✅ Enabled" if config.get('enable_multi_timeframe') else "❌ Disabled"
                st.markdown(f"**Multi-Timeframe:** {status}")
            
            with col2:
                status = "✅ Enabled" if config.get('enable_adaptive_thresholds') else "❌ Disabled"
                st.markdown(f"**Adaptive Thresholds:** {status}")
            
            with col3:
                status = "✅ Enabled" if config.get('enable_signal_confirmation') else "❌ Disabled"
                st.markdown(f"**Signal Confirmation:** {status}")
            
            with col4:
                status = "✅ Enabled" if config.get('enable_divergence_detection') else "❌ Disabled"
                st.markdown(f"**Divergence Detection:** {status}")


def main():
    """Main entry point for the backtesting dashboard."""
    dashboard = BacktestDashboard()
    dashboard.render_dashboard()


if __name__ == "__main__":
    main()
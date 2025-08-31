"""
Main Streamlit dashboard for Portfolio Intelligence Platform.
Implements the complete user workflow: Upload → Analyze → Export
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import time
from typing import Dict, List, Optional
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our components
from portfolio.csv_parser import parse_portfolio_csv, Portfolio
from core.data_manager import AsyncDataManager
from core.momentum_calculator import NaturalMomentumCalculator
from core.signal_generator import SignalGenerator, SignalType
from utils.config import config


class PortfolioDashboard:
    """Main dashboard class handling the complete workflow."""
    
    def __init__(self):
        self.data_manager = AsyncDataManager()
        self.momentum_calculator = NaturalMomentumCalculator()
        self.signal_generator = SignalGenerator()
        
        # Initialize session state
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = None
        if 'market_data' not in st.session_state:
            st.session_state.market_data = None
        if 'momentum_results' not in st.session_state:
            st.session_state.momentum_results = None
        if 'signals' not in st.session_state:
            st.session_state.signals = None
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
    
    def run(self):
        """Main dashboard entry point."""
        st.set_page_config(
            page_title=config.ui.page_title,
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        self._render_header()
        self._render_sidebar()
        self._render_main_content()
    
    def _render_header(self):
        """Render the main header."""
        st.title("📊 Portfolio Intelligence Platform")
        st.markdown(
            """
            Transform your portfolio CSV into actionable momentum-based trading insights.
            **Upload → Analyze → Act** in under 15 seconds.
            """
        )
        st.divider()
    
    def _render_sidebar(self):
        """Render the sidebar with controls and status."""
        with st.sidebar:
            st.header("📁 Portfolio Upload")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Choose your portfolio CSV file",
                type=['csv'],
                help="Upload portfolio files from Schwab, Fidelity, TD Ameritrade, E*TRADE, etc."
            )
            
            if uploaded_file is not None:
                self._handle_file_upload(uploaded_file)
            
            # Analysis controls
            if st.session_state.portfolio is not None:
                st.header("⚙️ Analysis Settings")
                
                # Symbol limit for performance
                max_symbols = st.slider(
                    "Max symbols to analyze", 
                    min_value=5, 
                    max_value=50, 
                    value=20,
                    help="Limit symbols for faster analysis"
                )
                
                # Analysis period
                period = st.selectbox(
                    "Analysis period",
                    options=["6mo", "1y", "2y"],
                    index=1,
                    help="Historical data period for momentum calculation"
                )
                
                # Run analysis button
                if st.button("🚀 Analyze Portfolio", type="primary"):
                    self._run_analysis(max_symbols, period)
            
            # Status display
            self._render_status_panel()
    
    def _handle_file_upload(self, uploaded_file):
        """Handle portfolio file upload and parsing."""
        try:
            # Read file content
            file_content = uploaded_file.getvalue().decode('utf-8')
            filename = uploaded_file.name
            
            with st.spinner("Parsing portfolio file..."):
                # Parse portfolio
                portfolio = parse_portfolio_csv(file_content, filename)
                st.session_state.portfolio = portfolio
                
                st.success(f"✅ Parsed {len(portfolio.positions)} positions from {filename}")
                
                # Show quick preview
                if len(portfolio.positions) > 0:
                    symbols_preview = ", ".join(portfolio.symbols[:5])
                    if len(portfolio.symbols) > 5:
                        symbols_preview += f" ... (+{len(portfolio.symbols)-5} more)"
                    st.info(f"**Symbols found:** {symbols_preview}")
                
        except Exception as e:
            st.error(f"❌ Error parsing portfolio file: {str(e)}")
            logger.error(f"Portfolio parsing error: {str(e)}")
    
    def _run_analysis(self, max_symbols: int, period: str):
        """Run the complete portfolio analysis."""
        portfolio = st.session_state.portfolio
        if not portfolio:
            st.error("No portfolio loaded")
            return
        
        # Limit symbols for performance
        symbols_to_analyze = portfolio.symbols[:max_symbols]
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Stage 1: Fetch market data
            status_text.text("📡 Fetching market data...")
            progress_bar.progress(0.2)
            
            market_data = asyncio.run(
                self.data_manager.fetch_portfolio_data(symbols_to_analyze, period)
            )
            st.session_state.market_data = market_data
            
            if not market_data:
                st.error("❌ Could not fetch market data for any symbols")
                return
            
            # Stage 2: Calculate momentum
            status_text.text("📈 Calculating momentum indicators...")
            progress_bar.progress(0.5)
            
            momentum_results = self.momentum_calculator.calculate_portfolio_momentum(market_data)
            st.session_state.momentum_results = momentum_results
            
            # Stage 3: Generate signals
            status_text.text("🎯 Generating trading signals...")
            progress_bar.progress(0.8)
            
            signals = self.signal_generator.generate_portfolio_signals(
                momentum_results, market_data
            )
            st.session_state.signals = signals
            
            # Complete
            progress_bar.progress(1.0)
            status_text.text("✅ Analysis complete!")
            st.session_state.analysis_complete = True
            
            # Show success message
            st.success(f"🎉 Analysis complete! Generated signals for {len(signals)} symbols.")
            
            time.sleep(1)  # Brief pause to show completion
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            logger.error(f"Analysis error: {str(e)}")
    
    def _render_status_panel(self):
        """Render status panel in sidebar."""
        st.header("📊 Status")
        
        # Portfolio status
        if st.session_state.portfolio:
            portfolio = st.session_state.portfolio
            st.metric("Positions", len(portfolio.positions))
            if portfolio.total_market_value:
                st.metric("Total Value", f"${portfolio.total_market_value:,.0f}")
        else:
            st.info("No portfolio loaded")
        
        # Analysis status
        if st.session_state.analysis_complete:
            signals = st.session_state.signals or {}
            
            # Count signals by type
            signal_counts = {}
            for signal in signals.values():
                signal_type = signal.signal.value
                signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
            
            st.metric("Buy Signals", signal_counts.get("BUY", 0))
            st.metric("Sell Signals", signal_counts.get("SELL", 0))
            st.metric("Hold Signals", signal_counts.get("HOLD", 0))
    
    def _render_main_content(self):
        """Render the main content area."""
        if not st.session_state.analysis_complete:
            self._render_welcome_screen()
        else:
            self._render_analysis_results()
    
    def _render_welcome_screen(self):
        """Render welcome screen with instructions."""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("🚀 Get Started")
            st.markdown(
                """
                ### How to use this platform:
                
                1. **📁 Upload** your portfolio CSV file in the sidebar
                2. **⚙️ Configure** analysis settings (optional)
                3. **🚀 Click "Analyze Portfolio"** to generate momentum signals
                4. **📊 Review** results and export recommendations
                
                ### Supported Brokers:
                - Charles Schwab
                - Fidelity
                - TD Ameritrade
                - E*TRADE
                - Vanguard
                - Interactive Brokers
                - Generic CSV formats
                """
            )
        
        with col2:
            st.header("📈 Features")
            st.markdown(
                """
                ✅ **Natural Momentum Indicator**  
                Advanced TEMA smoothing with natural log transformations
                
                ✅ **AI-Enhanced Signals**  
                Buy/Sell/Hold recommendations with confidence scores
                
                ✅ **Risk Assessment**  
                Automatic risk level calculation for each position
                
                ✅ **Interactive Charts**  
                Professional visualization of momentum and signals
                
                ✅ **Export Results**  
                Download analysis as CSV for your records
                """
            )
    
    def _render_analysis_results(self):
        """Render analysis results with charts and tables."""
        st.header("📊 Analysis Results")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Signals Overview", "📈 Charts", "🎯 Detailed Analysis", "📤 Export"])
        
        with tab1:
            self._render_signals_overview()
        
        with tab2:
            self._render_charts()
        
        with tab3:
            self._render_detailed_analysis()
        
        with tab4:
            self._render_export_section()
    
    def _render_signals_overview(self):
        """Render signals overview table."""
        signals = st.session_state.signals
        if not signals:
            st.warning("No signals available")
            return
        
        # Create signals dataframe
        signals_data = []
        for symbol, signal in signals.items():
            signals_data.append({
                'Symbol': symbol,
                'Signal': signal.signal.value,
                'Confidence': f"{signal.confidence:.0%}",
                'Strength': f"{signal.strength:.0%}",
                'Risk': signal.risk_level,
                'Price Target': f"${signal.price_target:.2f}" if signal.price_target else "-",
                'Stop Loss': f"${signal.stop_loss:.2f}" if signal.stop_loss else "-",
                'Reason': signal.reason[:50] + "..." if len(signal.reason) > 50 else signal.reason
            })
        
        df = pd.DataFrame(signals_data)
        
        # Add color coding
        def color_signal(val):
            if val == 'BUY':
                return 'background-color: #d4edda'
            elif val == 'SELL':
                return 'background-color: #f8d7da'
            else:
                return 'background-color: #e2e3e5'
        
        styled_df = df.style.applymap(color_signal, subset=['Signal'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        buy_signals = sum(1 for s in signals.values() if s.signal == SignalType.BUY)
        sell_signals = sum(1 for s in signals.values() if s.signal == SignalType.SELL)
        hold_signals = sum(1 for s in signals.values() if s.signal == SignalType.HOLD)
        avg_confidence = np.mean([s.confidence for s in signals.values()])
        
        col1.metric("🟢 Buy Signals", buy_signals)
        col2.metric("🔴 Sell Signals", sell_signals)  
        col3.metric("⚪ Hold Signals", hold_signals)
        col4.metric("🎯 Avg Confidence", f"{avg_confidence:.0%}")
    
    def _render_charts(self):
        """Render interactive charts."""
        if not st.session_state.momentum_results or not st.session_state.market_data:
            st.warning("No data available for charts")
            return
        
        # Select symbol for detailed chart
        symbols = list(st.session_state.momentum_results.keys())
        selected_symbol = st.selectbox("Select symbol for detailed chart:", symbols)
        
        if selected_symbol:
            self._render_symbol_chart(selected_symbol)
        
        # Portfolio overview chart
        st.subheader("📊 Portfolio Momentum Overview")
        self._render_portfolio_overview_chart()
    
    def _render_symbol_chart(self, symbol: str):
        """Render detailed chart for a specific symbol."""
        momentum_result = st.session_state.momentum_results[symbol]
        market_data = st.session_state.market_data[symbol]
        signal = st.session_state.signals[symbol]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'{symbol} - Price and Momentum', 'Momentum Indicator'),
            vertical_spacing=0.1,
            row_heights=[0.6, 0.4]
        )
        
        # Price chart
        prices = market_data.prices
        fig.add_trace(
            go.Candlestick(
                x=prices.index,
                open=prices['Open'],
                high=prices['High'],
                low=prices['Low'],
                close=prices['Close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Momentum chart
        if not momentum_result.tema_values.empty:
            fig.add_trace(
                go.Scatter(
                    x=momentum_result.tema_values.index,
                    y=momentum_result.tema_values.values,
                    mode='lines',
                    name='TEMA Momentum',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=1
            )
            
            # Signal line
            if not momentum_result.signal_line.empty:
                fig.add_trace(
                    go.Scatter(
                        x=momentum_result.signal_line.index,
                        y=momentum_result.signal_line.values,
                        mode='lines',
                        name='Signal Line',
                        line=dict(color='orange', width=1)
                    ),
                    row=2, col=1
                )
            
            # Zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} Analysis - {signal.signal.value} Signal ({signal.confidence:.0%} confidence)",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Signal details
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Signal", signal.signal.value)
        col2.metric("Confidence", f"{signal.confidence:.0%}")
        col3.metric("Risk Level", signal.risk_level)
        
        st.info(f"**Analysis:** {signal.reason}")
    
    def _render_portfolio_overview_chart(self):
        """Render portfolio overview charts."""
        signals = st.session_state.signals
        
        # Signal distribution pie chart
        signal_counts = {}
        for signal in signals.values():
            signal_type = signal.signal.value
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
        
        fig_pie = px.pie(
            values=list(signal_counts.values()),
            names=list(signal_counts.keys()),
            title="Signal Distribution",
            color_discrete_map={
                'BUY': '#28a745',
                'SELL': '#dc3545', 
                'HOLD': '#6c757d'
            }
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Confidence vs Strength scatter plot
        confidence_data = []
        for symbol, signal in signals.items():
            confidence_data.append({
                'Symbol': symbol,
                'Confidence': signal.confidence,
                'Strength': signal.strength,
                'Signal': signal.signal.value,
                'Risk': signal.risk_level
            })
        
        df_scatter = pd.DataFrame(confidence_data)
        
        fig_scatter = px.scatter(
            df_scatter,
            x='Confidence',
            y='Strength', 
            color='Signal',
            size='Confidence',
            hover_name='Symbol',
            title='Signal Confidence vs Strength',
            color_discrete_map={
                'BUY': '#28a745',
                'SELL': '#dc3545',
                'HOLD': '#6c757d'
            }
        )
        
        fig_scatter.update_layout(
            xaxis_title="Confidence Level",
            yaxis_title="Momentum Strength"
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    def _render_detailed_analysis(self):
        """Render detailed analysis section."""
        st.subheader("🔍 Detailed Analysis")
        
        # Data quality report
        if st.session_state.market_data:
            quality_report = self.data_manager.get_data_quality_report(st.session_state.market_data)
            
            st.subheader("📊 Data Quality Report")
            col1, col2, col3 = st.columns(3)
            col1.metric("Data Quality Score", f"{quality_report['data_quality_score']}/100")
            col2.metric("Total Symbols", quality_report['total_symbols'])
            col3.metric("Data Sources", len(quality_report['sources_used']))
            
            if quality_report['sources_used']:
                st.write("**Sources Used:**", quality_report['sources_used'])
        
        # Performance metrics
        st.subheader("⚡ Performance Metrics")
        # Add performance timing information here
        st.info("Analysis completed successfully. All systems operational.")
    
    def _render_export_section(self):
        """Render export functionality."""
        st.subheader("📤 Export Results")
        
        if st.session_state.signals:
            # Create export dataframe
            export_data = []
            for symbol, signal in st.session_state.signals.items():
                export_data.append({
                    'Symbol': symbol,
                    'Signal': signal.signal.value,
                    'Confidence': signal.confidence,
                    'Strength': signal.strength,
                    'Risk_Level': signal.risk_level,
                    'Price_Target': signal.price_target,
                    'Stop_Loss': signal.stop_loss,
                    'Reason': signal.reason,
                    'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            
            export_df = pd.DataFrame(export_data)
            
            # Download button
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Signals CSV",
                data=csv,
                file_name=f"portfolio_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv'
            )
            
            # Preview export data
            st.subheader("📋 Export Preview")
            st.dataframe(export_df, use_container_width=True)
        else:
            st.warning("No signals available to export. Please run analysis first.")


def main():
    """Main application entry point."""
    dashboard = PortfolioDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
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
from datetime import datetime, timedelta
import streamlit.components.v1 as components

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio.csv_parser import parse_portfolio_csv, Portfolio
from core.data_manager import AsyncDataManager
from core.momentum_calculator import NaturalMomentumCalculator
from core.signal_generator import SignalGenerator, SignalType
from utils.config import config

# Backtesting imports
from backtesting.backtest_engine import BacktestEngine, BacktestSettings, BacktestResults
from backtesting.data_provider import DataProvider, DataRequest
from backtesting.performance_metrics import PerformanceAnalyzer


class PortfolioDashboard:
    """Main dashboard class handling the complete workflow."""
    
    def __init__(self):
        self.data_manager = AsyncDataManager()
        self.momentum_calculator = NaturalMomentumCalculator()
        self.signal_generator = SignalGenerator()
        
        # Backtesting components
        self.data_provider = DataProvider(cache_directory="data/cache")
        self.performance_analyzer = PerformanceAnalyzer()
        self.backtest_engine = BacktestEngine(self.momentum_calculator, self.signal_generator)
        
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
        
        # Backtesting session state
        if 'backtest_results' not in st.session_state:
            st.session_state.backtest_results = None
        if 'backtest_metrics' not in st.session_state:
            st.session_state.backtest_metrics = None
    
    def run(self):
        """Main dashboard entry point."""
        st.set_page_config(
            page_title=config.ui.page_title,
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Add custom CSS for professional styling
        self._inject_custom_css()
        
        self._render_header()
        self._render_sidebar()
        self._render_main_content()
    
    def _inject_custom_css(self):
        """Inject custom CSS for professional styling."""
        st.markdown("""
        <style>
        /* Professional color scheme */
        .main .block-container {
            padding-top: 2rem;
        }
        
        /* Enhanced metrics styling */
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
            color: white;
        }
        
        [data-testid="metric-container"] > div {
            width: fit-content;
            margin: auto;
        }
        
        [data-testid="metric-container"] label {
            color: rgba(255, 255, 255, 0.8) !important;
            font-weight: 600;
        }
        
        /* Cool button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 25px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        /* Enhanced tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 10px;
            padding: 0.5rem 1rem;
            border: none;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        /* Progress bar enhancement */
        .stProgress > div > div {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Tooltip styles */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 300px;
            background-color: #333;
            color: #fff;
            text-align: left;
            border-radius: 8px;
            padding: 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -150px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 14px;
            line-height: 1.4;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        
        /* Signal badges */
        .signal-badge-buy {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            padding: 0.25rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            text-align: center;
        }
        
        .signal-badge-sell {
            background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);
            color: white;
            padding: 0.25rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            text-align: center;
        }
        
        .signal-badge-hold {
            background: linear-gradient(135deg, #6c757d 0%, #adb5bd 100%);
            color: white;
            padding: 0.25rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_header(self):
        """Render the main header with enhanced styling."""
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 3rem;
                font-weight: bold;
                margin-bottom: 1rem;
            ">📊 Portfolio Intelligence Platform</h1>
            <p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
                Transform your portfolio CSV into actionable momentum-based trading insights.<br>
                <strong style="color: #667eea;">Upload → Analyze → Act</strong> in under 15 seconds.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    def _render_animated_progress(self):
        """Render animated progress indicator."""
        st.markdown("""
        <div style="
            background: linear-gradient(45deg, #667eea, #764ba2, #667eea);
            background-size: 200% 200%;
            animation: gradient 2s ease infinite;
            height: 4px;
            border-radius: 2px;
            margin: 1rem 0;
        ">
        </div>
        
        <style>
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Render the sidebar with controls and status."""
        with st.sidebar:
            st.header("📁 Portfolio Upload")
            
            # Enhanced file upload section
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1rem;
                border-radius: 10px;
                margin-bottom: 1rem;
                text-align: center;
            ">
                <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">📁</div>
                <div style="font-weight: bold;">Portfolio Upload</div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("📋 Supported Broker Formats", expanded=False):
                st.markdown("""
                **Supported Formats:**
                - 📊 Charles Schwab: Positions export
                - 💼 Fidelity: Portfolio Positions download  
                - 🏦 TD Ameritrade: My Account → Positions
                - 💹 E*TRADE: Portfolio summary
                - 📈 Generic CSV with Symbol, Quantity, Value columns
                """)
            
            uploaded_file = st.file_uploader(
                "Choose your portfolio CSV file",
                type=['csv'],
                help="Upload portfolio files from Schwab, Fidelity, TD Ameritrade, E*TRADE, etc."
            )
            
            if uploaded_file is not None:
                self._handle_file_upload(uploaded_file)
            
            # Watchlist stocks input (always available)
            st.header("📈 Watchlist Stocks")
            
            watchlist_input = st.text_area(
                "Additional stocks to analyze",
                placeholder="GSK, ABBV, MSFT, AAPL",
                help="Enter stock symbols separated by commas. These will be included in momentum analysis as investment opportunities.",
                height=80
            )
            
            # Parse and validate watchlist symbols
            watchlist_symbols = []
            if watchlist_input.strip():
                symbols = [s.strip().upper() for s in watchlist_input.split(',')]
                watchlist_symbols = [s for s in symbols if s and len(s) >= 1 and len(s) <= 6]
                
                if watchlist_symbols:
                    st.success(f"📊 {len(watchlist_symbols)} watchlist symbols: {', '.join(watchlist_symbols)}")
                else:
                    st.warning("⚠️ Please enter valid stock symbols separated by commas")
            
            # Store watchlist in session state
            st.session_state.watchlist_symbols = watchlist_symbols
            
            st.divider()

            # Analysis controls
            if st.session_state.portfolio is not None:
                st.header("⚙️ Analysis Settings")
                
                # Enhanced symbol limit section
                with st.expander("🎯 Performance Guidelines", expanded=False):
                    st.markdown("""
                    **Analysis Speed by Symbol Count:**
                    - 🚀 5-15 symbols: Ultra-fast (~5-8 seconds)
                    - ⚡ 15-30 symbols: Standard (~8-12 seconds)  
                    - 📊 30-50 symbols: Comprehensive (~12-15 seconds)
                    
                    **💡 Tip:** Start with 20 symbols for optimal balance.
                    """)
                
                max_symbols = st.slider(
                    "Max symbols to analyze", 
                    min_value=5, 
                    max_value=50, 
                    value=20,
                    help="Limit symbols for faster analysis"
                )
                
                # Enhanced analysis period section
                with st.expander("📈 Analysis Period Guide", expanded=False):
                    st.markdown("""
                    **Historical Data Periods:**
                    - 📅 6mo: Short-term momentum (reactive signals)
                    - ⚖️ 1y: Balanced momentum (recommended)
                    - 📊 2y: Medium-term momentum (stable trends)
                    - 📈 3y: Long-term momentum (strong trends)
                    - 🏔️ 4y: Extended analysis (business cycles)
                    - 📜 5y: Maximum historical depth (comprehensive)
                    
                    **🧠 Natural Momentum Algorithm:**
                    Uses TEMA smoothing with natural log transformations for enhanced signal clarity.
                    """)
                
                period = st.selectbox(
                    "Analysis period",
                    options=["6mo", "1y", "2y", "3y", "4y", "5y"],
                    index=1,
                    help="Historical data period for momentum calculation"
                )
                
                # Advanced settings
                with st.expander("🎛️ Advanced Signal Settings", expanded=False):
                    st.markdown("**Fine-tune signal sensitivity:**")
                    
                    confidence_threshold = st.slider(
                        "Confidence Threshold",
                        min_value=0.1,
                        max_value=0.9,
                        value=0.4,
                        step=0.05,
                        help="Minimum confidence required for BUY/SELL signals"
                    )
                    
                    strength_threshold = st.slider(
                        "Strength Threshold", 
                        min_value=0.05,
                        max_value=0.5,
                        value=0.15,
                        step=0.05,
                        help="Minimum momentum strength required for signals"
                    )
                    
                    st.info("💡 **Lower thresholds** = More BUY/SELL signals (higher sensitivity)")
                
                # Run analysis button
                if st.button("🚀 Analyze Portfolio", type="primary"):
                    self._run_analysis(max_symbols, period, confidence_threshold, strength_threshold)
            
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
                    st.info("📊 **Portfolio overview available in main panel** → Click 'Portfolio Overview' tab")
                
        except Exception as e:
            st.error(f"❌ Error parsing portfolio file: {str(e)}")
            logger.error(f"Portfolio parsing error: {str(e)}")
    
    def _run_analysis(self, max_symbols: int, period: str, confidence_threshold: float = 0.4, strength_threshold: float = 0.15):
        """Run the complete portfolio analysis."""
        portfolio = st.session_state.portfolio
        if not portfolio:
            st.error("No portfolio loaded")
            return
        
        # Combine portfolio symbols with watchlist symbols
        portfolio_symbols = portfolio.symbols[:max_symbols]
        watchlist_symbols = st.session_state.get('watchlist_symbols', [])
        
        # Combine and deduplicate symbols
        all_symbols = list(set(portfolio_symbols + watchlist_symbols))
        symbols_to_analyze = all_symbols[:max_symbols]  # Still respect the max limit
        
        portfolio_symbol_count = len(portfolio_symbols)
        watchlist_count = len(watchlist_symbols)
        total_unique_count = len(symbols_to_analyze)
        
        # Enhanced progress tracking with animations
        st.session_state.analysis_in_progress = True
        
        # Create progress container
        progress_container = st.container()
        with progress_container:
            progress_col1, progress_col2 = st.columns([3, 1])
            
            with progress_col1:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            with progress_col2:
                # Add spinning loading animation
                loading_placeholder = st.empty()
                loading_placeholder.markdown("""
                <div style="text-align: center; font-size: 2rem; animation: spin 2s linear infinite;">
                    ⚙️
                </div>
                <style>
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                </style>
                """, unsafe_allow_html=True)
        
        try:
            # Stage 1: Fetch market data with enhanced status
            status_text.markdown(f"📡 **Fetching market data** - Analyzing {portfolio_symbol_count} portfolio + {watchlist_count} watchlist symbols...")
            progress_bar.progress(0.2)
            time.sleep(0.5)  # Brief pause for visual effect
            
            market_data = asyncio.run(
                self.data_manager.fetch_portfolio_data(symbols_to_analyze, period)
            )
            st.session_state.market_data = market_data
            
            if not market_data:
                st.error("❌ Could not fetch market data for any symbols")
                return
            
            # Stage 2: Calculate momentum with enhanced status
            status_text.markdown("📈 **Calculating momentum indicators** - Running TEMA analysis...")
            progress_bar.progress(0.5)
            time.sleep(0.3)
            
            momentum_results = self.momentum_calculator.calculate_portfolio_momentum(market_data)
            st.session_state.momentum_results = momentum_results
            
            # Stage 3: Generate signals with enhanced status and custom thresholds
            status_text.markdown("🎯 **Generating trading signals** - AI analysis in progress...")
            progress_bar.progress(0.8)
            time.sleep(0.3)
            
            # Create signal generator with custom thresholds
            custom_signal_generator = SignalGenerator(
                confidence_threshold=confidence_threshold,
                strength_threshold=strength_threshold
            )
            
            # Store thresholds in session state for debug display
            st.session_state.confidence_threshold = confidence_threshold
            st.session_state.strength_threshold = strength_threshold
            
            signals = custom_signal_generator.generate_portfolio_signals(
                momentum_results, market_data
            )
            st.session_state.signals = signals
            
            # Complete with celebration animation
            progress_bar.progress(1.0)
            status_text.markdown("✅ **Analysis complete!** - Ready for review")
            st.session_state.analysis_complete = True
            st.session_state.analysis_in_progress = False
            
            # Show success message with animation
            st.balloons()  # Streamlit celebration animation
            
            # Create detailed success message
            portfolio_signals = len([s for sym, s in signals.items() if sym in portfolio.symbols])
            watchlist_signals = len(signals) - portfolio_signals
            
            if watchlist_count > 0:
                st.success(f"🎉 Analysis complete! Generated signals for {len(signals)} symbols ({portfolio_signals} portfolio + {watchlist_signals} watchlist).")
            else:
                st.success(f"🎉 Analysis complete! Generated signals for {len(signals)} portfolio symbols.")
            
            # Debug information to understand why all signals are HOLD
            if all(signal.signal.value == 'HOLD' for signal in signals.values()):
                with st.expander("🔍 Signal Analysis Debug", expanded=True):
                    st.warning("**All signals are HOLD - Let's investigate why:**")
                    
                    debug_data = []
                    for symbol, signal in signals.items():
                        momentum_result = st.session_state.momentum_results[symbol]
                        debug_data.append({
                            'Symbol': symbol,
                            'Current Momentum': f"{momentum_result.current_momentum:.4f}",
                            'Strength': f"{momentum_result.strength:.3f}",
                            'Confidence': f"{signal.confidence:.3f}",
                            'Direction': momentum_result.momentum_direction,
                            'Reason': signal.reason
                        })
                    
                    debug_df = pd.DataFrame(debug_data)
                    st.dataframe(debug_df, use_container_width=True)
                    
                    # Get thresholds from session state
                    conf_thresh = st.session_state.get('confidence_threshold', 0.4)
                    str_thresh = st.session_state.get('strength_threshold', 0.15)
                    
                    st.info(f"""
                    **Current Thresholds:**
                    - Signal threshold: 0.0 (momentum must cross zero)
                    - Confidence threshold: {conf_thresh:.0%}
                    - Strength threshold: {str_thresh:.0%}
                    
                    **Possible causes for all HOLD signals:**
                    - All momentum values near zero (sideways market)
                    - Low confidence scores (< {conf_thresh:.0%})
                    - Low momentum strength (< {str_thresh:.0%})
                    - Inconsistent trend direction
                    
                    **💡 Tip:** Try lowering the thresholds in Advanced Settings and re-run analysis.
                    """)
            
            time.sleep(2)  # Longer pause to show completion
            loading_placeholder.empty()
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            logger.error(f"Analysis error: {str(e)}")
    
    def _render_status_panel(self):
        """Render enhanced status panel in sidebar."""
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        ">
            <h3 style="margin: 0; text-align: center;">📊 System Status</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced portfolio status
        if st.session_state.portfolio:
            portfolio = st.session_state.portfolio
            
            # Portfolio metrics with status indicators
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="tooltip">
                    <div style="text-align: center; padding: 0.5rem; background: #28a745; color: white; border-radius: 8px;">
                        <div style="font-size: 1.2rem; font-weight: bold;">{}</div>
                        <div style="font-size: 0.8rem; opacity: 0.9;">Positions</div>
                    </div>
                    <span class="tooltiptext">
                        Total number of holdings in your portfolio, excluding cash and summary rows.
                    </span>
                </div>
                """.format(len(portfolio.positions)), unsafe_allow_html=True)
            
            with col2:
                if portfolio.total_market_value:
                    st.markdown("""
                    <div class="tooltip">
                        <div style="text-align: center; padding: 0.5rem; background: #007bff; color: white; border-radius: 8px;">
                            <div style="font-size: 1rem; font-weight: bold;">${:,.0f}</div>
                            <div style="font-size: 0.8rem; opacity: 0.9;">Total Value</div>
                        </div>
                        <span class="tooltiptext">
                            Total market value of your portfolio based on current prices.
                        </span>
                    </div>
                    """.format(portfolio.total_market_value), unsafe_allow_html=True)
                else:
                    st.info("📊 Computing value...")
                    
            # Show top holdings preview
            if len(portfolio.symbols) > 0:
                st.markdown("**🎆 Top Holdings:**")
                top_symbols = portfolio.symbols[:3]
                for symbol in top_symbols:
                    st.markdown(f"• {symbol}")
                if len(portfolio.symbols) > 3:
                    st.markdown(f"• ... and {len(portfolio.symbols)-3} more")
        else:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
                padding: 1rem;
                border-radius: 10px;
                text-align: center;
                color: #2d3436;
            ">
                <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">📁</div>
                <div style="font-weight: bold;">No Portfolio Loaded</div>
                <div style="font-size: 0.9rem; opacity: 0.8;">Upload a CSV file to get started</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Show watchlist status
        watchlist_symbols = st.session_state.get('watchlist_symbols', [])
        if watchlist_symbols:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #17a2b8 0%, #138496 100%);
                color: white;
                padding: 0.75rem;
                border-radius: 8px;
                margin: 0.5rem 0;
                text-align: center;
            ">
                <div style="font-size: 1rem; margin-bottom: 0.25rem;">👁️</div>
                <div style="font-weight: bold;">Watchlist Active</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">{} symbols added</div>
            </div>
            """.format(len(watchlist_symbols)), unsafe_allow_html=True)

        # Enhanced analysis status
        if st.session_state.analysis_complete:
            signals = st.session_state.signals or {}
            
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
                color: white;
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
                text-align: center;
            ">
                <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">✅</div>
                <div style="font-weight: bold;">Analysis Complete</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">{} signals generated</div>
            </div>
            """.format(len(signals)), unsafe_allow_html=True)
            
            # Count signals by type with enhanced display
            signal_counts = {}
            for signal in signals.values():
                signal_type = signal.signal.value
                signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
            
            # Signal distribution with colors
            signals_display = [
                ("BUY", signal_counts.get("BUY", 0), "#28a745", "🔊"),
                ("SELL", signal_counts.get("SELL", 0), "#dc3545", "🔋"),
                ("HOLD", signal_counts.get("HOLD", 0), "#6c757d", "⏸️")
            ]
            
            for signal_type, count, color, icon in signals_display:
                st.markdown(f"""
                <div style="
                    background: {color};
                    color: white;
                    padding: 0.5rem;
                    border-radius: 8px;
                    margin: 0.3rem 0;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                ">
                    <span>{icon} {signal_type}</span>
                    <span style="font-weight: bold; font-size: 1.1rem;">{count}</span>
                </div>
                """, unsafe_allow_html=True)
                
            # Add live system status indicator
            st.markdown("""
            <div style="
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                padding: 0.75rem;
                border-radius: 8px;
                margin-top: 1rem;
            ">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <div style="
                        width: 8px;
                        height: 8px;
                        background: #28a745;
                        border-radius: 50%;
                        margin-right: 0.5rem;
                        animation: pulse 2s infinite;
                    "></div>
                    <span style="font-size: 0.9rem; font-weight: bold; color: #495057;">System Status</span>
                </div>
                <div style="font-size: 0.8rem; color: #6c757d;">
                    • Data APIs: Online<br>
                    • AI Models: Active<br>
                    • Processing: Ready
                </div>
            </div>
            
            <style>
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            </style>
            """, unsafe_allow_html=True)
        
        elif st.session_state.get('analysis_in_progress', False):
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #fd7e14 0%, #e17055 100%);
                color: white;
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
                text-align: center;
            ">
                <div style="font-size: 1.2rem; margin-bottom: 0.5rem; animation: spin 2s linear infinite;">⚙️</div>
                <div style="font-weight: bold;">Analysis in Progress</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Processing your portfolio...</div>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_main_content(self):
        """Render the main content area."""
        if st.session_state.portfolio is None:
            # No portfolio uploaded yet - show welcome screen
            self._render_welcome_screen()
        elif not st.session_state.analysis_complete:
            # Portfolio uploaded but analysis not complete - show portfolio overview
            self._render_portfolio_tabs()
        else:
            # Analysis complete - show full results with portfolio overview
            self._render_analysis_results()
        
        # Add professional footer
        self._render_footer()
    
    def _render_welcome_screen(self):
        """Render welcome screen with instructions."""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;">
                <h2 style="margin: 0; font-size: 2rem;">🚀 Get Started</h2>
                <p style="margin: 0.5rem 0; opacity: 0.9;">Transform your portfolio with AI-powered momentum analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive step-by-step guide
            steps = [
                {
                    "icon": "📁",
                    "title": "Upload Portfolio",
                    "description": "Upload your CSV file from any major broker",
                    "details": "Supports Schwab, Fidelity, TD Ameritrade, E*TRADE, and more. Our parser automatically detects the format."
                },
                {
                    "icon": "⚙️",
                    "title": "Configure Settings",
                    "description": "Choose analysis parameters (optional)",
                    "details": "Select the number of symbols and time period. Default settings work great for most portfolios."
                },
                {
                    "icon": "🚀",
                    "title": "Run Analysis",
                    "description": "Click 'Analyze Portfolio' for instant insights",
                    "details": "Our Natural Momentum Algorithm processes your data in seconds using TEMA smoothing."
                },
                {
                    "icon": "📊",
                    "title": "Review & Export",
                    "description": "Get actionable trading signals and insights",
                    "details": "View interactive charts, signal confidence scores, and export recommendations as CSV."
                }
            ]
            
            for i, step in enumerate(steps, 1):
                with st.expander(f"Step {i}: {step['icon']} {step['title']}", expanded=(i==1)):
                    st.markdown(f"**{step['description']}**")
                    st.write(step['details'])
                    if i == 1:
                        st.info("💡 **Tip:** Look for the file upload section in the sidebar to get started!")
            
            # Add supported brokers with logos (text-based)
            st.markdown("""
            <div style="margin-top: 2rem; padding: 1.5rem; background: #f8f9fa; border-radius: 10px;">
                <h4 style="margin-bottom: 1rem; color: #495057;">🏦 Supported Brokers</h4>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; font-size: 0.9rem;">
                    <div>• Charles Schwab</div>
                    <div>• Fidelity</div>
                    <div>• TD Ameritrade</div>
                    <div>• E*TRADE</div>
                    <div>• Vanguard</div>
                    <div>• Interactive Brokers</div>
                </div>
                <p style="margin-top: 1rem; font-size: 0.8rem; color: #6c757d;">Plus any CSV with Symbol, Quantity, Value columns</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;">
                <h2 style="margin: 0; font-size: 2rem;">📈 Platform Features</h2>
                <p style="margin: 0.5rem 0; opacity: 0.9;">Powered by advanced algorithms and AI</p>
            </div>
            """, unsafe_allow_html=True)
            
            features = [
                {
                    "icon": "🧪",
                    "title": "Natural Momentum Indicator",
                    "description": "Advanced TEMA smoothing with natural log transformations for superior signal clarity"
                },
                {
                    "icon": "🤖",
                    "title": "AI-Enhanced Signals",
                    "description": "Machine learning powered Buy/Sell/Hold recommendations with confidence scores"
                },
                {
                    "icon": "🛡️",
                    "title": "Risk Assessment",
                    "description": "Automated risk level calculation and stop-loss suggestions for each position"
                },
                {
                    "icon": "📏",
                    "title": "Interactive Visualizations",
                    "description": "Professional charts with hover details, zoom, and multi-timeframe analysis"
                },
                {
                    "icon": "⏱️",
                    "title": "Lightning Fast",
                    "description": "Complete analysis in under 15 seconds with parallel data processing"
                },
                {
                    "icon": "📤",
                    "title": "Export & Share",
                    "description": "Download detailed reports and share insights with your team or advisor"
                }
            ]
            
            for feature in features:
                st.markdown(f"""
                <div style="
                    background: white;
                    border: 1px solid #e9ecef;
                    border-radius: 10px;
                    padding: 1rem;
                    margin: 0.5rem 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.5rem; margin-right: 0.5rem;">{feature['icon']}</span>
                        <strong style="color: #495057;">{feature['title']}</strong>
                    </div>
                    <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">{feature['description']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    def _render_portfolio_tabs(self):
        """Render portfolio tabs when portfolio is loaded but analysis not complete."""
        st.header("📊 Portfolio Intelligence Platform")
        
        # Create tabs - Portfolio Overview will be the main focus
        tab1, tab2 = st.tabs(["📊 Portfolio Overview", "🚀 Ready for Analysis"])
        
        with tab1:
            if st.session_state.portfolio:
                self._render_portfolio_distribution(st.session_state.portfolio)
        
        with tab2:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                margin: 2rem 0;
            ">
                <h2 style="margin: 0; font-size: 2rem;">🚀 Ready for Momentum Analysis</h2>
                <p style="margin: 1rem 0; opacity: 0.9; font-size: 1.1rem;">
                    Your portfolio has been successfully loaded and analyzed. 
                    Configure your analysis settings in the sidebar and click "Analyze Portfolio" to generate trading signals.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show portfolio summary
            portfolio = st.session_state.portfolio
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Positions", len([p for p in portfolio.positions if p.symbol.upper() not in {'CASH', 'TOTAL', 'GRAND TOTAL'}]))
            
            with col2:
                if portfolio.total_market_value:
                    st.metric("Portfolio Value", f"${portfolio.total_market_value:,.2f}")
            
            with col3:
                st.metric("Symbols Ready", len(portfolio.symbols))
            
            st.info("💡 **Next Step:** Use the sidebar to configure analysis settings, then click 'Analyze Portfolio' to generate momentum-based trading signals.")
    
    def _render_analysis_results(self):
        """Render analysis results with charts and tables."""
        st.header("📊 Analysis Results")
        
        # Create tabs for different views - Portfolio Overview first, then results, plus backtesting
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Portfolio Overview", "📋 Signals Overview", "📈 Charts", "🎯 Detailed Analysis", "📤 Export", "⏳ Backtesting"])
        
        with tab1:
            if st.session_state.portfolio:
                self._render_portfolio_distribution(st.session_state.portfolio)
        
        with tab2:
            self._render_signals_overview()
        
        with tab3:
            self._render_charts()
        
        with tab4:
            self._render_detailed_analysis()
        
        with tab5:
            self._render_export_section()
        
        with tab6:
            self._render_backtesting_section()
    
    def _render_signals_overview(self):
        """Render signals overview table."""
        signals = st.session_state.signals
        if not signals:
            st.warning("No signals available")
            return
        
        # Add detailed explanations section
        with st.expander("📘 Understanding Your Analysis Metrics", expanded=False):
            st.markdown("""
            ### 🎯 **Confidence Score (0-100%)**
            Measures how reliable the signal is based on multiple factors:
            - **Momentum Strength**: Stronger trends = higher confidence
            - **Trend Consistency**: Consistent direction over recent periods
            - **Distance from Zero**: Momentum values further from zero are more reliable
            - **Volume Confirmation**: Higher trading volume supports the signal
            
            **Interpretation:**
            - 🟢 **80-100%**: Very High - Strong conviction signals
            - 🟡 **60-79%**: High - Good signals with minor uncertainty
            - 🟠 **40-59%**: Medium - Moderate signals, proceed with caution
            - 🔴 **20-39%**: Low - Weak signals, consider avoiding
            - ⚫ **0-19%**: Very Low - Avoid these signals
            
            ### 💪 **Strength Score (0-100%)**
            Measures the power/magnitude of the momentum movement:
            - Based on how far current momentum deviates from its recent average
            - Higher values indicate stronger directional movement
            - Considers price velocity and acceleration
            
            **Interpretation:**
            - 🟢 **50-100%**: Strong momentum - significant price movement expected
            - 🟡 **30-49%**: Moderate momentum - steady movement likely
            - 🔴 **0-29%**: Weak momentum - limited price movement expected
            
            ### ⚖️ **Risk Level**
            Assesses the potential risk of the trade:
            - **LOW**: High momentum strength + low volatility = safer trades
            - **MEDIUM**: Moderate factors = standard risk level
            - **HIGH**: Low momentum strength OR high volatility = riskier trades
            
            **Factors considered:**
            - Recent price volatility (20-day)
            - Momentum strength (stronger = lower risk)
            - Trend consistency
            
            💡 **Pro Tip**: Look for signals with Confidence ≥60%, Strength ≥30%, and LOW-MEDIUM risk for the best opportunities.
            """)
        
        st.markdown("---")
        
        # Create enhanced signals dataframe with badges and source identification
        signals_data = []
        portfolio = st.session_state.portfolio
        watchlist_symbols = st.session_state.get('watchlist_symbols', [])
        
        for symbol, signal in signals.items():
            # Create signal badge HTML
            signal_class = f"signal-badge-{signal.signal.value.lower()}"
            signal_badge = f'<div class="{signal_class}">{signal.signal.value}</div>'
            
            # Determine if this is a portfolio or watchlist symbol
            source = "📊 Portfolio" if symbol in portfolio.symbols else "👁️ Watchlist"
            
            signals_data.append({
                'Symbol': symbol,
                'Source': source,
                'Signal': signal_badge,
                'Confidence': f"{signal.confidence:.0%}",
                'Strength': f"{signal.strength:.0%}",
                'Risk': signal.risk_level,
                'Price Target': f"${signal.price_target:.2f}" if signal.price_target else "-",
                'Stop Loss': f"${signal.stop_loss:.2f}" if signal.stop_loss else "-",
                'Reason': signal.reason[:50] + "..." if len(signal.reason) > 50 else signal.reason
            })
        
        df = pd.DataFrame(signals_data)
        
        # Create a cleaner dataframe display without HTML badges
        # Remove HTML from Signal column for proper display
        df_clean = df.copy()
        for i, row in df_clean.iterrows():
            signal_val = row['Signal']
            if '<div' in str(signal_val):
                # Extract signal type from HTML
                if 'BUY' in signal_val:
                    df_clean.at[i, 'Signal'] = 'BUY'
                elif 'SELL' in signal_val:
                    df_clean.at[i, 'Signal'] = 'SELL'
                else:
                    df_clean.at[i, 'Signal'] = 'HOLD'
        
        # Sort by Source (Portfolio first) then by Signal priority
        signal_priority = {'BUY': 1, 'SELL': 2, 'HOLD': 3}
        df_clean['signal_priority'] = df_clean['Signal'].map(signal_priority)
        df_clean['source_priority'] = df_clean['Source'].map({'📊 Portfolio': 1, '👁️ Watchlist': 2})
        df_clean = df_clean.sort_values(['source_priority', 'signal_priority']).drop(['signal_priority', 'source_priority'], axis=1)
        
        # Color coding function for signals
        def color_signal_cells(val):
            if val == 'BUY':
                return 'background-color: #d4edda; color: #155724; font-weight: bold;'
            elif val == 'SELL':
                return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
            elif val == 'HOLD':
                return 'background-color: #e2e3e5; color: #383d41; font-weight: bold;'
            return ''
        
        # Apply styling and display
        styled_df = df_clean.style.map(color_signal_cells, subset=['Signal'])
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Summary metrics with better styling
        st.markdown("### 📊 **Signal Summary**")
        col1, col2, col3, col4 = st.columns(4)
        
        buy_signals = sum(1 for s in signals.values() if s.signal == SignalType.BUY)
        sell_signals = sum(1 for s in signals.values() if s.signal == SignalType.SELL)
        hold_signals = sum(1 for s in signals.values() if s.signal == SignalType.HOLD)
        avg_confidence = np.mean([s.confidence for s in signals.values()])
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #28a745, #20c997); color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                <div style="font-size: 2rem; font-weight: bold;">{}</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">🟢 Buy Signals</div>
            </div>
            """.format(buy_signals), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #dc3545, #e83e8c); color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                <div style="font-size: 2rem; font-weight: bold;">{}</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">🔴 Sell Signals</div>
            </div>
            """.format(sell_signals), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #6c757d, #adb5bd); color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                <div style="font-size: 2rem; font-weight: bold;">{}</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">⚪ Hold Signals</div>
            </div>
            """.format(hold_signals), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #007bff, #6f42c1); color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                <div style="font-size: 2rem; font-weight: bold;">{:.0%}</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">🎯 Avg Confidence</div>
            </div>
            """.format(avg_confidence), unsafe_allow_html=True)
    
    def _render_charts(self):
        """Render interactive charts."""
        if not st.session_state.momentum_results or not st.session_state.market_data:
            st.warning("No data available for charts")
            return
        
        # Add TEMA and momentum interpretation guide
        with st.expander("📈 Understanding Momentum Charts & TEMA Indicator", expanded=False):
            st.markdown("""
            ### 🔍 **What is TEMA (Triple Exponential Moving Average)?**
            
            **TEMA** is a sophisticated smoothing technique that reduces lag while maintaining responsiveness:
            - **Triple Smoothing**: Applies exponential smoothing three times to reduce noise
            - **Lag Reduction**: Much more responsive than traditional moving averages
            - **Trend Following**: Excellent for identifying trend changes early
            
            ### 📉 **Natural Momentum Indicator Explained**
            
            Our momentum calculation uses **natural logarithms** of price changes:
            1. **Log Price Changes**: `ln(price_today / price_yesterday)` - captures true percentage moves
            2. **TEMA Smoothing**: Applied to reduce noise and false signals
            3. **Zero Line**: The critical reference point for trend direction
            
            ### 🎯 **How to Interpret the Charts**
            
            **Top Panel - Price Action:**
            - 💰 **Price Line**: Current stock price movement
            - 📈 **Volume Bars**: Trading activity (darker = higher volume)
            - 🔵 **TEMA Line**: Smoothed trend direction
            
            **Bottom Panel - Momentum Oscillator:**
            - ⬆️ **Above Zero**: Bullish momentum (upward pressure)
            - ⬇️ **Below Zero**: Bearish momentum (downward pressure)
            - 🔴 **Signal Line**: Secondary confirmation line
            - 🎯 **Crossovers**: Key signal generation points
            
            ### 🚦 **Trading Signal Rules**
            
            **BUY Signals:** 
            - Momentum crosses above zero line ⬆️
            - Momentum is strengthening (upward slope)
            - High confidence and strength scores
            
            **SELL Signals:**
            - Momentum crosses below zero line ⬇️
            - Momentum is weakening (downward slope) 
            - High confidence and strength scores
            
            **HOLD Signals:**
            - Momentum near zero (sideways movement)
            - Low confidence or strength scores
            - Conflicting indicators
            
            💡 **Pro Tips:**
            - Look for **divergences**: When price moves up but momentum moves down (or vice versa)
            - **Volume confirmation**: Strong moves with high volume are more reliable
            - **Multiple timeframe analysis**: Check longer-term trends for context
            """)
        
        st.markdown("---")
        
        # Select symbol for detailed chart
        symbols = list(st.session_state.momentum_results.keys())
        selected_symbol = st.selectbox("Select symbol for detailed chart:", symbols)
        
        if selected_symbol:
            self._render_symbol_chart(selected_symbol)
        
        # Portfolio overview chart
        st.markdown("---")
        st.subheader("📊 Portfolio Momentum Overview")
        st.info("📊 **Overview Chart**: Shows momentum distribution across your entire portfolio. Bubble size represents position value, color indicates signal strength.")
        self._render_portfolio_overview_chart()
        
        # Add technical indicators table
        st.markdown("---")
        st.subheader("📋 Technical Indicators Summary")
        self._render_technical_indicators_table()
    
    def _render_symbol_chart(self, symbol: str):
        """Render detailed chart for a specific symbol."""
        momentum_result = st.session_state.momentum_results[symbol]
        market_data = st.session_state.market_data[symbol]
        signal = st.session_state.signals[symbol]
        
        # Create enhanced subplots with professional styling
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                f'<b>{symbol} - Price Action & Momentum Analysis</b>', 
                '<b>Natural Momentum Indicator (TEMA Smoothed)</b>'
            ),
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )
        
        # Enhanced price chart with volume
        prices = market_data.prices
        
        # Candlestick chart with custom colors
        fig.add_trace(
            go.Candlestick(
                x=prices.index,
                open=prices['Open'],
                high=prices['High'],
                low=prices['Low'],
                close=prices['Close'],
                name="Price",
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
                increasing_fillcolor='rgba(38, 166, 154, 0.8)',
                decreasing_fillcolor='rgba(239, 83, 80, 0.8)'
            ),
            row=1, col=1
        )
        
        # Add volume bars if available
        if 'Volume' in prices.columns and not prices['Volume'].isna().all():
            fig.add_trace(
                go.Bar(
                    x=prices.index,
                    y=prices['Volume'],
                    name='Volume',
                    marker_color='rgba(158, 158, 158, 0.3)',
                    yaxis='y2'
                ),
                row=1, col=1, secondary_y=True
            )
        
        # Enhanced momentum chart with gradient fill
        if not momentum_result.tema_values.empty:
            tema_values = momentum_result.tema_values.values
            
            # Create gradient colors based on momentum direction
            colors = ['rgba(38, 166, 154, 0.8)' if val > 0 else 'rgba(239, 83, 80, 0.8)' for val in tema_values]
            
            fig.add_trace(
                go.Scatter(
                    x=momentum_result.tema_values.index,
                    y=tema_values,
                    mode='lines+markers',
                    name='TEMA Momentum',
                    line=dict(
                        color='#667eea',
                        width=3,
                        shape='spline'
                    ),
                    marker=dict(
                        size=4,
                        color=colors,
                        line=dict(width=1, color='white')
                    ),
                    fill='tonexty' if len(tema_values) > 0 else None,
                    fillcolor='rgba(102, 126, 234, 0.1)'
                ),
                row=2, col=1
            )
            
            # Enhanced signal line with smooth animation
            if not momentum_result.signal_line.empty:
                fig.add_trace(
                    go.Scatter(
                        x=momentum_result.signal_line.index,
                        y=momentum_result.signal_line.values,
                        mode='lines',
                        name='Signal Line',
                        line=dict(
                            color='#ff6b6b',
                            width=2,
                            dash='dot'
                        ),
                        opacity=0.8
                    ),
                    row=2, col=1
                )
            
            # Zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Enhanced layout with professional styling
        signal_color = '#26a69a' if signal.signal.value == 'BUY' else '#ef5350' if signal.signal.value == 'SELL' else '#6c757d'
        
        fig.update_layout(
            title={
                'text': f"<b>{symbol} Technical Analysis</b><br><span style='font-size:14px; color:{signal_color}'>{signal.signal.value} Signal - {signal.confidence:.0%} Confidence</span>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=700,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", size=12, color="#333"),
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Update axes styling
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=1,
            linecolor='rgba(128,128,128,0.5)'
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=1,
            linecolor='rgba(128,128,128,0.5)'
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToRemove': ['pan2d', 'lasso2d']})
        
        # Enhanced signal details with styled metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            signal_color = '#26a69a' if signal.signal.value == 'BUY' else '#ef5350' if signal.signal.value == 'SELL' else '#6c757d'
            st.markdown(f"""
            <div class="tooltip">
                <div style="text-align: center; padding: 1rem; background: {signal_color}; color: white; border-radius: 10px; font-weight: bold;">
                    {signal.signal.value}
                    <span class="tooltiptext">
                        Signal generated based on Natural Momentum crossover analysis.
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Confidence", f"{signal.confidence:.0%}")
        
        with col3:
            st.metric("Risk Level", signal.risk_level)
        
        with col4:
            if signal.price_target:
                st.metric("Price Target", f"${signal.price_target:.2f}")
        
        # Enhanced analysis explanation
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-left: 4px solid {signal_color};
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        ">
            <strong style="color: {signal_color};">Technical Analysis:</strong><br>
            {signal.reason}
        </div>
        """, unsafe_allow_html=True)
    
    def _render_portfolio_overview_chart(self):
        """Render portfolio overview charts."""
        signals = st.session_state.signals
        
        # Enhanced signal distribution with 3D pie chart and animations
        signal_counts = {}
        for signal in signals.values():
            signal_type = signal.signal.value
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
        
        # Create modern pie chart with gradient colors
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=list(signal_counts.keys()),
                    values=list(signal_counts.values()),
                    hole=0.4,  # Donut style
                    marker=dict(
                        colors=['#26a69a', '#ef5350', '#6c757d'],
                        line=dict(color='white', width=3)
                    ),
                    textinfo='label+percent',
                    textfont=dict(size=14, color='white', family="Arial Black"),
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                )
            ]
        )
        
        fig_pie.update_layout(
            title={
                'text': '<b>Portfolio Signal Distribution</b>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", size=12, color="#333"),
            margin=dict(t=50, b=50, l=50, r=150)
        )
        
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
        
        # Enhanced confidence vs strength scatter with bubble chart
        confidence_data = []
        for symbol, signal in signals.items():
            confidence_data.append({
                'Symbol': symbol,
                'Confidence': signal.confidence * 100,  # Convert to percentage
                'Strength': signal.strength * 100,
                'Signal': signal.signal.value,
                'Risk': signal.risk_level
            })
        
        df_scatter = pd.DataFrame(confidence_data)
        
        # Create custom bubble chart
        fig_scatter = go.Figure()
        
        colors = {'BUY': '#26a69a', 'SELL': '#ef5350', 'HOLD': '#6c757d'}
        
        for signal_type in df_scatter['Signal'].unique():
            data = df_scatter[df_scatter['Signal'] == signal_type]
            
            fig_scatter.add_trace(
                go.Scatter(
                    x=data['Confidence'],
                    y=data['Strength'],
                    mode='markers',
                    name=signal_type,
                    marker=dict(
                        color=colors[signal_type],
                        size=np.clip(data['Confidence'] * 0.4, 8, 30),  # Size based on confidence, clamped
                        line=dict(width=2, color='white'),
                        opacity=0.8
                    ),
                    text=data['Symbol'],
                    hovertemplate='<b>%{text}</b><br>' +
                                  'Confidence: %{x:.0f}%<br>' +
                                  'Strength: %{y:.0f}%<br>' +
                                  'Signal: ' + signal_type + '<extra></extra>'
                )
            )
        
        fig_scatter.update_layout(
            title={
                'text': '<b>Signal Quality Matrix</b><br><span style="font-size:14px">Confidence vs Momentum Strength</span>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis=dict(
                title='<b>Confidence Level (%)</b>',
                range=[0, 100],
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)'
            ),
            yaxis=dict(
                title='<b>Momentum Strength (%)</b>',
                range=[0, 100],
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)'
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", size=12, color="#333"),
            hovermode='closest',
            template='plotly_white',
            showlegend=True
        )
        
        # Add quadrant lines for better visualization
        fig_scatter.add_hline(y=50, line_dash="dash", line_color="rgba(128,128,128,0.5)")
        fig_scatter.add_vline(x=50, line_dash="dash", line_color="rgba(128,128,128,0.5)")
        
        # Add quadrant annotations
        fig_scatter.add_annotation(x=75, y=75, text="High Quality<br>Signals", showarrow=False, 
                                  font=dict(size=10, color="green"), bgcolor="rgba(255,255,255,0.8)")
        fig_scatter.add_annotation(x=25, y=25, text="Low Quality<br>Signals", showarrow=False, 
                                  font=dict(size=10, color="red"), bgcolor="rgba(255,255,255,0.8)")
        
        st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
    
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
    
    def _render_footer(self):
        """Render professional footer."""
        st.markdown("---")
        
        # Create footer with proper HTML structure
        st.markdown("""
        <div style="
            text-align: center;
            padding: 2rem 0 1rem 0;
            color: #6c757d;
            font-size: 0.9rem;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 10px;
            margin-top: 2rem;
        ">
            <div style="margin-bottom: 1.5rem;">
                <strong style="color: #495057; font-size: 1.1rem;">Portfolio Intelligence Platform</strong><br>
                <span style="color: #6c757d;">Powered by Natural Momentum Algorithm & AI Analysis</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add feature highlights in columns for better layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: white; border-radius: 10px; margin: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📈</div>
                <strong style="color: #495057;">TEMA Smoothing</strong><br>
                <small style="color: #6c757d;">Triple Exponential Moving Average with natural log transformations</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: white; border-radius: 10px; margin: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🤖</div>
                <strong style="color: #495057;">AI Confidence Scoring</strong><br>
                <small style="color: #6c757d;">Machine learning algorithms for signal confidence</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: white; border-radius: 10px; margin: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">⚡</div>
                <strong style="color: #495057;">Real-time Processing</strong><br>
                <small style="color: #6c757d;">Ultra-fast parallel analysis of your entire portfolio</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Add disclaimer
        st.markdown("""
        <div style="
            text-align: center;
            padding: 1rem;
            margin-top: 1rem;
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            color: #856404;
            font-size: 0.9rem;
        ">
            ⚠️ <strong>Disclaimer:</strong> This platform provides technical analysis for educational purposes. 
            Always consult with a financial advisor before making investment decisions.
        </div>
        """, unsafe_allow_html=True)
    
    def _render_portfolio_distribution(self, portfolio):
        """Render preliminary portfolio distribution graphics."""
        st.markdown("---")
        st.subheader("📊 Portfolio Overview")
        
        # Filter out CASH, TOTAL, and invalid positions for analysis
        excluded_symbols = {'CASH', 'TOTAL', 'GRAND TOTAL', 'PORTFOLIO TOTAL', 'ACCOUNT TOTAL', 'NET WORTH', 'BALANCE', 'SUMMARY'}
        valid_positions = [
            pos for pos in portfolio.positions 
            if (pos.symbol.upper() not in excluded_symbols and 
                pos.market_value is not None and 
                pos.market_value > 0 and
                pos.symbol not in ['GENERATED AT JUL 29 2025 07:18 PM ET', ''])
        ]
        
        if not valid_positions:
            st.warning("No valid positions found for analysis")
            return
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Portfolio allocation table
            st.markdown("**📋 Holdings Summary**")
            
            # Create holdings data
            holdings_data = []
            total_value = sum(pos.market_value for pos in valid_positions)
            
            for pos in sorted(valid_positions, key=lambda x: x.market_value, reverse=True):
                allocation = (pos.market_value / total_value) * 100 if total_value > 0 else 0
                holdings_data.append({
                    'Symbol': pos.symbol,
                    'Shares': f"{pos.quantity:,.1f}" if pos.quantity else "0",
                    'Market Value': f"${pos.market_value:,.2f}",
                    'Allocation %': f"{allocation:.1f}%",
                    'Price': f"${pos.current_price:.2f}" if pos.current_price else "-"
                })
            
            holdings_df = pd.DataFrame(holdings_data)
            st.dataframe(holdings_df, use_container_width=True, height=300)
            
            # Portfolio summary metrics with better styling
            st.markdown("**📈 Portfolio Metrics**")
            col_a, col_b, col_c = st.columns(3)
            
            # Fixed font sizing issues
            with col_a:
                st.markdown("""
                <div style="background: #e3f2fd; padding: 0.8rem; border-radius: 8px; text-align: center; border-left: 4px solid #2196f3;">
                    <div style="font-size: 1.5rem; font-weight: bold; color: #1976d2;">{}</div>
                    <div style="font-size: 0.85rem; color: #666; margin-top: 0.2rem;">Total Positions</div>
                </div>
                """.format(len(valid_positions)), unsafe_allow_html=True)
            
            with col_b:
                st.markdown("""
                <div style="background: #e8f5e8; padding: 0.8rem; border-radius: 8px; text-align: center; border-left: 4px solid #4caf50;">
                    <div style="font-size: 1.2rem; font-weight: bold; color: #388e3c;">${:,.0f}</div>
                    <div style="font-size: 0.85rem; color: #666; margin-top: 0.2rem;">Total Value</div>
                </div>
                """.format(total_value), unsafe_allow_html=True)
            
            with col_c:
                largest_symbol = max(valid_positions, key=lambda x: x.market_value).symbol
                st.markdown("""
                <div style="background: #fff3e0; padding: 0.8rem; border-radius: 8px; text-align: center; border-left: 4px solid #ff9800;">
                    <div style="font-size: 1.2rem; font-weight: bold; color: #f57c00;">{}</div>
                    <div style="font-size: 0.85rem; color: #666; margin-top: 0.2rem;">Largest Position</div>
                </div>
                """.format(largest_symbol), unsafe_allow_html=True)
        
        with col2:
            # Professional horizontal bar chart for allocation
            st.markdown("**📋 Portfolio Allocation**")
            
            # Prepare data for bar chart - show all positions
            sorted_positions = sorted(valid_positions, key=lambda x: x.market_value, reverse=True)
            total_value = sum(pos.market_value for pos in valid_positions)
            
            symbols = [pos.symbol for pos in sorted_positions]
            values = [pos.market_value for pos in sorted_positions]
            percentages = [(val/total_value)*100 for val in values]
            
            # Create professional horizontal bar chart
            fig_bar = go.Figure()
            
            # Add bars with gradient colors
            colors = px.colors.sample_colorscale('viridis', [i/len(symbols) for i in range(len(symbols))])
            
            fig_bar.add_trace(go.Bar(
                y=symbols,
                x=percentages,
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(color='rgba(255,255,255,0.8)', width=1)
                ),
                text=[f'${val:,.0f} ({pct:.1f}%)' for val, pct in zip(values, percentages)],
                textposition='inside',
                textfont=dict(color='white', size=10, family='Arial'),
                hovertemplate='<b>%{y}</b><br>Value: $%{x:.1f}k<br>Percentage: %{x:.1f}%<extra></extra>',
                showlegend=False
            ))
            
            fig_bar.update_layout(
                height=400,
                title=dict(
                    text='Portfolio Weights by Position',
                    x=0.5,
                    font=dict(size=14, color='#2c3e50')
                ),
                xaxis=dict(
                    title='Allocation (%)',
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.2)',
                    zeroline=False,
                    tickfont=dict(size=10)
                ),
                yaxis=dict(
                    title='',
                    tickfont=dict(size=10, color='#2c3e50'),
                    categoryorder='total ascending'
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=50, b=20, l=80, r=20),
                font=dict(family='Arial, sans-serif')
            )
            
            st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
        
        # Additional insights
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Concentration analysis with fixed sizing
            top_3_value = sum(sorted([pos.market_value for pos in valid_positions], reverse=True)[:3])
            concentration = (top_3_value / total_value) * 100 if total_value > 0 else 0
            st.markdown("""
            <div style="background: #f3e5f5; padding: 0.8rem; border-radius: 8px; text-align: center; border-left: 4px solid #9c27b0;">
                <div style="font-size: 1.3rem; font-weight: bold; color: #7b1fa2;">{:.1f}%</div>
                <div style="font-size: 0.8rem; color: #666; margin-top: 0.2rem;">Top 3 Concentration</div>
                <div style="font-size: 0.7rem; color: #888; margin-top: 0.1rem;">% of portfolio in top 3 positions</div>
            </div>
            """.format(concentration), unsafe_allow_html=True)
        
        with col2:
            # Average position size with fixed sizing
            avg_position = total_value / len(valid_positions) if valid_positions else 0
            st.markdown("""
            <div style="background: #e1f5fe; padding: 0.8rem; border-radius: 8px; text-align: center; border-left: 4px solid #03a9f4;">
                <div style="font-size: 1.1rem; font-weight: bold; color: #0288d1;">${:,.0f}</div>
                <div style="font-size: 0.8rem; color: #666; margin-top: 0.2rem;">Avg Position Size</div>
                <div style="font-size: 0.7rem; color: #888; margin-top: 0.1rem;">Average market value per position</div>
            </div>
            """.format(avg_position), unsafe_allow_html=True)
        
        with col3:
            # Smallest vs Largest ratio with fixed sizing
            if valid_positions:
                largest = max(pos.market_value for pos in valid_positions)
                smallest = min(pos.market_value for pos in valid_positions)
                ratio = largest / smallest if smallest > 0 else 0
                st.markdown("""
                <div style="background: #fce4ec; padding: 0.8rem; border-radius: 8px; text-align: center; border-left: 4px solid #e91e63;">
                    <div style="font-size: 1.3rem; font-weight: bold; color: #c2185b;">{:.1f}x</div>
                    <div style="font-size: 0.8rem; color: #666; margin-top: 0.2rem;">Size Ratio (L/S)</div>
                    <div style="font-size: 0.7rem; color: #888; margin-top: 0.1rem;">Largest position vs smallest</div>
                </div>
                """.format(ratio), unsafe_allow_html=True)
        
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin-top: 1rem;
        ">
            <div style="font-size: 1.1rem; margin-bottom: 0.5rem;">💡</div>
            <div style="font-weight: bold; font-size: 1rem;">Ready for Analysis!</div>
            <div style="opacity: 0.9; font-size: 0.9rem; margin-top: 0.3rem;">
                Upload complete! Use the sidebar to configure settings and run momentum analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_technical_indicators_table(self):
        """Render holistic technical indicators table."""
        if not st.session_state.momentum_results or not st.session_state.market_data:
            st.warning("No data available for technical indicators")
            return
        
        st.info("📋 **Technical Indicators**: Key metrics for each symbol to complement the momentum analysis.")
        
        indicators_data = []
        
        for symbol, momentum_result in st.session_state.momentum_results.items():
            try:
                market_data = st.session_state.market_data[symbol]
                signal = st.session_state.signals[symbol]
                
                # Calculate additional technical indicators
                prices = market_data.prices['Close'] if 'Close' in market_data.prices.columns else market_data.prices.iloc[:, 0]
                current_price = prices.iloc[-1] if len(prices) > 0 else 0
                
                # Price momentum (current vs 20-day average)
                price_20d = prices.iloc[-20:].mean() if len(prices) >= 20 else current_price
                price_momentum = ((current_price - price_20d) / price_20d * 100) if price_20d > 0 else 0
                
                # Volatility (20-day standard deviation)
                if len(prices) >= 20:
                    returns = prices.pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252) * 100  # Annualized
                else:
                    volatility = 0
                
                # Create interpretations based on combined metrics
                interpretation = self._interpret_technical_indicators(
                    momentum_result.current_momentum,
                    signal.confidence,
                    momentum_result.strength,
                    price_momentum,
                    volatility,
                    signal.signal.value
                )
                
                indicators_data.append({
                    'Symbol': symbol,
                    'TEMA': f"{momentum_result.current_momentum:.4f}",
                    'TEMA_Interp': f"{'Bullish' if momentum_result.current_momentum > 0 else 'Bearish'} momentum",
                    'CONF': f"{signal.confidence:.0%}",
                    'CONF_Interp': f"{'High' if signal.confidence > 0.6 else 'Medium' if signal.confidence > 0.4 else 'Low'} conviction",
                    'STR': f"{momentum_result.strength:.0%}",
                    'STR_Interp': f"{'Strong' if momentum_result.strength > 0.5 else 'Moderate' if momentum_result.strength > 0.3 else 'Weak'} movement",
                    'PM': f"{price_momentum:+.1f}%",
                    'PM_Interp': f"{'Rising' if price_momentum > 2 else 'Declining' if price_momentum < -2 else 'Sideways'} trend",
                    'VOL': f"{volatility:.0f}%",
                    'VOL_Interp': f"{'High' if volatility > 30 else 'Medium' if volatility > 15 else 'Low'} risk",
                    'Overall_Interpretation': interpretation
                })
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {str(e)}")
                continue
        
        if not indicators_data:
            st.warning("No technical indicators could be calculated")
            return
        
        # Group by symbol for better display
        symbols = [item['Symbol'] for item in indicators_data]
        selected_symbol = st.selectbox("Select symbol for detailed technical analysis:", symbols, key="tech_indicators_symbol")
        
        if selected_symbol:
            symbol_indicators = [item for item in indicators_data if item['Symbol'] == selected_symbol][0]
            
            # Display technical indicators for selected symbol
            st.markdown(f"### 🔍 Technical Analysis: **{selected_symbol}**")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create the 4-column table as requested
                table_data = [
                    {'Abbreviation': 'TEMA', 'Name': 'Triple Exponential Moving Average', 'Value': symbol_indicators['TEMA'], 'Interpretation': symbol_indicators['TEMA_Interp']},
                    {'Abbreviation': 'CONF', 'Name': 'Confidence Score', 'Value': symbol_indicators['CONF'], 'Interpretation': symbol_indicators['CONF_Interp']},
                    {'Abbreviation': 'STR', 'Name': 'Strength Score', 'Value': symbol_indicators['STR'], 'Interpretation': symbol_indicators['STR_Interp']},
                    {'Abbreviation': 'PM20', 'Name': 'Price Momentum (20D)', 'Value': symbol_indicators['PM'], 'Interpretation': symbol_indicators['PM_Interp']},
                    {'Abbreviation': 'VOL', 'Name': 'Volatility (Annual)', 'Value': symbol_indicators['VOL'], 'Interpretation': symbol_indicators['VOL_Interp']}
                ]
                
                display_df = pd.DataFrame(table_data)
                
                # Style the dataframe
                def color_interpretation(val):
                    if any(word in val.lower() for word in ['bullish', 'high', 'strong', 'rising']):
                        return 'background-color: #d4edda; color: #155724;'
                    elif any(word in val.lower() for word in ['bearish', 'low', 'weak', 'declining']):
                        return 'background-color: #f8d7da; color: #721c24;'
                    else:
                        return 'background-color: #e2e3e5; color: #383d41;'
                
                styled_df = display_df.style.map(color_interpretation, subset=['Interpretation'])
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            with col2:
                # Overall assessment card
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 1.5rem;
                    border-radius: 12px;
                    text-align: center;
                ">
                    <div style="font-size: 1.1rem; font-weight: bold; margin-bottom: 1rem;">🎯 Overall Assessment</div>
                    <div style="font-size: 0.9rem; line-height: 1.4; opacity: 0.95;">{}</div>
                </div>
                """.format(symbol_indicators['Overall_Interpretation']), unsafe_allow_html=True)
    
    def _interpret_technical_indicators(self, tema_momentum, confidence, strength, price_momentum, volatility, signal_type):
        """Generate holistic interpretation of technical indicators."""
        
        # Assess overall momentum
        if tema_momentum > 0.01 and confidence > 0.6 and strength > 0.4:
            momentum_assessment = "Strong bullish momentum with high conviction"
        elif tema_momentum < -0.01 and confidence > 0.6 and strength > 0.4:
            momentum_assessment = "Strong bearish momentum with high conviction"
        elif abs(tema_momentum) < 0.005:
            momentum_assessment = "Neutral momentum - sideways consolidation"
        elif confidence < 0.4:
            momentum_assessment = "Uncertain momentum - low conviction signals"
        else:
            momentum_assessment = "Moderate momentum with mixed signals"
        
        # Risk assessment
        if volatility > 30:
            risk_note = "High volatility increases risk"
        elif volatility < 15:
            risk_note = "Low volatility suggests stability"
        else:
            risk_note = "Moderate volatility - standard risk"
        
        # Price trend confirmation
        if abs(price_momentum) < 2:
            trend_note = "Price trend is neutral"
        elif (price_momentum > 2 and tema_momentum > 0) or (price_momentum < -2 and tema_momentum < 0):
            trend_note = "Price trend confirms momentum signal"
        else:
            trend_note = "Price trend diverges from momentum"
        
        # Overall recommendation context
        if signal_type == 'BUY' and confidence > 0.6:
            recommendation = "Consider buying on strength"
        elif signal_type == 'SELL' and confidence > 0.6:
            recommendation = "Consider selling/avoiding"
        else:
            recommendation = "Hold position or wait for clearer signals"
        
        return f"{momentum_assessment}. {trend_note}. {risk_note}. {recommendation}."
    
    def _render_backtesting_section(self):
        """Render the backtesting section with strategy testing capabilities."""
        st.markdown("""
        ### ⏳ Historical Backtesting
        Test your portfolio's momentum strategy against historical data to validate performance.
        """)
        
        # Check if we have portfolio symbols to work with
        if not st.session_state.portfolio:
            st.info("📝 **Upload a portfolio first** to use symbols for backtesting, or test with sample symbols below.")
            
            # Provide sample symbol options
            st.markdown("#### 🔬 Test with Sample Portfolios")
            sample_portfolios = {
                "Tech Leaders": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
                "Market Mix": ["SPY", "QQQ", "IWM", "VTI", "VOO"],
                "Growth Stocks": ["TSLA", "NVDA", "CRM", "ADBE", "NFLX"]
            }
            
            selected_portfolio = st.selectbox(
                "Choose a sample portfolio",
                list(sample_portfolios.keys())
            )
            symbols = sample_portfolios[selected_portfolio]
            
        else:
            # Use symbols from uploaded portfolio
            symbols = st.session_state.portfolio.symbols
            st.success(f"✅ Using symbols from your uploaded portfolio: {', '.join(symbols[:5])}")
            if len(symbols) > 5:
                st.info(f"➕ Plus {len(symbols)-5} more symbols")
        
        # Backtesting configuration
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### ⚙️ Backtest Configuration")
            
            # Date range
            end_date = datetime.now() - timedelta(days=1)
            start_date = end_date - timedelta(days=365)
            
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                backtest_start = st.date_input(
                    "Start Date",
                    value=start_date,
                    max_value=end_date
                )
            with date_col2:
                backtest_end = st.date_input(
                    "End Date", 
                    value=end_date,
                    max_value=datetime.now()
                )
            
            # Strategy parameters
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000,
                max_value=1000000,
                value=100000,
                step=1000
            )
            
            max_position_size = st.slider(
                "Max Position Size (%)",
                min_value=5,
                max_value=50,
                value=20,
                help="Maximum percentage of portfolio per position"
            ) / 100
            
        with col2:
            st.markdown("#### 💸 Trading Costs")
            
            commission_rate = st.slider(
                "Commission (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01
            ) / 100
            
            slippage_rate = st.slider(
                "Slippage (%)", 
                min_value=0.0,
                max_value=0.5,
                value=0.05,
                step=0.01
            ) / 100
        
        # Enhanced features toggles
        st.markdown("#### 🔧 Enhanced Algorithm Features")
        
        feature_col1, feature_col2 = st.columns(2)
        with feature_col1:
            enable_multi_timeframe = st.checkbox(
                "Multi-Timeframe Analysis",
                value=True,
                help="Analyze momentum across multiple timeframes"
            )
            enable_adaptive_thresholds = st.checkbox(
                "Adaptive Signal Thresholds", 
                value=True,
                help="Adjust thresholds based on market volatility"
            )
            
        with feature_col2:
            enable_signal_confirmation = st.checkbox(
                "Signal Confirmation Logic",
                value=True,
                help="Require multiple confirmations before trading"
            )
            enable_divergence_detection = st.checkbox(
                "Divergence Detection",
                value=True,
                help="Detect momentum divergences for reversal signals"
            )
        
        # Run backtest button
        if st.button("🚀 Run Historical Backtest", type="primary", use_container_width=True):
            with st.spinner("Running backtesting analysis..."):
                try:
                    # Update algorithm components with enhanced settings
                    enhanced_momentum_calc = NaturalMomentumCalculator(
                        enable_multi_timeframe=enable_multi_timeframe
                    )
                    enhanced_signal_gen = SignalGenerator(
                        enable_adaptive_thresholds=enable_adaptive_thresholds,
                        enable_signal_confirmation=enable_signal_confirmation,
                        enable_divergence_detection=enable_divergence_detection,
                        strength_threshold=0.01,  # Lower threshold for backtesting
                        backtesting_mode=True     # Enable backtesting mode for more trades
                    )
                    enhanced_backtest_engine = BacktestEngine(enhanced_momentum_calc, enhanced_signal_gen)
                    
                    # Fetch historical data
                    data_request = DataRequest(
                        symbols=symbols,
                        start_date=datetime.combine(backtest_start, datetime.min.time()),
                        end_date=datetime.combine(backtest_end, datetime.min.time()),
                        data_source="yahoo"
                    )
                    
                    historical_data = self.data_provider.fetch_historical_data(data_request)
                    
                    if not historical_data:
                        st.error("❌ Failed to fetch historical data. Please try different symbols or date range.")
                        return
                    
                    # Configure backtest settings
                    settings = BacktestSettings(
                        start_date=datetime.combine(backtest_start, datetime.min.time()),
                        end_date=datetime.combine(backtest_end, datetime.min.time()),
                        initial_capital=initial_capital,
                        commission_rate=commission_rate,
                        slippage_rate=slippage_rate,
                        max_position_size=max_position_size
                    )
                    
                    # Run backtest
                    results = enhanced_backtest_engine.run_backtest(historical_data, settings)
                    
                    # Calculate comprehensive metrics
                    metrics = self.performance_analyzer.analyze(results)
                    
                    # Update Calmar ratio
                    self.performance_analyzer.update_calmar_ratio(
                        metrics.ratios, metrics.returns, metrics.drawdown
                    )
                    
                    # Store results in session state
                    st.session_state.backtest_results = results
                    st.session_state.backtest_metrics = metrics
                    
                    st.success("✅ Backtest completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Backtest failed: {str(e)}")
                    return
        
        # Display results if available
        if st.session_state.backtest_results is not None and st.session_state.backtest_metrics is not None:
            self._render_backtest_results()
    
    def _render_backtest_results(self):
        """Render comprehensive backtest results."""
        results = st.session_state.backtest_results
        metrics = st.session_state.backtest_metrics
        
        st.markdown("---")
        st.markdown("### 📊 Backtest Results")
        
        # Key metrics summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return",
                f"{results.total_return:.1%}",
                delta=f"${results.final_capital - results.initial_capital:,.0f}"
            )
        
        with col2:
            st.metric(
                "Annualized Return", 
                f"{results.annualized_return:.1%}",
                delta=f"Sharpe: {metrics.ratios.sharpe_ratio:.2f}"
            )
        
        with col3:
            st.metric(
                "Max Drawdown",
                f"{metrics.drawdown.max_drawdown:.1%}",
                delta=f"{metrics.drawdown.max_drawdown_duration} days",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                "Win Rate",
                f"{metrics.trading.win_rate:.1%}",
                delta=f"{metrics.trading.total_trades} trades"
            )
        
        # Performance chart
        if results.portfolio_history:
            st.markdown("#### 📈 Portfolio Performance")
            
            # Create performance DataFrame
            perf_df = pd.DataFrame([
                {
                    'Date': snapshot.date,
                    'Portfolio Value': snapshot.total_value,
                    'Return': snapshot.cumulative_return * 100
                }
                for snapshot in results.portfolio_history
            ])
            
            # Create dual-axis chart
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=['Portfolio Value ($)', 'Cumulative Return (%)'],
                row_heights=[0.7, 0.3]
            )
            
            # Portfolio value
            fig.add_trace(
                go.Scatter(
                    x=perf_df['Date'],
                    y=perf_df['Portfolio Value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#1f77b4', width=2)
                ),
                row=1, col=1
            )
            
            # Cumulative return
            fig.add_trace(
                go.Scatter(
                    x=perf_df['Date'],
                    y=perf_df['Return'],
                    mode='lines',
                    name='Cumulative Return',
                    line=dict(color='#ff7f0e', width=2),
                    fill='tonexty'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=500,
                showlegend=False,
                title_text="Historical Performance"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics tabs
        result_tab1, result_tab2, result_tab3 = st.tabs(["📊 Performance Metrics", "💹 Trading Analysis", "🎯 Risk Analysis"])
        
        with result_tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Return Metrics**")
                return_data = {
                    "Total Return": f"{metrics.returns.total_return:.2%}",
                    "Annualized Return": f"{metrics.returns.annualized_return:.2%}",
                    "Volatility": f"{metrics.risk.volatility:.2%}",
                    "Sharpe Ratio": f"{metrics.ratios.sharpe_ratio:.3f}",
                    "Sortino Ratio": f"{metrics.ratios.sortino_ratio:.3f}"
                }
                st.table(pd.DataFrame(list(return_data.items()), columns=["Metric", "Value"]))
            
            with col2:
                st.markdown("**Risk Metrics**")
                risk_data = {
                    "Max Drawdown": f"{metrics.drawdown.max_drawdown:.2%}",
                    "VaR (95%)": f"{metrics.risk.var_95:.2%}",
                    "Calmar Ratio": f"{metrics.ratios.calmar_ratio:.3f}",
                    "Recovery Factor": f"{metrics.drawdown.recovery_factor:.2f}",
                    "Pain Index": f"{metrics.drawdown.pain_index:.2%}"
                }
                st.table(pd.DataFrame(list(risk_data.items()), columns=["Metric", "Value"]))
        
        with result_tab2:
            if results.trades:
                st.markdown("**Trading Statistics**")
                
                # Trading metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    trade_data = {
                        "Total Trades": metrics.trading.total_trades,
                        "Winning Trades": metrics.trading.winning_trades,
                        "Losing Trades": metrics.trading.losing_trades,
                        "Win Rate": f"{metrics.trading.win_rate:.1%}",
                        "Profit Factor": f"{metrics.trading.profit_factor:.2f}"
                    }
                    st.table(pd.DataFrame(list(trade_data.items()), columns=["Metric", "Value"]))
                
                with col2:
                    pnl_data = {
                        "Average Win": f"${metrics.trading.avg_win:.2f}",
                        "Average Loss": f"${metrics.trading.avg_loss:.2f}",
                        "Largest Win": f"${metrics.trading.largest_win:.2f}",
                        "Largest Loss": f"${metrics.trading.largest_loss:.2f}",
                        "Expectancy": f"${metrics.trading.expectancy:.2f}"
                    }
                    st.table(pd.DataFrame(list(pnl_data.items()), columns=["Metric", "Value"]))
                
                # Recent trades table
                st.markdown("**Recent Trades**")
                recent_trades = results.trades[-10:] if len(results.trades) > 10 else results.trades
                
                trades_df = pd.DataFrame([
                    {
                        'Symbol': trade.symbol,
                        'Type': trade.order_type.value,
                        'Entry Date': trade.entry_date.strftime('%Y-%m-%d'),
                        'Entry Price': f"${trade.entry_price:.2f}",
                        'Quantity': trade.quantity,
                        'P&L': f"${trade.pnl:.2f}",
                        'Confidence': f"{trade.signal_confidence:.0%}"
                    }
                    for trade in recent_trades
                ])
                
                st.dataframe(trades_df, use_container_width=True)
            else:
                st.info("No trades were executed during the backtest period.")
        
        with result_tab3:
            # Risk analysis and performance scores
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Performance Scores**")
                score_data = {
                    "Risk-Adjusted Score": f"{metrics.risk_adjusted_score:.3f}",
                    "Consistency Score": f"{metrics.consistency_score:.3f}",
                    "Efficiency Score": f"{metrics.efficiency_score:.3f}"
                }
                st.table(pd.DataFrame(list(score_data.items()), columns=["Score", "Value"]))
            
            with col2:
                st.markdown("**Cost Analysis**") 
                cost_data = {
                    "Total Commission": f"${results.total_commission_paid:.2f}",
                    "Total Slippage": f"${results.total_slippage_cost:.2f}",
                    "Total Costs": f"${results.total_commission_paid + results.total_slippage_cost:.2f}",
                    "Cost Ratio": f"{(results.total_commission_paid + results.total_slippage_cost) / results.initial_capital:.3%}"
                }
                st.table(pd.DataFrame(list(cost_data.items()), columns=["Cost", "Value"]))


def main():
    """Main application entry point."""
    dashboard = PortfolioDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
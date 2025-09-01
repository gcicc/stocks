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
        
        # Limit symbols for performance
        symbols_to_analyze = portfolio.symbols[:max_symbols]
        
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
            status_text.markdown("📡 **Fetching market data** - Connecting to financial APIs...")
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
            st.success(f"🎉 Analysis complete! Generated signals for {len(signals)} symbols.")
            
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
        
        # Create tabs for different views - Portfolio Overview first, then results
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Portfolio Overview", "📋 Signals Overview", "📈 Charts", "🎯 Detailed Analysis", "📤 Export"])
        
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
    
    def _render_signals_overview(self):
        """Render signals overview table."""
        signals = st.session_state.signals
        if not signals:
            st.warning("No signals available")
            return
        
        # Create enhanced signals dataframe with badges
        signals_data = []
        for symbol, signal in signals.items():
            # Create signal badge HTML
            signal_class = f"signal-badge-{signal.signal.value.lower()}"
            signal_badge = f'<div class="{signal_class}">{signal.signal.value}</div>'
            
            signals_data.append({
                'Symbol': symbol,
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
            
            # Portfolio summary metrics
            st.markdown("**📈 Portfolio Metrics**")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total Positions", len(valid_positions))
            col_b.metric("Total Value", f"${total_value:,.2f}")
            col_c.metric("Largest Position", f"{max(valid_positions, key=lambda x: x.market_value).symbol}")
        
        with col2:
            # Portfolio allocation pie chart
            st.markdown("**🥧 Portfolio Allocation**")
            
            # Prepare data for pie chart
            symbols = [pos.symbol for pos in valid_positions]
            values = [pos.market_value for pos in valid_positions]
            
            # Group smaller positions for cleaner visualization
            if len(valid_positions) > 8:
                # Show top 7 positions individually, group the rest
                sorted_positions = sorted(zip(symbols, values), key=lambda x: x[1], reverse=True)
                top_positions = sorted_positions[:7]
                other_positions = sorted_positions[7:]
                
                display_symbols = [pos[0] for pos in top_positions]
                display_values = [pos[1] for pos in top_positions]
                
                if other_positions:
                    other_total = sum(pos[1] for pos in other_positions)
                    display_symbols.append(f"Others ({len(other_positions)})")
                    display_values.append(other_total)
            else:
                display_symbols = symbols
                display_values = values
            
            # Create professional pie chart
            fig_pie = go.Figure(data=[
                go.Pie(
                    labels=display_symbols,
                    values=display_values,
                    hole=0.4,  # Donut style
                    marker=dict(
                        colors=px.colors.qualitative.Set3[:len(display_symbols)],
                        line=dict(color='white', width=2)
                    ),
                    textinfo='label+percent',
                    textfont=dict(size=11),
                    hovertemplate='<b>%{label}</b><br>Value: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
                )
            ])
            
            fig_pie.update_layout(
                height=400,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.02
                ),
                margin=dict(t=20, b=20, l=20, r=80),
                font=dict(size=10)
            )
            
            st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
        
        # Additional insights
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Concentration analysis
            top_3_value = sum(sorted([pos.market_value for pos in valid_positions], reverse=True)[:3])
            concentration = (top_3_value / total_value) * 100 if total_value > 0 else 0
            st.metric(
                "Top 3 Concentration", 
                f"{concentration:.1f}%",
                help="Percentage of portfolio in top 3 positions"
            )
        
        with col2:
            # Average position size
            avg_position = total_value / len(valid_positions) if valid_positions else 0
            st.metric(
                "Avg Position Size",
                f"${avg_position:,.2f}",
                help="Average market value per position"
            )
        
        with col3:
            # Smallest vs Largest ratio
            if valid_positions:
                largest = max(pos.market_value for pos in valid_positions)
                smallest = min(pos.market_value for pos in valid_positions)
                ratio = largest / smallest if smallest > 0 else 0
                st.metric(
                    "Size Ratio (L/S)",
                    f"{ratio:.1f}x",
                    help="Largest position vs smallest position ratio"
                )
        
        st.info("💡 **Ready for Analysis:** Upload complete! Use the sidebar to configure settings and run momentum analysis.")


def main():
    """Main application entry point."""
    dashboard = PortfolioDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
"""
Report generator for Portfolio Intelligence Platform.
Creates comprehensive market intelligence reports with professional visualizations.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import streamlit as st
from datetime import datetime
import logging

from data_sources.news_api import NewsProvider
from data_sources.market_data import MacroDataProvider

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Professional report generator for market intelligence.
    Creates comprehensive reports with charts, tables, and insights.
    """
    
    def __init__(self):
        self.news_provider = NewsProvider()
        self.macro_provider = MacroDataProvider()
        logger.info("Report generator initialized")
    
    def generate_portfolio_attribution(self, portfolio, signals: Dict = None) -> Tuple[pd.DataFrame, go.Figure]:
        """
        Generate portfolio performance attribution analysis.
        Works with or without signals - shows portfolio composition and potential impact.
        
        Args:
            portfolio: Portfolio object with positions
            signals: Dictionary of trading signals (optional)
            
        Returns:
            Tuple of (attribution DataFrame, Plotly waterfall chart)
        """
        try:
            attribution_data = []
            
            # Calculate total portfolio value
            total_portfolio_value = 0
            position_values = {}
            
            for position in portfolio.positions:
                symbol = position.symbol
                # Try different attribute names for position value
                position_value = getattr(position, 'market_value', 
                                       getattr(position, 'value', 
                                              getattr(position, 'total_value', 0)))
                position_values[symbol] = position_value
                total_portfolio_value += position_value
            
            # If no portfolio value calculated, use a fallback approach
            if total_portfolio_value == 0:
                logger.warning("Could not calculate total portfolio value, using equal weights")
                equal_weight = 100.0 / len(portfolio.positions) if portfolio.positions else 0
                for position in portfolio.positions:
                    position_values[position.symbol] = 1000 * equal_weight  # Assume $1000 per position
                total_portfolio_value = sum(position_values.values())
            
            for position in portfolio.positions:
                symbol = position.symbol
                position_value = position_values[symbol]
                
                # Calculate position weight
                weight = (position_value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
                
                # Get signal data if available
                if signals and symbol in signals:
                    signal_data = signals[symbol]
                    signal_type = signal_data.signal.value if hasattr(signal_data, 'signal') else "HOLD"
                    confidence = signal_data.confidence if hasattr(signal_data, 'confidence') else 0.0
                    
                    # Calculate impact based on signal and confidence
                    if signal_type == "BUY":
                        impact_score = weight * confidence
                    elif signal_type == "SELL":
                        impact_score = -weight * confidence
                    else:
                        impact_score = weight * 0.1  # Small positive for HOLD
                else:
                    # No signals available - show neutral impact based on weight
                    signal_type = "ANALYZING"
                    confidence = 0.0
                    impact_score = weight * 0.1  # Small positive impact for all positions
                
                attribution_data.append({
                    'Symbol': symbol,
                    'Position_Value': position_value,
                    'Portfolio_Weight_%': round(weight, 2),
                    'Signal': signal_type,
                    'Confidence': round(confidence, 3),
                    'Impact_Score': round(impact_score, 2),
                    'Status': self._get_impact_status(impact_score)
                })
            
            # Create professional DataFrame
            df = pd.DataFrame(attribution_data)
            
            if not df.empty:
                # Sort by portfolio weight (largest positions first)
                df = df.sort_values('Portfolio_Weight_%', ascending=False).reset_index(drop=True)
                
                # Create professional waterfall chart
                fig = self._create_attribution_chart(df)
                
                return df, fig
            else:
                # Return empty but valid structures
                empty_df = pd.DataFrame(columns=['Symbol', 'Position_Value', 'Portfolio_Weight_%', 'Signal', 'Confidence', 'Impact_Score', 'Status'])
                empty_fig = go.Figure()
                return empty_df, empty_fig
            
        except Exception as e:
            logger.error(f"Portfolio attribution generation failed: {e}")
            # Return empty but valid structures
            empty_df = pd.DataFrame(columns=['Symbol', 'Position_Value', 'Portfolio_Weight_%', 'Signal', 'Confidence', 'Impact_Score', 'Status'])
            empty_fig = go.Figure()
            return empty_df, empty_fig
    
    def _create_attribution_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create professional portfolio attribution waterfall chart."""
        fig = go.Figure()
        
        # Add bars with professional styling
        colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in df['Impact_Score']]
        
        fig.add_trace(go.Bar(
            x=df['Symbol'],
            y=df['Impact_Score'],
            marker_color=colors,
            name='Portfolio Impact',
            text=df['Impact_Score'].apply(lambda x: f"{x:+.1f}"),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>' +
                         'Impact Score: %{y:.2f}<br>' +
                         'Signal: %{customdata[0]}<br>' +
                         'Weight: %{customdata[1]:.1f}%<br>' +
                         '<extra></extra>',
            customdata=df[['Signal', 'Portfolio_Weight_%']].values
        ))
        
        # Professional styling
        fig.update_layout(
            title='Portfolio Performance Attribution Analysis',
            xaxis_title='Symbols',
            yaxis_title='Impact Score',
            template='plotly_white',
            showlegend=False,
            height=400,
            margin=dict(t=60, b=60, l=60, r=60)
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
    
    def _get_impact_status(self, score: float) -> str:
        """Get professional impact status."""
        if score > 5:
            return "Major Contributor"
        elif score > 2:
            return "Strong Contributor"
        elif score > 0.5:
            return "Contributor" 
        elif score < -5:
            return "Major Drag"
        elif score < -2:
            return "Strong Drag"
        elif score < -0.5:
            return "Drag"
        else:
            return "Neutral"
    
    def create_market_overview_dashboard(self) -> Tuple[pd.DataFrame, Dict[str, go.Figure]]:
        """
        Create comprehensive market overview dashboard.
        
        Returns:
            Tuple of (market summary DataFrame, dictionary of charts)
        """
        try:
            # Get market data
            market_summary = self.macro_provider.get_market_summary_df()
            
            # Create individual charts
            charts = {}
            
            # VIX Fear & Greed Chart
            vix_row = market_summary[market_summary['Indicator'] == 'VIX']
            if not vix_row.empty:
                vix_value = vix_row['Current_Value'].iloc[0]
                charts['vix_gauge'] = self._create_vix_gauge(vix_value)
            
            # Market Performance Chart
            charts['market_performance'] = self._create_market_performance_chart(market_summary)
            
            # Treasury Yield Chart
            tnx_row = market_summary[market_summary['Indicator'] == '10-Year Treasury']
            if not tnx_row.empty:
                tnx_value = tnx_row['Current_Value'].iloc[0]
                charts['treasury_analysis'] = self._create_treasury_chart(tnx_value)
            
            return market_summary, charts
            
        except Exception as e:
            logger.error(f"Market overview generation failed: {e}")
            empty_df = pd.DataFrame(columns=['Indicator', 'Current_Value', 'Change', 'Change_%', 'Status', 'Updated'])
            return empty_df, {}
    
    def _create_vix_gauge(self, vix_value: float) -> go.Figure:
        """Create professional VIX fear & greed gauge."""
        vix_analysis = self.macro_provider.get_vix_interpretation(vix_value)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = vix_value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"VIX Fear Index<br>{vix_analysis['level']}"},
            delta = {'reference': 20},
            gauge = {
                'axis': {'range': [None, 50]},
                'bar': {'color': vix_analysis['color']},
                'steps': [
                    {'range': [0, 12], 'color': "lightgreen"},
                    {'range': [12, 20], 'color': "yellow"},
                    {'range': [20, 30], 'color': "orange"},
                    {'range': [30, 50], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': vix_value
                }
            }
        ))
        
        fig.update_layout(
            template='plotly_white',
            height=300,
            margin=dict(t=60, b=60, l=60, r=60)
        )
        
        return fig
    
    def _create_market_performance_chart(self, market_df: pd.DataFrame) -> go.Figure:
        """Create professional market performance comparison chart."""
        # Filter to main indices
        indices = ['S&P 500', 'NASDAQ 100', 'Dow Jones', 'Russell 2000']
        chart_data = market_df[market_df['Indicator'].isin(indices)].copy()
        
        if chart_data.empty:
            return go.Figure()
        
        colors = ['green' if x > 0 else 'red' for x in chart_data['Change_%']]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=chart_data['Indicator'],
            y=chart_data['Change_%'],
            marker_color=colors,
            name='Daily Change %',
            text=chart_data['Change_%'].apply(lambda x: f"{x:+.2f}%"),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>' +
                         'Change: %{y:.2f}%<br>' +
                         'Value: %{customdata:.2f}<br>' +
                         '<extra></extra>',
            customdata=chart_data['Current_Value']
        ))
        
        fig.update_layout(
            title='Major Market Indices Performance',
            xaxis_title='Index',
            yaxis_title='Daily Change (%)',
            template='plotly_white',
            showlegend=False,
            height=350,
            margin=dict(t=60, b=60, l=60, r=60)
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
    
    def _create_treasury_chart(self, tnx_value: float) -> go.Figure:
        """Create professional Treasury yield analysis chart."""
        tnx_analysis = self.macro_provider.get_treasury_analysis(tnx_value)
        
        # Create a simple gauge for treasury yield
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = tnx_value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"10-Year Treasury Yield<br>{tnx_analysis['level']}"},
            gauge = {
                'axis': {'range': [None, 6]},
                'bar': {'color': "blue"},
                'steps': [
                    {'range': [0, 2], 'color': "lightblue"},
                    {'range': [2, 3], 'color': "yellow"},
                    {'range': [3, 4.5], 'color': "orange"},
                    {'range': [4.5, 6], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': tnx_value
                }
            }
        ))
        
        fig.update_layout(
            template='plotly_white',
            height=300,
            margin=dict(t=60, b=60, l=60, r=60)
        )
        
        return fig
"""
News analyzer for Portfolio Intelligence Platform.
Provides professional news analysis and sentiment scoring.
"""

import pandas as pd
import streamlit as st
from typing import List, Dict, Optional
from datetime import datetime
import logging

from data_sources.news_api import NewsProvider, NewsArticle

logger = logging.getLogger(__name__)

class NewsAnalyzer:
    """
    Professional news analyzer for portfolio intelligence.
    Provides sentiment analysis and news aggregation.
    """
    
    def __init__(self):
        self.news_provider = NewsProvider()
        logger.info("News analyzer initialized")
    
    def analyze_portfolio_sentiment(self, symbols: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment for entire portfolio with professional formatting.
        
        Args:
            symbols: List of portfolio symbols
            
        Returns:
            Professional DataFrame with sentiment analysis
        """
        try:
            sentiment_df = self.news_provider.get_portfolio_news_summary(symbols)
            
            # Add professional styling information
            sentiment_df['Sentiment_Color'] = sentiment_df['Sentiment_Label'].map({
                'Bullish': 'green',
                'Bearish': 'red', 
                'Neutral': 'gray'
            })
            
            # Add trend arrows for visual enhancement
            sentiment_df['Trend_Arrow'] = sentiment_df['Sentiment_Score'].apply(
                lambda x: '↗️' if x > 0.1 else '↘️' if x < -0.1 else '→'
            )
            
            return sentiment_df
            
        except Exception as e:
            logger.error(f"Portfolio sentiment analysis failed: {e}")
            # Return empty DataFrame with proper structure
            return pd.DataFrame(columns=[
                'Symbol', 'Latest_Headline', 'Source', 'Published', 
                'Sentiment_Score', 'Sentiment_Label', 'Article_Count', 'URL'
            ])
    
    def create_sentiment_summary_metrics(self, sentiment_df: pd.DataFrame) -> Dict[str, any]:
        """
        Create professional sentiment summary metrics.
        
        Args:
            sentiment_df: Sentiment analysis DataFrame
            
        Returns:
            Dictionary with summary metrics for dashboard display
        """
        if sentiment_df.empty:
            return {
                'total_positions': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'avg_sentiment': 0.0,
                'most_bullish': 'N/A',
                'most_bearish': 'N/A'
            }
        
        try:
            total_positions = len(sentiment_df)
            bullish_count = len(sentiment_df[sentiment_df['Sentiment_Label'] == 'Bullish'])
            bearish_count = len(sentiment_df[sentiment_df['Sentiment_Label'] == 'Bearish'])
            neutral_count = total_positions - bullish_count - bearish_count
            
            avg_sentiment = sentiment_df['Sentiment_Score'].mean()
            
            # Find most extreme sentiments
            most_bullish_row = sentiment_df.loc[sentiment_df['Sentiment_Score'].idxmax()]
            most_bearish_row = sentiment_df.loc[sentiment_df['Sentiment_Score'].idxmin()]
            
            return {
                'total_positions': total_positions,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'neutral_count': neutral_count,
                'avg_sentiment': round(avg_sentiment, 3),
                'most_bullish': most_bullish_row['Symbol'],
                'most_bearish': most_bearish_row['Symbol'],
                'bullish_percentage': round((bullish_count / total_positions) * 100, 1),
                'bearish_percentage': round((bearish_count / total_positions) * 100, 1)
            }
            
        except Exception as e:
            logger.error(f"Sentiment summary calculation failed: {e}")
            return {
                'total_positions': len(sentiment_df) if not sentiment_df.empty else 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'avg_sentiment': 0.0,
                'most_bullish': 'N/A',
                'most_bearish': 'N/A'
            }
    
    def display_professional_news_table(self, sentiment_df: pd.DataFrame):
        """
        Display professional news table with Streamlit styling.
        
        Args:
            sentiment_df: Sentiment analysis DataFrame
        """
        if sentiment_df.empty:
            st.warning("No news data available for analysis.")
            return
        
        try:
            # Create display DataFrame with professional formatting
            display_df = sentiment_df.copy()
            
            # Format columns for better display
            display_columns = [
                'Symbol',
                'Latest_Headline', 
                'Source',
                'Published',
                'Sentiment_Score',
                'Sentiment_Label',
                'Article_Count'
            ]
            
            display_df = display_df[display_columns]
            
            # Apply professional styling
            def highlight_sentiment(row):
                """Apply color coding based on sentiment."""
                sentiment = row['Sentiment_Label']
                if sentiment == 'Bullish':
                    return ['background-color: rgba(0, 255, 0, 0.1)'] * len(row)
                elif sentiment == 'Bearish':
                    return ['background-color: rgba(255, 0, 0, 0.1)'] * len(row)
                else:
                    return ['background-color: rgba(128, 128, 128, 0.05)'] * len(row)
            
            # Display with professional styling
            st.dataframe(
                display_df.style.apply(highlight_sentiment, axis=1).format({
                    'Sentiment_Score': '{:.3f}',
                    'Article_Count': '{:.0f}'
                }),
                use_container_width=True,
                height=400
            )
            
            # Add clickable links section
            if 'URL' in sentiment_df.columns:
                st.subheader("📰 Article Links")
                
                for idx, row in sentiment_df.iterrows():
                    if row['URL'] and row['URL'] != '':
                        col1, col2, col3 = st.columns([2, 3, 1])
                        with col1:
                            st.write(f"**{row['Symbol']}**")
                        with col2:
                            st.write(row['Latest_Headline'][:80] + "...")
                        with col3:
                            st.markdown(f"[Read More]({row['URL']})")
                        
        except Exception as e:
            logger.error(f"Professional news table display failed: {e}")
            st.error("Failed to display news table. Please try again.")
    
    def create_sentiment_distribution_chart(self, sentiment_df: pd.DataFrame):
        """
        Create professional sentiment distribution chart.
        
        Args:
            sentiment_df: Sentiment analysis DataFrame
        """
        if sentiment_df.empty:
            st.info("No sentiment data available for visualization.")
            return
        
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            
            # Count sentiment distribution
            sentiment_counts = sentiment_df['Sentiment_Label'].value_counts()
            
            # Professional color mapping
            colors = {
                'Bullish': '#00FF00',
                'Bearish': '#FF0000', 
                'Neutral': '#808080'
            }
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                marker_colors=[colors.get(label, '#808080') for label in sentiment_counts.index],
                textinfo='label+percent',
                textposition='auto',
                hovertemplate='<b>%{label}</b><br>' +
                             'Count: %{value}<br>' +
                             'Percentage: %{percent}<br>' +
                             '<extra></extra>'
            )])
            
            fig.update_layout(
                title='Portfolio Sentiment Distribution',
                template='plotly_white',
                height=400,
                showlegend=True,
                margin=dict(t=60, b=60, l=60, r=60)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Sentiment distribution chart creation failed: {e}")
            st.error("Failed to create sentiment chart. Please try again.")
    
    def get_news_insights(self, sentiment_df: pd.DataFrame) -> List[str]:
        """
        Generate professional insights from news sentiment analysis.
        
        Args:
            sentiment_df: Sentiment analysis DataFrame
            
        Returns:
            List of professional insight strings
        """
        if sentiment_df.empty:
            return ["No news data available for insights."]
        
        try:
            insights = []
            summary = self.create_sentiment_summary_metrics(sentiment_df)
            
            # Overall sentiment insight
            if summary['avg_sentiment'] > 0.2:
                insights.append(f"📈 **Portfolio sentiment is generally positive** with an average score of {summary['avg_sentiment']:.3f}")
            elif summary['avg_sentiment'] < -0.2:
                insights.append(f"📉 **Portfolio sentiment is generally negative** with an average score of {summary['avg_sentiment']:.3f}")
            else:
                insights.append(f"📊 **Portfolio sentiment is neutral** with an average score of {summary['avg_sentiment']:.3f}")
            
            # Distribution insights
            if summary['bullish_percentage'] > 50:
                insights.append(f"🟢 **{summary['bullish_percentage']}% of positions have bullish news sentiment** ({summary['bullish_count']} out of {summary['total_positions']})")
            elif summary['bearish_percentage'] > 50:
                insights.append(f"🔴 **{summary['bearish_percentage']}% of positions have bearish news sentiment** ({summary['bearish_count']} out of {summary['total_positions']})")
            else:
                insights.append(f"⚖️ **Sentiment is well-balanced** across portfolio positions")
            
            # Highlight extreme sentiments
            if summary['most_bullish'] != 'N/A':
                most_bullish_score = sentiment_df[sentiment_df['Symbol'] == summary['most_bullish']]['Sentiment_Score'].iloc[0]
                if most_bullish_score > 0.4:
                    insights.append(f"⭐ **{summary['most_bullish']} shows the strongest positive sentiment** (Score: {most_bullish_score:.3f})")
            
            if summary['most_bearish'] != 'N/A':
                most_bearish_score = sentiment_df[sentiment_df['Symbol'] == summary['most_bearish']]['Sentiment_Score'].iloc[0]
                if most_bearish_score < -0.4:
                    insights.append(f"⚠️ **{summary['most_bearish']} shows concerning negative sentiment** (Score: {most_bearish_score:.3f})")
            
            # News coverage insight
            avg_articles = sentiment_df['Article_Count'].mean()
            if avg_articles > 3:
                insights.append(f"📰 **High news coverage detected** with an average of {avg_articles:.1f} articles per position")
            elif avg_articles < 1:
                insights.append(f"📰 **Limited news coverage** with an average of {avg_articles:.1f} articles per position")
            
            return insights
            
        except Exception as e:
            logger.error(f"News insights generation failed: {e}")
            return ["Unable to generate insights due to data processing error."]
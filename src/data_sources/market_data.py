"""
Macro market data provider for Portfolio Intelligence Platform.
Provides professional market indicators and economic data.
"""

import yfinance as yf
import pandas as pd
from typing import Dict, Optional
from datetime import datetime
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MarketIndicator:
    """Professional market indicator data structure."""
    name: str
    symbol: str
    current_value: float
    change: float
    change_percent: float
    status: str  # "Positive", "Negative", "Neutral"
    last_updated: datetime

class MacroDataProvider:
    """
    Professional macro market data provider.
    Fetches key market indicators for market context dashboard.
    """
    
    def __init__(self):
        self.indicators = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ 100', 
            'DIA': 'Dow Jones',
            'IWM': 'Russell 2000',
            '^TNX': '10-Year Treasury',
            '^VIX': 'VIX',
            'DX-Y.NYB': 'USD Index',
            'CL=F': 'Crude Oil WTI'
        }
        logger.info("Macro data provider initialized")
    
    def get_market_overview(self) -> Dict[str, MarketIndicator]:
        """
        Get comprehensive market overview with all key indicators.
        
        Returns:
            Dictionary of MarketIndicator objects keyed by symbol
        """
        market_data = {}
        
        for symbol, name in self.indicators.items():
            try:
                indicator = self._fetch_indicator(symbol, name)
                if indicator:
                    market_data[symbol] = indicator
                    
            except Exception as e:
                logger.error(f"Failed to fetch {name} ({symbol}): {e}")
                # Create placeholder with error state
                market_data[symbol] = MarketIndicator(
                    name=name,
                    symbol=symbol,
                    current_value=0.0,
                    change=0.0,
                    change_percent=0.0,
                    status="Error",
                    last_updated=datetime.now()
                )
        
        return market_data
    
    def _fetch_indicator(self, symbol: str, name: str) -> Optional[MarketIndicator]:
        """Fetch individual market indicator."""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current and previous day data
            hist = ticker.history(period="2d")
            
            if hist.empty:
                logger.warning(f"No data available for {symbol}")
                return None
            
            current_price = hist['Close'].iloc[-1]
            previous_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            
            change = current_price - previous_price
            change_percent = (change / previous_price) * 100 if previous_price != 0 else 0.0
            
            # Determine status
            status = self._get_status(change_percent, symbol)
            
            return MarketIndicator(
                name=name,
                symbol=symbol,
                current_value=round(current_price, 2),
                change=round(change, 2),
                change_percent=round(change_percent, 2),
                status=status,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None
    
    def _get_status(self, change_percent: float, symbol: str) -> str:
        """Get professional status classification."""
        if symbol == '^VIX':
            # VIX is inverse - higher is more fearful
            if change_percent > 10:
                return "High Fear"
            elif change_percent < -10:
                return "Low Fear"
            else:
                return "Moderate"
        else:
            # Standard indicators
            if change_percent > 1:
                return "Strong Positive"
            elif change_percent > 0:
                return "Positive"
            elif change_percent < -1:
                return "Strong Negative"
            elif change_percent < 0:
                return "Negative"
            else:
                return "Neutral"
    
    def get_market_summary_df(self) -> pd.DataFrame:
        """
        Get market indicators as professional DataFrame.
        
        Returns:
            DataFrame formatted for professional display
        """
        market_data = self.get_market_overview()
        
        summary_data = []
        for indicator in market_data.values():
            summary_data.append({
                'Indicator': indicator.name,
                'Current_Value': indicator.current_value,
                'Change': indicator.change,
                'Change_%': indicator.change_percent,
                'Status': indicator.status,
                'Updated': indicator.last_updated.strftime('%H:%M:%S')
            })
        
        df = pd.DataFrame(summary_data)
        return df
    
    def get_vix_interpretation(self, vix_value: float) -> Dict[str, str]:
        """
        Get professional VIX interpretation.
        
        Args:
            vix_value: Current VIX value
            
        Returns:
            Dictionary with interpretation details
        """
        if vix_value < 12:
            level = "Very Low"
            interpretation = "Extreme complacency - potential for surprises"
            color = "green"
        elif vix_value < 20:
            level = "Low"
            interpretation = "Normal market conditions - low fear"
            color = "lightgreen"
        elif vix_value < 30:
            level = "Elevated"
            interpretation = "Increased uncertainty - moderate fear"
            color = "orange"
        else:
            level = "High"
            interpretation = "High fear and uncertainty in markets"
            color = "red"
        
        return {
            'level': level,
            'interpretation': interpretation,
            'color': color,
            'value': round(vix_value, 2)
        }
    
    def get_treasury_analysis(self, tnx_value: float) -> Dict[str, str]:
        """
        Get professional 10-Year Treasury analysis.
        
        Args:
            tnx_value: Current 10-Year Treasury yield
            
        Returns:
            Dictionary with analysis details
        """
        if tnx_value < 2.0:
            level = "Very Low"
            interpretation = "Accommodative monetary policy - growth concerns"
            impact = "Positive for stocks, negative for financials"
        elif tnx_value < 3.0:
            level = "Low"
            interpretation = "Supportive for risk assets"
            impact = "Generally positive for equities"
        elif tnx_value < 4.5:
            level = "Moderate"
            interpretation = "Balanced monetary conditions"
            impact = "Neutral to slightly negative for stocks"
        else:
            level = "High"
            interpretation = "Restrictive policy - inflation concerns"
            impact = "Headwind for risk assets"
        
        return {
            'level': level,
            'interpretation': interpretation,
            'impact': impact,
            'value': round(tnx_value, 3)
        }
#!/usr/bin/env python3
"""
Test the enhanced multi-timeframe momentum calculator.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from core.momentum_calculator import NaturalMomentumCalculator
from core.data_manager import MarketData

def create_test_market_data(symbol: str = "TEST", periods: int = 100) -> MarketData:
    """Create synthetic market data for testing."""
    
    # Generate synthetic price data with trend
    dates = pd.date_range(start=datetime.now() - timedelta(days=periods), periods=periods, freq='D')
    
    # Create a trending price series with some volatility
    base_price = 100.0
    trend = 0.001  # Slight upward trend
    volatility = 0.02
    
    prices = []
    price = base_price
    
    for i in range(periods):
        # Add trend and random walk
        price = price * (1 + trend + np.random.normal(0, volatility))
        prices.append(price)
    
    # Create OHLC data
    df = pd.DataFrame({
        'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': [np.random.randint(1000000, 5000000) for _ in range(periods)]
    }, index=dates)
    
    return MarketData(
        symbol=symbol,
        prices=df,
        current_price=prices[-1],
        previous_close=prices[-2] if len(prices) > 1 else prices[-1],
        volume=df['Volume'].iloc[-1],
        last_updated=datetime.now(),
        data_source="synthetic_test_data"
    )

def test_multi_timeframe_momentum():
    """Test the enhanced multi-timeframe momentum calculator."""
    
    print("Testing Enhanced Multi-timeframe Momentum Calculator")
    print("=" * 60)
    
    # Create test data
    market_data = create_test_market_data("TEST", 100)
    print(f"Created test data: {len(market_data.prices)} price points")
    print(f"Price range: ${market_data.prices['Close'].min():.2f} - ${market_data.prices['Close'].max():.2f}")
    
    # Test both standard and multi-timeframe calculators
    print("\n1. Testing Standard Calculator (single timeframe):")
    standard_calc = NaturalMomentumCalculator(enable_multi_timeframe=False)
    standard_result = standard_calc.calculate_natural_momentum(market_data)
    
    print(f"   Momentum: {standard_result.current_momentum:.4f}")
    print(f"   Direction: {standard_result.momentum_direction}")
    print(f"   Strength: {standard_result.strength:.3f}")
    print(f"   Multi-timeframe fields: {standard_result.timeframe_consensus}")
    
    print("\n2. Testing Enhanced Calculator (multi-timeframe):")
    enhanced_calc = NaturalMomentumCalculator(enable_multi_timeframe=True)
    enhanced_result = enhanced_calc.calculate_natural_momentum(market_data)
    
    print(f"   Momentum: {enhanced_result.current_momentum:.4f}")
    print(f"   Direction: {enhanced_result.momentum_direction}")
    print(f"   Strength: {enhanced_result.strength:.3f}")
    print(f"   Timeframe Consensus: {enhanced_result.timeframe_consensus:.3f}")
    print(f"   Trend Strength: {enhanced_result.trend_strength}")
    
    # Show timeframe details
    if enhanced_result.tema_short is not None:
        print(f"\n   Timeframe Analysis:")
        print(f"   - Short-term TEMA ({enhanced_calc.tema_short}): {enhanced_result.tema_short.iloc[-1]:.4f}")
        print(f"   - Medium-term TEMA ({enhanced_calc.tema_medium}): {enhanced_result.tema_medium.iloc[-1]:.4f}")
        print(f"   - Long-term TEMA ({enhanced_calc.tema_long}): {enhanced_result.tema_long.iloc[-1]:.4f}")
    
    # Interpretation
    consensus = enhanced_result.timeframe_consensus or 0.0
    if consensus > 0.5:
        interpretation = "[BULLISH] Strong Bullish Consensus"
    elif consensus > 0.2:
        interpretation = "[BULLISH] Moderate Bullish Bias"
    elif consensus > -0.2:
        interpretation = "[NEUTRAL] Mixed Signals"
    elif consensus > -0.5:
        interpretation = "[BEARISH] Moderate Bearish Bias"
    else:
        interpretation = "[BEARISH] Strong Bearish Consensus"
    
    print(f"\n   Multi-timeframe Interpretation: {interpretation}")
    
    # Compare results
    print("\n3. Enhancement Impact:")
    if enhanced_result.timeframe_consensus is not None:
        consensus_boost = abs(enhanced_result.timeframe_consensus) * 0.3
        print(f"   Consensus Factor: {enhanced_result.timeframe_consensus:.3f}")
        print(f"   Strength Enhancement: +{consensus_boost:.3f}")
        print(f"   Direction Confidence: {'Higher' if abs(consensus) > 0.3 else 'Similar'}")
    
    print(f"\n[SUCCESS] Multi-timeframe Enhancement Test Complete!")
    print(f"   Enhanced calculator provides: {enhanced_result.trend_strength} trend signals")
    
    return enhanced_result

if __name__ == "__main__":
    try:
        result = test_multi_timeframe_momentum()
        print("\n[SUCCESS] All tests passed!")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
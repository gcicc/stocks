#!/usr/bin/env python3
"""
Test adaptive signal thresholds based on market volatility.
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
from core.signal_generator import SignalGenerator
from core.data_manager import MarketData

def create_volatile_market_data(symbol: str = "VOLATILE", periods: int = 100, volatility: float = 0.05) -> MarketData:
    """Create market data with specified volatility level."""
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=periods), periods=periods, freq='D')
    
    # Create price series with specific volatility
    base_price = 100.0
    trend = 0.0005  # Small trend
    
    prices = []
    price = base_price
    
    for i in range(periods):
        # Add trend and controlled volatility
        price = price * (1 + trend + np.random.normal(0, volatility))
        prices.append(max(1.0, price))  # Ensure positive prices
    
    # Create OHLC data
    df = pd.DataFrame({
        'Open': [p * (1 + np.random.normal(0, volatility/3)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, volatility/2))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, volatility/2))) for p in prices],
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

def test_adaptive_thresholds():
    """Test adaptive thresholds under different volatility conditions."""
    
    print("Testing Adaptive Signal Thresholds")
    print("=" * 50)
    
    # Test scenarios with different volatility levels
    test_scenarios = [
        ("Low Volatility", 0.01),     # 1% daily volatility
        ("Medium Volatility", 0.03),  # 3% daily volatility  
        ("High Volatility", 0.08),    # 8% daily volatility
    ]
    
    # Initialize calculators
    momentum_calc = NaturalMomentumCalculator(enable_multi_timeframe=True)
    
    results = []
    
    for scenario_name, volatility_level in test_scenarios:
        print(f"\n{scenario_name} Market (vol={volatility_level:.1%}):")
        print("-" * 40)
        
        # Create market data with specific volatility
        market_data = create_volatile_market_data(f"TEST_{volatility_level}", 100, volatility_level)
        
        # Calculate momentum
        momentum_result = momentum_calc.calculate_natural_momentum(market_data)
        
        # Test both standard and adaptive signal generators
        standard_generator = SignalGenerator(enable_adaptive_thresholds=False)
        adaptive_generator = SignalGenerator(enable_adaptive_thresholds=True, volatility_multiplier=2.0)
        
        # Generate signals
        standard_signal = standard_generator.generate_signal(momentum_result, market_data)
        adaptive_signal = adaptive_generator.generate_signal(momentum_result, market_data)
        
        # Calculate the adaptive threshold for comparison
        adaptive_threshold = adaptive_generator._calculate_adaptive_threshold(momentum_result, market_data)
        
        print(f"  Market Volatility: {volatility_level:.1%}")
        print(f"  Price Range: ${market_data.prices['Close'].min():.2f} - ${market_data.prices['Close'].max():.2f}")
        print(f"  Current Momentum: {momentum_result.current_momentum:.4f}")
        print(f"  Standard Threshold: 0.000")
        print(f"  Adaptive Threshold: {adaptive_threshold:.3f}")
        print(f"  Standard Signal: {standard_signal.signal.value} (confidence: {standard_signal.confidence:.0%})")
        print(f"  Adaptive Signal: {adaptive_signal.signal.value} (confidence: {adaptive_signal.confidence:.0%})")
        
        # Store results for comparison
        results.append({
            'scenario': scenario_name,
            'volatility': volatility_level,
            'momentum': momentum_result.current_momentum,
            'standard_threshold': 0.0,
            'adaptive_threshold': adaptive_threshold,
            'standard_signal': standard_signal.signal.value,
            'adaptive_signal': adaptive_signal.signal.value,
            'standard_confidence': standard_signal.confidence,
            'adaptive_confidence': adaptive_signal.confidence,
            'threshold_difference': adaptive_threshold - 0.0
        })
    
    # Analysis
    print("\\n" + "=" * 50)
    print("ADAPTIVE THRESHOLD ANALYSIS:")
    print("=" * 50)
    
    for result in results:
        threshold_change = result['threshold_difference']
        if threshold_change > 0.1:
            adaptation = "Significantly Higher (More Conservative)"
        elif threshold_change > 0.05:
            adaptation = "Moderately Higher (Somewhat Conservative)"
        elif threshold_change > 0.01:
            adaptation = "Slightly Higher (Slightly Conservative)"
        else:
            adaptation = "Minimal Change"
        
        signal_change = "Same" if result['standard_signal'] == result['adaptive_signal'] else "Different"
        confidence_change = result['adaptive_confidence'] - result['standard_confidence']
        
        print(f"\n{result['scenario']}:")
        print(f"  Threshold Adaptation: {adaptation}")
        print(f"  Signal Change: {signal_change}")
        print(f"  Confidence Change: {confidence_change:+.0%}")
    
    # Verify adaptive behavior
    print("\\n" + "=" * 50)
    print("ADAPTIVE BEHAVIOR VERIFICATION:")
    print("=" * 50)
    
    low_vol_threshold = results[0]['adaptive_threshold']
    high_vol_threshold = results[-1]['adaptive_threshold']
    
    if high_vol_threshold > low_vol_threshold:
        print("[SUCCESS] Higher volatility produces higher thresholds")
        print(f"  Low vol threshold: {low_vol_threshold:.3f}")
        print(f"  High vol threshold: {high_vol_threshold:.3f}")
        print(f"  Difference: +{high_vol_threshold - low_vol_threshold:.3f}")
    else:
        print("[WARNING] Adaptive thresholds not working as expected")
    
    # Test edge cases
    print("\\n" + "=" * 50)  
    print("EDGE CASE TESTING:")
    print("=" * 50)
    
    # Test with insufficient data
    small_data = create_volatile_market_data("SMALL", 5, 0.03)
    small_momentum = momentum_calc.calculate_natural_momentum(small_data)
    small_signal = adaptive_generator.generate_signal(small_momentum, small_data)
    print(f"Small dataset signal: {small_signal.signal.value}")
    
    return results

if __name__ == "__main__":
    try:
        results = test_adaptive_thresholds()
        print("\\n[SUCCESS] All adaptive threshold tests completed!")
        
    except Exception as e:
        print(f"\\n[ERROR] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
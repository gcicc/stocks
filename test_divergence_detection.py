#!/usr/bin/env python3
"""
Test momentum divergence detection for trend reversal signals.
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
from core.signal_generator import SignalGenerator, SignalType
from core.data_manager import MarketData

def create_bullish_divergence_data(symbol: str = "BULL_DIV", periods: int = 60) -> MarketData:
    """Create market data showing bullish divergence pattern."""
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=periods), periods=periods, freq='D')
    
    # Create price pattern: declining trend with lower lows
    base_price = 120.0
    prices = []
    
    for i in range(periods):
        # Overall declining trend with lower lows pattern
        if i < periods * 0.4:  # First 40% - decline
            trend_factor = -0.005 * (i / periods * 4)  # Accelerating decline
        elif i < periods * 0.7:  # Next 30% - steeper decline to create lower low
            trend_factor = -0.008 * (i / periods * 3)
        else:  # Final 30% - slight recovery
            trend_factor = 0.002
        
        # Add some noise
        noise = np.random.normal(0, 0.01)
        
        if i == 0:
            prices.append(base_price)
        else:
            new_price = prices[-1] * (1 + trend_factor + noise)
            prices.append(max(50.0, new_price))  # Floor price
    
    # Create volume pattern (increasing volume during divergence)
    volumes = []
    for i in range(periods):
        base_vol = 3000000
        if i > periods * 0.5:  # Increasing volume in second half
            vol_factor = 1 + (i - periods * 0.5) / (periods * 0.5) * 0.5
        else:
            vol_factor = 1.0
        volumes.append(int(base_vol * vol_factor * (0.8 + 0.4 * np.random.random())))
    
    df = pd.DataFrame({
        'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': volumes
    }, index=dates)
    
    return MarketData(
        symbol=symbol,
        prices=df,
        current_price=prices[-1],
        previous_close=prices[-2] if len(prices) > 1 else prices[-1],
        volume=volumes[-1],
        last_updated=datetime.now(),
        data_source="synthetic_divergence_data"
    )

def create_bearish_divergence_data(symbol: str = "BEAR_DIV", periods: int = 60) -> MarketData:
    """Create market data showing bearish divergence pattern."""
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=periods), periods=periods, freq='D')
    
    # Create price pattern: rising trend with higher highs
    base_price = 80.0
    prices = []
    
    for i in range(periods):
        # Overall rising trend with higher highs pattern
        if i < periods * 0.4:  # First 40% - modest rise
            trend_factor = 0.004 * (i / periods * 3)
        elif i < periods * 0.7:  # Next 30% - steeper rise to create higher high
            trend_factor = 0.007 * (i / periods * 2)
        else:  # Final 30% - momentum weakening but price still rising
            trend_factor = 0.002
        
        # Add some noise
        noise = np.random.normal(0, 0.015)
        
        if i == 0:
            prices.append(base_price)
        else:
            new_price = prices[-1] * (1 + trend_factor + noise)
            prices.append(min(200.0, new_price))  # Ceiling price
    
    # Create volume pattern
    volumes = [np.random.randint(2000000, 6000000) for _ in range(periods)]
    
    df = pd.DataFrame({
        'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': volumes
    }, index=dates)
    
    return MarketData(
        symbol=symbol,
        prices=df,
        current_price=prices[-1],
        previous_close=prices[-2] if len(prices) > 1 else prices[-1],
        volume=volumes[-1],
        last_updated=datetime.now(),
        data_source="synthetic_divergence_data"
    )

def create_no_divergence_data(symbol: str = "NO_DIV", periods: int = 60) -> MarketData:
    """Create market data with no divergence (price and momentum aligned)."""
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=periods), periods=periods, freq='D')
    
    base_price = 100.0
    trend = 0.003  # Consistent uptrend
    volatility = 0.02
    
    prices = []
    price = base_price
    
    for i in range(periods):
        # Consistent trend with normal volatility
        price = price * (1 + trend + np.random.normal(0, volatility))
        prices.append(max(50.0, price))
    
    volumes = [np.random.randint(1000000, 4000000) for _ in range(periods)]
    
    df = pd.DataFrame({
        'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': volumes
    }, index=dates)
    
    return MarketData(
        symbol=symbol,
        prices=df,
        current_price=prices[-1],
        previous_close=prices[-2] if len(prices) > 1 else prices[-1],
        volume=volumes[-1],
        last_updated=datetime.now(),
        data_source="synthetic_no_divergence_data"
    )

def test_divergence_detection():
    """Test momentum divergence detection across different scenarios."""
    
    print("Testing Momentum Divergence Detection")
    print("=" * 50)
    
    # Initialize calculator and generators
    momentum_calc = NaturalMomentumCalculator(enable_multi_timeframe=True)
    
    # Test both with and without divergence detection
    standard_generator = SignalGenerator(
        enable_adaptive_thresholds=True,
        enable_signal_confirmation=True,
        enable_divergence_detection=False
    )
    
    divergence_generator = SignalGenerator(
        enable_adaptive_thresholds=True,
        enable_signal_confirmation=True,
        enable_divergence_detection=True,
        divergence_lookback=15
    )
    
    # Test scenarios
    test_scenarios = [
        ("Bullish Divergence", create_bullish_divergence_data("BULL_DIV", 60)),
        ("Bearish Divergence", create_bearish_divergence_data("BEAR_DIV", 60)),
        ("No Divergence", create_no_divergence_data("NO_DIV", 60)),
    ]
    
    results = []
    
    for scenario_name, market_data in test_scenarios:
        print(f"\n{scenario_name}:")
        print("-" * 35)
        
        # Calculate momentum
        momentum_result = momentum_calc.calculate_natural_momentum(market_data)
        
        # Generate signals with and without divergence detection
        standard_signal = standard_generator.generate_signal(momentum_result, market_data)
        divergence_signal = divergence_generator.generate_signal(momentum_result, market_data)
        
        # Test divergence detection directly
        detected_divergence = divergence_generator._detect_momentum_divergence(momentum_result, market_data)
        
        print(f"  Momentum: {momentum_result.current_momentum:.4f}")
        print(f"  Price Range: ${market_data.prices['Close'].min():.2f} - ${market_data.prices['Close'].max():.2f}")
        print(f"  Price Trend: {((market_data.current_price / market_data.prices['Close'].iloc[0]) - 1) * 100:+.1f}%")
        
        print(f"  \\n  Standard Signal: {standard_signal.signal.value} (confidence: {standard_signal.confidence:.0%})")
        print(f"  Divergence-Aware Signal: {divergence_signal.signal.value} (confidence: {divergence_signal.confidence:.0%})")
        
        # Show divergence detection results
        if detected_divergence:
            div_type = detected_divergence.get('type', 'unknown')
            div_strength = detected_divergence.get('strength', 'unknown')
            print(f"  Detected Divergence: {div_strength.upper()} {div_type.upper()}")
        else:
            print(f"  Detected Divergence: NONE")
        
        # Show signal impact
        if standard_signal.signal != divergence_signal.signal:
            print(f"  Divergence Impact: {standard_signal.signal.value} -> {divergence_signal.signal.value}")
        else:
            print(f"  Divergence Impact: NO CHANGE")
        
        # Show detailed reason
        if hasattr(divergence_signal, 'reason') and 'divergence' in divergence_signal.reason.lower():
            print(f"  Enhanced Reason: {divergence_signal.reason}")
        
        results.append({
            'scenario': scenario_name,
            'detected_divergence': detected_divergence is not None,
            'divergence_type': detected_divergence.get('type') if detected_divergence else None,
            'divergence_strength': detected_divergence.get('strength') if detected_divergence else None,
            'standard_signal': standard_signal.signal.value,
            'divergence_signal': divergence_signal.signal.value,
            'signal_changed': standard_signal.signal != divergence_signal.signal,
            'momentum': momentum_result.current_momentum,
            'price_change': ((market_data.current_price / market_data.prices['Close'].iloc[0]) - 1) * 100
        })
    
    # Analysis
    print("\\n" + "=" * 50)
    print("DIVERGENCE DETECTION ANALYSIS:")
    print("=" * 50)
    
    divergence_detected = sum(1 for r in results if r['detected_divergence'])
    signals_changed = sum(1 for r in results if r['signal_changed'])
    
    print(f"\\nDetection Performance:")
    print(f"  Scenarios tested: {len(results)}")
    print(f"  Divergences detected: {divergence_detected}")
    print(f"  Signals modified by divergence: {signals_changed}")
    
    print(f"\\nDetailed Results:")
    for result in results:
        status = "DETECTED" if result['detected_divergence'] else "NOT DETECTED"
        change_status = "MODIFIED" if result['signal_changed'] else "UNCHANGED"
        
        print(f"  {result['scenario']}: {status}")
        print(f"    Signal: {result['standard_signal']} -> {result['divergence_signal']} ({change_status})")
        
        if result['detected_divergence']:
            print(f"    Divergence: {result['divergence_strength']} {result['divergence_type']}")
        
        print(f"    Price vs Momentum: {result['price_change']:+.1f}% price, {result['momentum']:.3f} momentum\\n")
    
    # Test edge cases
    print("=" * 50)
    print("EDGE CASE TESTING:")
    print("=" * 50)
    
    # Test with minimal data
    small_data = create_no_divergence_data("SMALL", 10)
    small_momentum = momentum_calc.calculate_natural_momentum(small_data)
    small_divergence = divergence_generator._detect_momentum_divergence(small_momentum, small_data)
    
    print(f"\\nMinimal Data Test (10 periods):")
    print(f"  Divergence detected: {small_divergence is not None}")
    print(f"  Expected: False (insufficient data)")
    
    return results

if __name__ == "__main__":
    try:
        results = test_divergence_detection()
        print("\\n[SUCCESS] Divergence detection testing completed!")
        
    except Exception as e:
        print(f"\\n[ERROR] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
#!/usr/bin/env python3
"""
Test signal confirmation logic for reducing false positives.
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

def create_trending_market_data(symbol: str = "TREND", periods: int = 50, trend_strength: float = 0.002) -> MarketData:
    """Create market data with a clear trend for testing confirmation logic."""
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=periods), periods=periods, freq='D')
    
    # Create trending price series
    base_price = 100.0
    volatility = 0.015  # 1.5% daily volatility
    
    prices = []
    price = base_price
    
    for i in range(periods):
        # Add consistent trend with some noise
        trend_component = trend_strength * (1 + 0.1 * np.sin(i / 5))  # Slight variation in trend
        noise_component = np.random.normal(0, volatility)
        price = price * (1 + trend_component + noise_component)
        prices.append(max(1.0, price))
    
    # Create OHLC data with good volume
    volumes = [np.random.randint(2000000, 8000000) for _ in range(periods)]  # High volume
    
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
        data_source="synthetic_test_data"
    )

def create_choppy_market_data(symbol: str = "CHOPPY", periods: int = 50) -> MarketData:
    """Create choppy/sideways market data that should generate fewer confirmed signals."""
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=periods), periods=periods, freq='D')
    
    base_price = 100.0
    volatility = 0.025  # Higher volatility for choppiness
    
    prices = []
    price = base_price
    
    for i in range(periods):
        # Create choppy movement with no clear trend
        cycle_component = 0.01 * np.sin(i / 3) * np.cos(i / 7)  # Creates choppy pattern
        noise_component = np.random.normal(0, volatility)
        price = price * (1 + cycle_component + noise_component)
        prices.append(max(1.0, price))
    
    # Create OHLC data with variable volume
    volumes = [np.random.randint(500000, 3000000) for _ in range(periods)]  # Lower/variable volume
    
    df = pd.DataFrame({
        'Open': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.015))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.015))) for p in prices],
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
        data_source="synthetic_test_data"
    )

def test_signal_confirmation():
    """Test signal confirmation logic under different market conditions."""
    
    print("Testing Signal Confirmation Logic")
    print("=" * 50)
    
    # Initialize calculator with multi-timeframe analysis
    momentum_calc = NaturalMomentumCalculator(enable_multi_timeframe=True)
    
    # Test scenarios
    test_scenarios = [
        ("Strong Uptrend", create_trending_market_data("UPTREND", 50, 0.004)),
        ("Weak Uptrend", create_trending_market_data("WEAK_UP", 50, 0.001)),
        ("Choppy Market", create_choppy_market_data("CHOPPY", 50)),
        ("Strong Downtrend", create_trending_market_data("DOWNTREND", 50, -0.003)),
    ]
    
    results = []
    
    for scenario_name, market_data in test_scenarios:
        print(f"\n{scenario_name}:")
        print("-" * 30)
        
        # Calculate momentum
        momentum_result = momentum_calc.calculate_natural_momentum(market_data)
        
        # Test both standard and confirmation-enabled generators
        standard_generator = SignalGenerator(
            enable_adaptive_thresholds=True,
            enable_signal_confirmation=False
        )
        
        confirmation_generator = SignalGenerator(
            enable_adaptive_thresholds=True,
            enable_signal_confirmation=True,
            confirmation_periods=3,
            min_confirmations=2
        )
        
        # Generate signals
        standard_signal = standard_generator.generate_signal(momentum_result, market_data)
        confirmed_signal = confirmation_generator.generate_signal(momentum_result, market_data)
        
        print(f"  Momentum: {momentum_result.current_momentum:.4f}")
        print(f"  Timeframe Consensus: {getattr(momentum_result, 'timeframe_consensus', 'N/A')}")
        print(f"  Price Range: ${market_data.prices['Close'].min():.2f} - ${market_data.prices['Close'].max():.2f}")
        print(f"  Volume (current/avg): {market_data.volume:,} / {market_data.prices['Volume'].mean():.0f}")
        
        print(f"  \\n  Standard Signal: {standard_signal.signal.value} (confidence: {standard_signal.confidence:.0%})")
        print(f"  Confirmed Signal: {confirmed_signal.signal.value} (confidence: {confirmed_signal.confidence:.0%})")
        
        # Show confirmation impact
        if standard_signal.signal != confirmed_signal.signal:
            if confirmed_signal.signal == SignalType.HOLD:
                impact = "DOWNGRADED to HOLD"
            else:
                impact = f"CHANGED to {confirmed_signal.signal.value}"
            print(f"  Confirmation Impact: {impact}")
        else:
            print(f"  Confirmation Impact: CONFIRMED {confirmed_signal.signal.value}")
        
        # Show reason if available
        if hasattr(confirmed_signal, 'reason'):
            print(f"  Reason: {confirmed_signal.reason}")
        
        results.append({
            'scenario': scenario_name,
            'momentum': momentum_result.current_momentum,
            'consensus': getattr(momentum_result, 'timeframe_consensus', None),
            'standard_signal': standard_signal.signal.value,
            'confirmed_signal': confirmed_signal.signal.value,
            'standard_confidence': standard_signal.confidence,
            'confirmed_confidence': confirmed_signal.confidence,
            'was_downgraded': standard_signal.signal != confirmed_signal.signal and confirmed_signal.signal == SignalType.HOLD
        })
    
    # Analysis
    print("\\n" + "=" * 50)
    print("CONFIRMATION LOGIC ANALYSIS:")
    print("=" * 50)
    
    total_signals = len([r for r in results if r['standard_signal'] != 'HOLD'])
    downgraded_signals = len([r for r in results if r['was_downgraded']])
    confirmed_strong_signals = len([r for r in results if r['confirmed_signal'] in ['BUY', 'SELL']])
    
    print(f"\\nSignal Quality Metrics:")
    print(f"  Total signals generated (standard): {total_signals}")
    print(f"  Signals downgraded by confirmation: {downgraded_signals}")
    print(f"  Strong signals after confirmation: {confirmed_strong_signals}")
    print(f"  False positive reduction: {downgraded_signals/total_signals*100:.1f}%" if total_signals > 0 else "  False positive reduction: N/A")
    
    # Detailed confirmation analysis
    print(f"\\nConfirmation Results by Scenario:")
    for result in results:
        action = "DOWNGRADED" if result['was_downgraded'] else "CONFIRMED"
        confidence_change = result['confirmed_confidence'] - result['standard_confidence']
        print(f"  {result['scenario']}: {action} ({confidence_change:+.0%} confidence)")
    
    # Test confirmation factors individually
    print("\\n" + "=" * 50)
    print("CONFIRMATION FACTOR TESTING:")
    print("=" * 50)
    
    # Test with different confirmation settings
    strict_generator = SignalGenerator(
        enable_signal_confirmation=True,
        confirmation_periods=5,
        min_confirmations=4  # Very strict
    )
    
    lenient_generator = SignalGenerator(
        enable_signal_confirmation=True,
        confirmation_periods=2,
        min_confirmations=1  # Very lenient
    )
    
    print(f"\\nStrict vs Lenient Confirmation:")
    for scenario_name, market_data in test_scenarios[:2]:  # Test first two scenarios
        momentum_result = momentum_calc.calculate_natural_momentum(market_data)
        
        strict_signal = strict_generator.generate_signal(momentum_result, market_data)
        lenient_signal = lenient_generator.generate_signal(momentum_result, market_data)
        
        print(f"  {scenario_name}:")
        print(f"    Strict: {strict_signal.signal.value} ({strict_signal.confidence:.0%})")
        print(f"    Lenient: {lenient_signal.signal.value} ({lenient_signal.confidence:.0%})")
    
    return results

if __name__ == "__main__":
    try:
        results = test_signal_confirmation()
        print("\\n[SUCCESS] Signal confirmation testing completed!")
        
    except Exception as e:
        print(f"\\n[ERROR] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
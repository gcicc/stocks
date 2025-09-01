#!/usr/bin/env python3
"""
Simple test for divergence detection with clear patterns.
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

def test_peak_trough_detection():
    """Test the basic peak and trough detection algorithm."""
    
    print("Testing Peak and Trough Detection")
    print("=" * 40)
    
    # Create a simple pattern with clear peaks and troughs
    test_data = np.array([1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.0, 2.0, 3.5, 2.8])
    
    generator = SignalGenerator(enable_divergence_detection=True)
    peaks_troughs = generator._find_peaks_and_troughs(test_data)
    
    print(f"Test data: {test_data}")
    print(f"Detected peaks: {peaks_troughs['peaks']}")
    print(f"Detected troughs: {peaks_troughs['troughs']}")
    
    return peaks_troughs

def create_manual_divergence_data():
    """Create very clear divergence pattern manually."""
    
    # Create 30 data points with clear bullish divergence
    periods = 30
    dates = pd.date_range(start=datetime.now() - timedelta(days=periods), periods=periods, freq='D')
    
    # Price pattern: clear double bottom with second bottom lower
    prices = []
    for i in range(periods):
        if i < 8:  # First decline
            price = 100 - i * 2
        elif i < 12:  # First recovery  
            price = 84 + (i - 8) * 3
        elif i < 20:  # Second decline (lower low)
            price = 96 - (i - 12) * 3.5  # Goes lower than first bottom
        else:  # Second recovery
            price = 68 + (i - 20) * 2
        
        prices.append(price)
    
    print(f"Manual price pattern: {prices}")
    
    # Create OHLCV data
    volumes = [3000000] * periods
    
    df = pd.DataFrame({
        'Open': prices,
        'High': [p * 1.02 for p in prices],
        'Low': [p * 0.98 for p in prices], 
        'Close': prices,
        'Volume': volumes
    }, index=dates)
    
    return MarketData(
        symbol="MANUAL_DIV",
        prices=df,
        current_price=prices[-1],
        previous_close=prices[-2],
        volume=volumes[-1],
        last_updated=datetime.now(),
        data_source="manual_test_data"
    )

def test_manual_divergence():
    """Test divergence detection on manually crafted data."""
    
    print("\nTesting Manual Divergence Pattern")
    print("=" * 40)
    
    # Create manual divergence data
    market_data = create_manual_divergence_data()
    
    # Calculate momentum
    momentum_calc = NaturalMomentumCalculator(enable_multi_timeframe=False)
    momentum_result = momentum_calc.calculate_natural_momentum(market_data)
    
    print(f"Momentum values: {momentum_result.tema_values.values[-10:]}")  # Last 10 values
    print(f"Price values: {market_data.prices['Close'].values[-10:]}")     # Last 10 values
    
    # Test divergence detection
    generator = SignalGenerator(enable_divergence_detection=True, divergence_lookback=20)
    divergence = generator._detect_momentum_divergence(momentum_result, market_data)
    
    print(f"Detected divergence: {divergence}")
    
    # Test peak/trough detection on price and momentum separately
    print(f"\nDirect peak/trough testing:")
    price_peaks = generator._find_peaks_and_troughs(market_data.prices['Close'].values[-15:])
    momentum_peaks = generator._find_peaks_and_troughs(momentum_result.tema_values.values[-15:])
    
    print(f"Price peaks/troughs: {price_peaks}")
    print(f"Momentum peaks/troughs: {momentum_peaks}")
    
    # Test bullish divergence check
    bullish_div = generator._check_bullish_divergence(price_peaks, momentum_peaks)
    print(f"Bullish divergence check: {bullish_div}")
    
    return divergence

if __name__ == "__main__":
    try:
        # Test basic detection
        peaks_troughs = test_peak_trough_detection()
        
        # Test manual divergence
        divergence = test_manual_divergence()
        
        print(f"\n[SUCCESS] Simple divergence testing completed!")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
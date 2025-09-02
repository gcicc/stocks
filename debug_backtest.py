"""
Debug backtesting issue - check what's happening with signal generation.
"""

import sys
sys.path.insert(0, 'src')

from datetime import datetime, timedelta
from backtesting.data_provider import DataProvider, DataRequest
from core.momentum_calculator import NaturalMomentumCalculator
from core.signal_generator import SignalGenerator
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.DEBUG)

def debug_backtest():
    print("=== BACKTESTING DEBUG ===")
    
    # Test with simple symbols
    symbols = ['AAPL', 'MSFT']
    
    # Fetch historical data
    data_provider = DataProvider()
    request = DataRequest(
        symbols=symbols,
        start_date=datetime.now() - timedelta(days=200),
        end_date=datetime.now() - timedelta(days=1),
        data_source="yahoo"
    )
    
    print(f"Fetching data for {symbols}...")
    historical_data = data_provider.fetch_historical_data(request)
    
    print(f"Data fetched for {len(historical_data)} symbols")
    for symbol, data in historical_data.items():
        print(f"  {symbol}: {len(data)} rows, date range {data.index.min()} to {data.index.max()}")
    
    # Test momentum calculation
    momentum_calc = NaturalMomentumCalculator(enable_multi_timeframe=False)
    signal_gen = SignalGenerator(
        enable_adaptive_thresholds=False,
        enable_signal_confirmation=False,
        enable_divergence_detection=False,
        confidence_threshold=0.05,  # Very low threshold
        strength_threshold=0.01,    # Much lower strength threshold
        backtesting_mode=True       # Enable backtesting mode to relax filters
    )
    
    print("\n=== TESTING SINGLE DAY ANALYSIS ===")
    
    # Pick a recent date and test the analysis
    test_date = datetime.now() - timedelta(days=10)
    
    for symbol, data in historical_data.items():
        # Get data up to test date
        test_data = data[data.index <= test_date].copy()
        
        if len(test_data) < 50:
            print(f"  {symbol}: Insufficient data ({len(test_data)} rows)")
            continue
            
        print(f"\n  {symbol}: Testing with {len(test_data)} rows")
        
        # Create MarketData object
        from core.data_manager import MarketData
        market_data = MarketData(
            symbol=symbol,
            prices=test_data,
            current_price=float(test_data['Close'].iloc[-1]),
            previous_close=float(test_data['Close'].iloc[-2]),
            volume=float(test_data['Volume'].iloc[-1]) if 'Volume' in test_data else 0.0,
            data_source='backtest',
            last_updated=test_data.index[-1]
        )
        
        # Calculate momentum
        momentum_result = momentum_calc.calculate_natural_momentum(market_data)
        print(f"    Momentum: {momentum_result.current_momentum:.4f}")
        print(f"    Direction: {momentum_result.momentum_direction}")
        print(f"    Strength: {momentum_result.strength:.4f}")
        
        # Generate signal
        signal = signal_gen.generate_signal(momentum_result, market_data)
        print(f"    Signal: {signal.signal.value}")
        print(f"    Confidence: {signal.confidence:.2%}")
        print(f"    Reason: {signal.reason}")

if __name__ == "__main__":
    debug_backtest()
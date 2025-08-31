"""
Test basic workflow to ensure all components work together.
"""

import sys
import os
import asyncio

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from portfolio.csv_parser import parse_portfolio_csv
from core.data_manager import AsyncDataManager
from core.momentum_calculator import NaturalMomentumCalculator
from core.signal_generator import SignalGenerator

def test_csv_parsing():
    """Test CSV parsing with sample data."""
    print("TEST: Testing CSV parsing...")
    
    sample_csv = """Symbol,Quantity,Market Value,Cost Basis,Current Price,% of Portfolio
AAPL,100,15000.00,12000.00,150.00,25.0
MSFT,50,12500.00,10000.00,250.00,20.8
GOOGL,25,6250.00,5000.00,250.00,10.4"""
    
    try:
        portfolio = parse_portfolio_csv(sample_csv, "test.csv")
        print(f"SUCCESS: Parsed {len(portfolio.positions)} positions")
        print(f"SYMBOLS: {portfolio.symbols}")
        return portfolio
    except Exception as e:
        print(f"ERROR: CSV parsing failed: {str(e)}")
        return None

async def test_data_fetching(symbols):
    """Test data fetching for symbols."""
    print("TEST: Testing data fetching...")
    
    data_manager = AsyncDataManager()
    
    try:
        # Limit to 3 symbols for quick test
        test_symbols = symbols[:3]
        print(f"FETCHING data for: {test_symbols}")
        
        market_data = await data_manager.fetch_portfolio_data(test_symbols, period="6mo")
        
        print(f"SUCCESS: Fetched data for {len(market_data)} symbols")
        for symbol, data in market_data.items():
            print(f"   {symbol}: {len(data.prices)} days, current: ${data.current_price:.2f}")
        
        return market_data
    except Exception as e:
        print(f"ERROR: Data fetching failed: {str(e)}")
        return None

def test_momentum_calculation(market_data):
    """Test momentum calculations."""
    print("TEST: Testing momentum calculations...")
    
    calculator = NaturalMomentumCalculator()
    
    try:
        momentum_results = calculator.calculate_portfolio_momentum(market_data)
        
        print(f"SUCCESS: Calculated momentum for {len(momentum_results)} symbols")
        for symbol, result in momentum_results.items():
            print(f"   {symbol}: {result.momentum_direction}, strength: {result.strength:.2f}")
        
        return momentum_results
    except Exception as e:
        print(f"ERROR: Momentum calculation failed: {str(e)}")
        return None

def test_signal_generation(momentum_results, market_data):
    """Test signal generation."""
    print("TEST: Testing signal generation...")
    
    signal_generator = SignalGenerator()
    
    try:
        signals = signal_generator.generate_portfolio_signals(momentum_results, market_data)
        
        print(f"SUCCESS: Generated signals for {len(signals)} symbols")
        for symbol, signal in signals.items():
            print(f"   {symbol}: {signal.signal.value} (confidence: {signal.confidence:.0%})")
        
        return signals
    except Exception as e:
        print(f"ERROR: Signal generation failed: {str(e)}")
        return None

async def main():
    """Run complete workflow test."""
    print("STARTING: Testing Complete Portfolio Intelligence Workflow\n")
    
    # Step 1: Test CSV parsing
    portfolio = test_csv_parsing()
    if not portfolio:
        print("ERROR: Workflow failed at CSV parsing")
        return
    print()
    
    # Step 2: Test data fetching
    market_data = await test_data_fetching(portfolio.symbols)
    if not market_data:
        print("ERROR: Workflow failed at data fetching")
        return
    print()
    
    # Step 3: Test momentum calculation
    momentum_results = test_momentum_calculation(market_data)
    if not momentum_results:
        print("ERROR: Workflow failed at momentum calculation")
        return
    print()
    
    # Step 4: Test signal generation
    signals = test_signal_generation(momentum_results, market_data)
    if not signals:
        print("ERROR: Workflow failed at signal generation")
        return
    print()
    
    # Summary
    print("SUCCESS: COMPLETE WORKFLOW TEST SUCCESSFUL!")
    print(f"Portfolio: {len(portfolio.positions)} positions")
    print(f"Data: {len(market_data)} symbols fetched")
    print(f"Momentum: {len(momentum_results)} calculations completed")
    print(f"Signals: {len(signals)} trading recommendations generated")
    
    # Signal summary
    signal_counts = {}
    for signal in signals.values():
        signal_type = signal.signal.value
        signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
    
    print(f"Signal breakdown: {signal_counts}")
    print("\nREADY: Ready for production deployment!")

if __name__ == "__main__":
    asyncio.run(main())
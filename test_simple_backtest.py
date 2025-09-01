"""
Simple test to debug backtesting issues.
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_simple_backtest():
    print("SIMPLE BACKTESTING TEST")
    print("=" * 30)
    
    try:
        # Import components
        from backtesting.backtest_engine import BacktestEngine, BacktestSettings
        from core.momentum_calculator import NaturalMomentumCalculator
        from core.signal_generator import SignalGenerator
        print("[OK] Imports successful")
        
        # Create simple synthetic data with 100 days
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        dates = dates[dates.weekday < 5][:70]  # 70 trading days
        
        # Create single symbol data
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
        
        data = {
            'TEST': pd.DataFrame({
                'Open': prices * 0.99,
                'High': prices * 1.01,
                'Low': prices * 0.98,
                'Close': prices,
                'Adj Close': prices,
                'Volume': np.full(len(prices), 1000000)
            }, index=dates)
        }
        
        print(f"[OK] Created data: {len(dates)} days, price range ${prices.min():.2f}-${prices.max():.2f}")
        
        # Initialize components with conservative settings
        momentum_calc = NaturalMomentumCalculator(
            tema_period=5,  # Shorter period
            momentum_lookback=10,  # Shorter lookback
            enable_multi_timeframe=False  # Disable for simplicity
        )
        
        signal_gen = SignalGenerator(
            signal_threshold=0.0,
            confidence_threshold=0.1,  # Very low threshold
            enable_adaptive_thresholds=False,
            enable_signal_confirmation=False,
            enable_divergence_detection=False
        )
        
        engine = BacktestEngine(momentum_calc, signal_gen)
        print("[OK] Components initialized")
        
        # Simple backtest settings
        settings = BacktestSettings(
            start_date=dates[20],  # Start after some data for momentum calc
            end_date=dates[-1],
            initial_capital=10000,
            commission_rate=0.0,  # No costs for simplicity
            slippage_rate=0.0,
            max_position_size=1.0  # 100% allocation
        )
        
        print(f"[OK] Settings: {settings.start_date} to {settings.end_date}")
        
        # Check what trading dates we get
        trading_dates = engine._get_trading_dates(data, settings)
        print(f"[DEBUG] Trading dates found: {len(trading_dates)}")
        if trading_dates:
            print(f"[DEBUG] First date: {trading_dates[0]}, Last date: {trading_dates[-1]}")
        
        # Run backtest
        results = engine.run_backtest(data, settings)
        print(f"[OK] Backtest completed")
        
        # Check results
        if results.portfolio_history:
            print(f"[OK] Portfolio history: {len(results.portfolio_history)} snapshots")
            print(f"     Initial: ${results.initial_capital:,.2f}")
            print(f"     Final: ${results.final_capital:,.2f}")
            print(f"     Return: {results.total_return:.1%}")
        else:
            print("[WARNING] No portfolio history generated")
        
        if results.trades:
            print(f"[OK] Trades: {len(results.trades)} executed")
        else:
            print("[INFO] No trades executed")
            
        print("\nSUCCESS: Simple backtest working")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_simple_backtest()
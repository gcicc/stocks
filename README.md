# Portfolio Intelligence Platform

A production-ready portfolio analysis system that provides momentum-based trading insights with comprehensive backtesting, risk management, and documentation.

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   streamlit run src/dashboard/app.py --server.port 8505
   ```

3. **Access the Dashboard**
   - Open http://localhost:8505 in your browser
   - Upload your portfolio CSV file
   - Configure analysis parameters
   - Click "Analyze Portfolio" for trading signals
   - Switch to "Documentation" tab for detailed guides

## ✨ Production Features

### Core Analysis Engine
- **Natural Momentum Algorithm** - TEMA-based momentum with adaptive thresholds
- **AI-Enhanced Signal Generation** - BUY/SELL/HOLD with confidence scoring
- **Multi-Timeframe Analysis** - Short (7), medium (14), and long (28) period analysis
- **Signal Confirmation** - Cross-period validation to reduce false signals
- **Divergence Detection** - Identify momentum/price divergences

### Risk Management
- **Position Sizing** - Automatic risk-adjusted position calculations (max 20% per stock)
- **Stop Loss Generation** - Dynamic stop-loss levels based on volatility
- **Risk Level Assessment** - LOW/MEDIUM/HIGH risk classification per position
- **Portfolio Risk Metrics** - Sharpe ratio, max drawdown, volatility analysis

### Historical Backtesting
- **Complete Backtesting Engine** - Test strategies against historical data
- **Performance Analytics** - Comprehensive metrics vs. S&P 500 benchmark
- **Trade Analysis** - Win/loss ratios, profit factors, trade statistics
- **Risk-Adjusted Returns** - Sharpe ratio, Sortino ratio, risk metrics

### Professional Dashboard
- **Clean UI** - Professional interface optimized for analysis workflow
- **Real-Time Processing** - Ultra-fast parallel analysis of entire portfolios
- **Interactive Charts** - Plotly-powered visualizations and trend analysis
- **Comprehensive Documentation** - Built-in guides for all features
- **Export Capabilities** - CSV/Excel export of all analysis results

## 📊 Supported Formats

Compatible with portfolio exports from:
- Charles Schwab, Fidelity, TD Ameritrade, E*TRADE
- Vanguard, Interactive Brokers, Robinhood
- Any CSV with Symbol, Quantity, Price columns

## 🏗️ Architecture

```
src/
├── dashboard/
│   └── app.py                 # Main Streamlit application
├── core/
│   ├── data_manager.py        # Async multi-source data fetching
│   ├── momentum_calculator.py # TEMA & momentum calculations
│   └── signal_generator.py    # Trading signal generation
├── backtesting/
│   ├── backtest_engine.py     # Historical strategy testing
│   ├── data_provider.py       # Historical data management
│   └── performance_metrics.py # Performance analysis
├── portfolio/
│   └── csv_parser.py          # Robust portfolio parsing
├── risk/
│   └── risk_manager.py        # Risk assessment & position sizing
└── utils/
    └── config.py              # System configuration

docs/                          # Comprehensive documentation
├── USER_GUIDE.md             # Complete user guide
├── TECHNICAL_ANALYSIS_GUIDE.md # Technical indicator details
├── RISK_MANAGEMENT_GUIDE.md  # Risk principles & guidelines
└── API_DOCUMENTATION.md      # Developer reference

data/                         # Data storage
├── cache/                    # Market data cache
└── sample_portfolios/        # Example files
```

## 🎯 Key Performance Metrics

- **Analysis Speed**: Complete portfolio analysis in < 10 seconds
- **Data Sources**: Primary (Yahoo Finance) + fallback sources
- **Concurrent Processing**: Parallel analysis of all positions
- **Memory Efficient**: Optimized for large portfolios (100+ positions)
- **Cache System**: 5-minute TTL for real-time performance

## ⚙️ Configuration

Key settings in `src/utils/config.py`:
- **Signal Thresholds**: Momentum crossover levels
- **Confidence Levels**: Minimum confidence for actionable signals  
- **Risk Parameters**: Position sizing and stop-loss calculations
- **Backtesting Settings**: Historical analysis periods and benchmarks

## 📚 Documentation

The platform includes comprehensive built-in documentation:
- **User Guide**: Step-by-step usage instructions
- **Technical Analysis**: Detailed explanation of momentum algorithms
- **Risk Management**: Position sizing and risk control principles  
- **API Reference**: Developer documentation for all components

Access via the "📚 Documentation" tab in the dashboard.

## 🔒 Production Ready

- **Error Handling**: Graceful degradation for data source failures
- **Logging**: Comprehensive logging for debugging and monitoring
- **Data Validation**: Robust portfolio parsing with multiple fallbacks
- **Performance Optimized**: Numba acceleration for mathematical calculations
- **Clean Architecture**: SOLID principles with modular, testable components

## 🚀 Usage

1. Upload your portfolio CSV file via the sidebar
2. Adjust analysis parameters (confidence threshold, risk level, etc.)
3. Click "Analyze Portfolio" to generate momentum-based trading signals
4. Review live analysis results with charts and recommendations
5. Run historical backtesting to validate strategy performance
6. Export results and analysis reports

## 📈 Analysis Workflow

```
Portfolio Upload → Data Validation → Market Data Fetch → 
Momentum Calculation → Signal Generation → Risk Assessment → 
Backtesting Analysis → Results & Export
```

## 🔧 Development

Built with modern Python stack:
- **Streamlit** - Interactive web interface
- **Pandas/NumPy** - Data processing and analysis  
- **Plotly** - Interactive visualizations
- **yfinance** - Market data integration
- **Numba** - High-performance calculations
- **Asyncio** - Concurrent data processing

## 📄 License

Copyright (c) 2024 Portfolio Intelligence Platform - Personal Use
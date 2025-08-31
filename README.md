# Portfolio Intelligence Platform

Transform your portfolio CSV into actionable momentum-based trading insights in under 15 seconds.

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   streamlit run app.py
   ```

3. **Upload Your Portfolio**
   - Upload your portfolio CSV file from any major broker
   - Click "Analyze Portfolio" to generate momentum signals
   - Review results and export recommendations

## 📁 Supported Portfolio Formats

- Charles Schwab
- Fidelity
- TD Ameritrade
- E*TRADE
- Vanguard
- Interactive Brokers
- Generic CSV formats

## 🎯 Features

### Core Functionality (Milestone 1 - MVP)
✅ **Portfolio CSV Upload & Parsing** - Handles multiple broker formats with robust error handling  
✅ **Natural Momentum Indicator** - Advanced TEMA smoothing with natural log transformations  
✅ **Signal Generation** - Buy/Sell/Hold recommendations with confidence scores  
✅ **Interactive Dashboard** - Professional Streamlit interface with real-time analysis  
✅ **Risk Assessment** - Automatic risk level calculation for each position  
✅ **Export Results** - Download analysis as CSV for your records  

### Performance Targets
- ⚡ Portfolio upload to insights: < 15 seconds
- 📊 Dashboard load time: < 2 seconds  
- 🎯 Signal accuracy: > 55% profitable recommendations
- 📈 Support for 20+ position portfolios

## 📊 How It Works

### Natural Momentum Algorithm
1. **Natural Log Transformation** - Apply ln() to price data for better trend detection
2. **TEMA Smoothing** - Triple Exponential Moving Average reduces noise
3. **Zero-Line Crossover** - Generate signals when momentum crosses zero
4. **Confidence Scoring** - AI-enhanced filtering reduces false positives

### User Workflow
```
Upload CSV → Parse Portfolio → Fetch Market Data → Calculate Momentum → Generate Signals → Export Results
```

## 🧪 Testing

Use the sample portfolio file for testing:
```
data/sample_portfolios/sample_portfolio.csv
```

## 📁 Project Structure

```
stocks/
├── src/
│   ├── core/                    # Core algorithm components
│   │   ├── data_manager.py      # Async data fetching with fallbacks
│   │   ├── momentum_calculator.py # Natural Momentum with TEMA
│   │   └── signal_generator.py   # Buy/Sell/Hold logic
│   ├── portfolio/               # Portfolio handling
│   │   └── csv_parser.py        # Robust CSV parsing
│   ├── dashboard/               # Streamlit interface
│   │   └── app.py              # Main dashboard application
│   └── utils/                   # Configuration and utilities
│       └── config.py           # Settings management
├── data/
│   └── sample_portfolios/      # Test data
├── docs/                       # Documentation
├── app.py                      # Application entry point
└── requirements.txt            # Python dependencies
```

## ⚙️ Configuration

Key settings can be adjusted in `src/utils/config.py`:

- **TEMA Period**: Default 14 (momentum smoothing)
- **Signal Threshold**: Default 0.0 (zero-line crossover)
- **Confidence Threshold**: Default 0.6 (minimum confidence for signals)
- **Max Concurrent Requests**: Default 10 (data fetching performance)

## 🔮 Roadmap

### Next Milestones (Weeks 3-6)
- **Enhanced Algorithm** - Refined TEMA and signal confirmation
- **Professional Visualizations** - Advanced charts and performance metrics
- **ML Signal Filtering** - Reduce false positives with machine learning
- **Comprehensive Backtesting** - Historical validation with statistical significance

### Future Features (Weeks 7+)
- Real-time monitoring and alerts
- Multi-asset support (crypto, forex, commodities)
- API integration and mobile optimization
- Enterprise features and cloud deployment

## 📈 Technical Details

### Data Sources
- **Primary**: Yahoo Finance (yfinance)
- **Backup**: Alpha Vantage (requires API key)
- **Caching**: 5-minute TTL for market data
- **Fallback**: Graceful degradation with error handling

### Performance Optimizations
- Async/parallel data fetching for multiple symbols
- Numba acceleration for mathematical calculations (optional)
- Streamlit caching for dashboard performance
- Progressive loading for better user experience

## 🤝 Contributing

This is Milestone 1 (MVP) of the Portfolio Intelligence Platform. The architecture is designed for incremental enhancement based on user feedback and performance requirements.

## 📄 License

Copyright (c) 2024 Portfolio Intelligence Platform
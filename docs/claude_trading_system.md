# Claude.md - AI Portfolio Management System

## Project Overview
An automated portfolio management system where Claude analyzes micro-cap stocks (under $300M market cap) and makes autonomous trading decisions within defined parameters. The system combines real market data ingestion, deep research capabilities, and systematic portfolio management.

## System Architecture

### Data Pipeline
- **Data Source**: GitHub repository with daily market data
- **Data Types**: Prices, volumes, market cap, fundamental metrics, news sentiment
- **Update Frequency**: Daily market data, weekend deep analysis
- **Storage**: Local SQLite/PostgreSQL for historical data, pandas for analysis

### Core Components

#### 1. Data Ingestion Module (`data_ingester.py`)
```python
# Fetch daily market data from GitHub
# Update local database
# Validate data quality
# Generate data quality reports
```

#### 2. Claude Integration Module (`claude_interface.py`)
```python
# Interface with Claude API
# Pass market data and portfolio state
# Receive trading decisions and research analysis
# Log all interactions for audit trail
```

#### 3. Portfolio Manager (`portfolio_manager.py`)
```python
# Track current positions
# Calculate performance metrics
# Generate portfolio reports
# Validate trade constraints (micro-cap only, position limits)
```

#### 4. Research Engine (`research_engine.py`)
```python
# Weekend deep analysis workflows
# Stock screening and idea generation
# Risk assessment and position sizing
# Performance attribution analysis
```

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up GitHub data repository structure
- [ ] Create data ingestion pipeline
- [ ] Build basic portfolio tracking system
- [ ] Implement Claude API integration
- [ ] Create micro-cap stock universe filter

### Phase 2: Core Trading Logic (Weeks 3-4)
- [ ] Implement constraint validation (micro-cap only, position limits)
- [ ] Build trade execution simulation
- [ ] Create performance tracking and reporting
- [ ] Develop risk management framework
- [ ] Add logging and audit trails

### Phase 3: Research & Analysis (Weeks 5-6)
- [ ] Weekend research automation
- [ ] Stock screening algorithms
- [ ] Fundamental analysis integration
- [ ] Technical analysis tools
- [ ] Sentiment analysis pipeline

### Phase 4: Advanced Features (Weeks 7-8)
- [ ] Portfolio optimization algorithms
- [ ] Backtesting framework
- [ ] Performance comparison tools
- [ ] Risk metrics dashboard
- [ ] Automated reporting system

## Technical Specifications

### Data Requirements
- **Market Data**: OHLCV, market cap, float, institutional ownership
- **Fundamental Data**: Revenue, earnings, cash flow, debt levels
- **News/Sentiment**: Company-specific news, sector trends
- **Macro Data**: Market indices, sector performance, economic indicators

### Constraints & Rules
- **Universe**: US-listed stocks only, market cap < $300M
- **Position Type**: Full shares only (no fractional shares)
- **Time Horizon**: 6-month fixed periods
- **Decision Authority**: Claude has full buy/sell authority within constraints
- **Manual Execution**: Human manually places trades based on Claude's decisions

### Key Performance Metrics
- **Total Return**: Absolute and risk-adjusted performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough loss
- **Win Rate**: Percentage of profitable positions
- **Average Hold Time**: Position duration analysis

## Data Structure

### Daily Data Schema
```python
daily_data = {
    'date': 'YYYY-MM-DD',
    'symbol': 'TICKER',
    'open': float,
    'high': float,
    'low': float,
    'close': float,
    'volume': int,
    'market_cap': float,
    'shares_outstanding': int,
    'float': int
}
```

### Portfolio Schema
```python
portfolio_state = {
    'cash': float,
    'positions': {
        'TICKER': {
            'shares': int,
            'avg_cost': float,
            'current_price': float,
            'market_value': float,
            'unrealized_pnl': float,
            'entry_date': 'YYYY-MM-DD'
        }
    },
    'total_value': float,
    'daily_return': float,
    'cumulative_return': float
}
```

## Claude Prompt Engineering

### Daily Analysis Prompt
```
You are managing a micro-cap portfolio (under $300M market cap). 

Current Portfolio State: {portfolio_json}
Market Data: {market_data_json}
Date: {current_date}
Time Remaining: {days_until_end} days

Based on this data, should you:
1. Make any position changes?
2. Adjust stop-losses?
3. Take profits on any positions?

Respond with specific actions in JSON format.
```

### Weekend Research Prompt
```
Weekend Deep Research Session

Portfolio: {portfolio_json}
Recent Performance: {performance_metrics}
Market Environment: {market_summary}

Tasks:
1. Analyze current holdings for continued validity
2. Screen for new micro-cap opportunities
3. Assess sector rotation opportunities
4. Update risk management parameters

Provide detailed research with specific actionable recommendations.
```

## Risk Management Framework

### Position Sizing Rules
- Maximum single position: 15% of portfolio
- Minimum position size: 2% of portfolio
- Maximum sector concentration: 30%
- Cash reserves: 5-20% depending on market conditions

### Stop-Loss Strategy
- Initial stop: 15% below entry price
- Trailing stops: Adjust based on volatility
- Fundamental stops: Exit if thesis invalidated

## Technology Stack

### Core Technologies
- **Python 3.9+**: Main programming language
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **requests**: API calls and data fetching
- **SQLAlchemy**: Database ORM
- **Jupyter**: Research and development environment

### Optional Enhancements
- **Streamlit**: Web-based dashboard
- **yfinance**: Additional market data
- **scikit-learn**: ML models for screening
- **plotly**: Interactive visualizations
- **redis**: Caching layer

## Development Environment Setup

```bash
# Create virtual environment
python -m venv claude_trading
source claude_trading/bin/activate  # Linux/Mac
# or
claude_trading\Scripts\activate  # Windows

# Install dependencies
pip install pandas numpy requests sqlalchemy jupyter streamlit yfinance plotly

# Clone GitHub data repository
git clone https://github.com/your-username/trading-data.git

# Set up environment variables
export ANTHROPIC_API_KEY="your_key_here"
export GITHUB_TOKEN="your_github_token"
```

## Ethical and Legal Considerations

### Compliance
- **Disclosure**: This is a simulated trading system for educational purposes
- **Not Financial Advice**: Results should not be considered investment advice
- **Risk Warning**: Micro-cap stocks carry significant risks
- **Data Usage**: Ensure compliance with data provider terms of service

### Risk Disclaimers
- Past performance does not guarantee future results
- Micro-cap stocks are highly volatile and illiquid
- System may experience technical failures
- Human oversight required for all trade executions

## Success Metrics & Goals

### Primary Objectives
1. **Absolute Return**: Target 20%+ over 6-month period
2. **Risk Management**: Maximum drawdown < 25%
3. **Research Quality**: Document all investment theses
4. **System Reliability**: 99%+ uptime for daily analysis

### Lessons Learned from Their Experience

**What's Working Well**:
- The AI outperforming key benchmarks over four weeks validates the core concept
- Structured prompt engineering with specific order formats produces actionable results
- Weekly deep research sessions provide necessary context for decisions
- CSV-based tracking is simple but effective for transparency

**Critical Improvements Needed**:
- **Manual workflow bottleneck**: "ALL PROMPTING IS MANUAL AT THE MOMENT" limits scalability
- **Statistical validity concerns**: "Statistical reliability is extremely weak — trends may appear significant but are likely the result of randomness"
- **Technical debt**: "All of this is still VERY NEW, so there is bugs"
- **Limited data sources**: Only yFinance for micro-cap data may miss important information

**Risk Management Insights**:
- "Micro-cap, low-float, speculative, or illiquid securities, which can be subject to extreme volatility, bid-ask spread issues, or complete capital loss"
- Need for sophisticated liquidity and spread monitoring
- Importance of proper position sizing for volatile micro-caps

### Our Competitive Advantages

**1. Statistical Expertise Integration**
- Your PhD statistics background enables proper significance testing
- Advanced risk modeling beyond simple stop-losses
- Monte Carlo simulation for strategy validation
- Proper confidence intervals and performance attribution

**2. Professional Python Development**
- Clean, modular architecture vs their beginner-friendly but basic scripts
- Comprehensive error handling and logging
- Automated testing and validation frameworks
- Scalable database design vs CSV files

**3. Claude's Superior Reasoning**
- Claude Sonnet 4's advanced reasoning capabilities
- Better context handling for complex financial analysis
- More sophisticated research synthesis
- Superior risk assessment and portfolio construction

**4. Advanced Data Integration**
- Multiple data sources for better micro-cap coverage
- Real-time data quality validation
- Automated fundamental analysis from SEC filings
- Sentiment analysis integration from news and social media

**5. Institutional-Grade Risk Management**
- Volatility-adjusted position sizing
- Correlation and concentration monitoring
- Sector allocation constraints
- Advanced performance attribution analysis

## Next Steps

1. **Define Data Sources**: Identify reliable micro-cap data providers
2. **Set Up GitHub Repository**: Create standardized data format
3. **Build MVP**: Start with basic portfolio tracking and Claude integration
4. **Backtest Framework**: Test strategies on historical data
5. **Paper Trading**: Run system in simulation before real money

---

*This document serves as the foundational specification for the Claude.md trading system. Regular updates will be made as the system evolves and new requirements emerge.*
# Portfolio Intelligence Platform - API Documentation

## Architecture Overview

The Portfolio Intelligence Platform follows a modular architecture with clear separation of concerns:

```
src/
├── core/                     # Core analysis engines
│   ├── momentum_calculator.py    # TEMA & momentum analysis
│   ├── signal_generator.py       # Trading signal logic
│   └── portfolio_manager.py      # Position management
├── backtesting/             # Historical analysis
│   ├── backtest_engine.py        # Simulation engine
│   ├── data_provider.py          # Market data interface
│   └── performance_metrics.py    # Results analysis
├── risk/                    # Risk management
│   └── risk_manager.py           # Position sizing & controls
├── dashboard/              # User interface
│   └── app.py                    # Streamlit application
└── utils/                  # Utilities
    └── data_loader.py            # CSV data handling
```

## Core Modules

### MomentumCalculator (`src/core/momentum_calculator.py`)

**Purpose:** Calculates Triple Exponential Moving Average (TEMA) and momentum indicators.

#### Class: `NaturalMomentumCalculator`

```python
class NaturalMomentumCalculator:
    def __init__(self, 
                 tema_period: int = 14,
                 enable_multi_timeframe: bool = True,
                 short_period: int = 5,
                 medium_period: int = 14,
                 long_period: int = 30):
```

**Parameters:**
- `tema_period`: Period for TEMA calculation (default: 14)
- `enable_multi_timeframe`: Enable multi-timeframe analysis
- `short_period`: Short-term momentum period (default: 5)
- `medium_period`: Medium-term momentum period (default: 14)
- `long_period`: Long-term momentum period (default: 30)

#### Key Methods:

**`calculate_tema(data: pd.DataFrame) -> pd.Series`**
```python
def calculate_tema(self, data: pd.DataFrame) -> pd.Series:
    """
    Calculate Triple Exponential Moving Average.
    
    Args:
        data: DataFrame with 'Close' column
        
    Returns:
        Series with TEMA values
    """
```

**`calculate_momentum(data: pd.DataFrame) -> Dict[str, pd.Series]`**
```python
def calculate_momentum(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate momentum across multiple timeframes.
    
    Args:
        data: DataFrame with price data
        
    Returns:
        Dictionary with momentum series for each timeframe
    """
```

**Usage Example:**
```python
from core.momentum_calculator import NaturalMomentumCalculator

calc = NaturalMomentumCalculator(tema_period=21)
tema = calc.calculate_tema(price_data)
momentum = calc.calculate_momentum(price_data)
```

### SignalGenerator (`src/core/signal_generator.py`)

**Purpose:** Generates BUY/SELL/HOLD signals based on technical analysis.

#### Class: `SignalGenerator`

```python
class SignalGenerator:
    def __init__(self,
                 strength_threshold: float = 0.15,
                 confidence_threshold: float = 0.3,
                 enable_adaptive_thresholds: bool = True,
                 enable_signal_confirmation: bool = True,
                 enable_divergence_detection: bool = True,
                 backtesting_mode: bool = False):
```

**Parameters:**
- `strength_threshold`: Minimum momentum strength for signals
- `confidence_threshold`: Minimum confidence for signal generation
- `enable_adaptive_thresholds`: Dynamic threshold adjustment
- `enable_signal_confirmation`: Multi-factor signal validation
- `enable_divergence_detection`: Enable price/momentum divergence analysis
- `backtesting_mode`: Relaxed filtering for historical analysis

#### Key Methods:

**`generate_signals(momentum_data: Dict, price_data: pd.DataFrame, volume_data: pd.Series) -> Dict`**
```python
def generate_signals(self, momentum_data: Dict, 
                    price_data: pd.DataFrame,
                    volume_data: pd.Series = None) -> Dict:
    """
    Generate trading signals based on momentum and price analysis.
    
    Args:
        momentum_data: Multi-timeframe momentum data
        price_data: OHLCV price data
        volume_data: Trading volume data
        
    Returns:
        Dictionary containing signals, confidence scores, and metadata
    """
```

**Usage Example:**
```python
from core.signal_generator import SignalGenerator

signal_gen = SignalGenerator(strength_threshold=0.1)
signals = signal_gen.generate_signals(momentum_data, price_data, volume_data)
```

### DataProvider (`src/backtesting/data_provider.py`)

**Purpose:** Fetches and manages market data from various sources.

#### Class: `DataProvider`

```python
class DataProvider:
    def __init__(self, cache_enabled: bool = True, cache_dir: str = "data/cache"):
```

#### Class: `DataRequest`

```python
@dataclass
class DataRequest:
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    data_source: str = "yahoo"  # "yahoo", "alpha_vantage", "csv"
    interval: str = "1d"        # "1d", "1h", "5m"
```

**Key Methods:**

**`fetch_historical_data(request: DataRequest) -> Dict[str, pd.DataFrame]`**
```python
def fetch_historical_data(self, request: DataRequest) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical market data for multiple symbols.
    
    Args:
        request: DataRequest object with fetch parameters
        
    Returns:
        Dictionary mapping symbols to OHLCV DataFrames
    """
```

**Usage Example:**
```python
from backtesting.data_provider import DataProvider, DataRequest
from datetime import datetime, timedelta

provider = DataProvider()
request = DataRequest(
    symbols=['AAPL', 'MSFT'],
    start_date=datetime.now() - timedelta(days=365),
    end_date=datetime.now(),
    data_source="yahoo"
)
data = provider.fetch_historical_data(request)
```

### BacktestEngine (`src/backtesting/backtest_engine.py`)

**Purpose:** Simulates historical trading strategy performance.

#### Class: `BacktestEngine`

```python
class BacktestEngine:
    def __init__(self, 
                 momentum_calculator: NaturalMomentumCalculator,
                 signal_generator: SignalGenerator,
                 risk_manager: RiskManager = None):
```

#### Class: `BacktestSettings`

```python
@dataclass
class BacktestSettings:
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    max_position_size: float = 0.2
    enable_risk_management: bool = True
    risk_management_method: str = "kelly_criterion"  # "kelly_criterion", "risk_parity", "volatility_adjusted"
    max_portfolio_risk: float = 0.02
    stop_loss_method: str = "atr"  # "atr", "fixed", "dynamic"
    stop_loss_multiplier: float = 2.0
    correlation_threshold: float = 0.7
```

**Key Methods:**

**`run_backtest(historical_data: Dict, settings: BacktestSettings) -> BacktestResults`**
```python
def run_backtest(self, historical_data: Dict[str, pd.DataFrame], 
                settings: BacktestSettings) -> BacktestResults:
    """
    Execute complete backtest simulation.
    
    Args:
        historical_data: Symbol-to-DataFrame mapping
        settings: Backtest configuration parameters
        
    Returns:
        BacktestResults object with performance metrics
    """
```

**Usage Example:**
```python
from backtesting.backtest_engine import BacktestEngine, BacktestSettings
from datetime import datetime, timedelta

engine = BacktestEngine(momentum_calc, signal_gen, risk_manager)
settings = BacktestSettings(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    initial_capital=100000
)
results = engine.run_backtest(historical_data, settings)
```

### RiskManager (`src/risk/risk_manager.py`)

**Purpose:** Manages portfolio risk through position sizing and stop-loss controls.

#### Class: `RiskManager`

```python
class RiskManager:
    def __init__(self, risk_parameters: RiskParameters):
```

#### Class: `RiskParameters`

```python
@dataclass
class RiskParameters:
    max_position_size: float = 0.2        # Maximum position as % of portfolio
    max_portfolio_risk: float = 0.02      # Maximum risk per trade
    stop_loss_method: str = "atr"         # "atr", "fixed", "dynamic"
    stop_loss_multiplier: float = 2.0     # Multiplier for stop-loss calculation
    correlation_threshold: float = 0.7     # Maximum correlation between positions
    kelly_fraction: float = 0.25          # Fraction of full Kelly to use
    enable_correlation_check: bool = True
    enable_drawdown_control: bool = True
    max_drawdown_threshold: float = 0.1   # Stop trading at 10% drawdown
```

**Key Methods:**

**`calculate_position_sizes(signals: Dict, market_data: Dict, portfolio_value: float, method: str) -> Dict`**
```python
def calculate_position_sizes(self, signals: Dict, market_data: Dict, 
                           portfolio_value: float, 
                           method: str = "kelly_criterion") -> Dict:
    """
    Calculate optimal position sizes for given signals.
    
    Args:
        signals: Trading signals by symbol
        market_data: Historical price/volume data
        portfolio_value: Current portfolio value
        method: Position sizing method
        
    Returns:
        Dictionary with position sizes by symbol
    """
```

**`calculate_stop_loss(entry_price: float, market_data: pd.DataFrame, method: str) -> float`**
```python
def calculate_stop_loss(self, entry_price: float, 
                       market_data: pd.DataFrame, 
                       method: str = "atr") -> float:
    """
    Calculate stop-loss level for a position.
    
    Args:
        entry_price: Position entry price
        market_data: Historical price data for ATR calculation
        method: Stop-loss calculation method
        
    Returns:
        Stop-loss price level
    """
```

**Usage Example:**
```python
from risk.risk_manager import RiskManager, RiskParameters

risk_params = RiskParameters(
    max_position_size=0.15,
    max_portfolio_risk=0.02,
    kelly_fraction=0.25
)
risk_manager = RiskManager(risk_params)
position_sizes = risk_manager.calculate_position_sizes(signals, data, 100000)
```

## Data Formats

### Price Data Format

Standard OHLCV format used throughout the system:

```python
pd.DataFrame({
    'Open': [100.0, 101.0, 99.5],
    'High': [102.0, 103.0, 101.0],
    'Low': [99.0, 100.5, 98.0],
    'Close': [101.0, 99.5, 100.5],
    'Volume': [1000000, 1200000, 800000]
}, index=pd.DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03']))
```

### Signal Format

Trading signals returned by SignalGenerator:

```python
{
    'AAPL': {
        'signal': SignalType.BUY,           # BUY, SELL, HOLD
        'confidence': 0.75,                 # 0.0 to 1.0
        'strength': 0.12,                   # Momentum strength
        'reasons': ['momentum_bullish', 'volume_confirm'],
        'price_target': 155.0,              # Optional price target
        'stop_loss': 145.0,                 # Suggested stop-loss
        'timestamp': datetime(2023, 1, 1)
    }
}
```

### Backtest Results Format

```python
@dataclass
class BacktestResults:
    # Performance Metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Trading Statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Portfolio Metrics
    initial_capital: float
    final_capital: float
    
    # Benchmark Comparison
    benchmark_symbol: str = "SPY"
    benchmark_total_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    
    # Trade Details
    trades: List[Trade] = field(default_factory=list)
    daily_returns: pd.Series = None
    equity_curve: pd.Series = None
```

## Configuration Management

### Environment Variables

Create a `.env` file in the project root:

```bash
# Data Provider Settings
ALPHA_VANTAGE_API_KEY=your_api_key_here
YAHOO_FINANCE_ENABLED=true

# Risk Management
DEFAULT_MAX_POSITION_SIZE=0.20
DEFAULT_STOP_LOSS_MULTIPLIER=2.0

# Backtesting
DEFAULT_COMMISSION_RATE=0.001
DEFAULT_SLIPPAGE_RATE=0.0005

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/portfolio_intelligence.log
```

### Application Configuration

Configuration file at `config/settings.yaml`:

```yaml
momentum_calculator:
  tema_period: 14
  enable_multi_timeframe: true
  short_period: 5
  medium_period: 14
  long_period: 30

signal_generator:
  strength_threshold: 0.15
  confidence_threshold: 0.3
  enable_adaptive_thresholds: true
  enable_signal_confirmation: true
  enable_divergence_detection: true

risk_management:
  max_position_size: 0.20
  max_portfolio_risk: 0.02
  stop_loss_method: "atr"
  stop_loss_multiplier: 2.0
  correlation_threshold: 0.7
  kelly_fraction: 0.25

data_provider:
  cache_enabled: true
  cache_dir: "data/cache"
  default_source: "yahoo"
```

## Error Handling

### Custom Exceptions

```python
# src/utils/exceptions.py

class PortfolioIntelligenceException(Exception):
    """Base exception for Portfolio Intelligence Platform."""
    pass

class DataProviderException(PortfolioIntelligenceException):
    """Exception raised by data provider operations."""
    pass

class SignalGenerationException(PortfolioIntelligenceException):
    """Exception raised during signal generation."""
    pass

class BacktestException(PortfolioIntelligenceException):
    """Exception raised during backtesting operations."""
    pass

class RiskManagementException(PortfolioIntelligenceException):
    """Exception raised by risk management operations."""
    pass
```

### Error Handling Patterns

```python
try:
    signals = signal_generator.generate_signals(momentum_data, price_data)
except SignalGenerationException as e:
    logger.error(f"Signal generation failed: {e}")
    # Fallback to previous signals or default HOLD
    signals = create_default_hold_signals(symbols)
except Exception as e:
    logger.critical(f"Unexpected error: {e}")
    raise
```

## Testing Framework

### Unit Testing

```python
# tests/test_momentum_calculator.py
import unittest
from core.momentum_calculator import NaturalMomentumCalculator

class TestMomentumCalculator(unittest.TestCase):
    
    def setUp(self):
        self.calc = NaturalMomentumCalculator(tema_period=14)
        self.sample_data = create_sample_price_data()
    
    def test_tema_calculation(self):
        tema = self.calc.calculate_tema(self.sample_data)
        self.assertIsNotNone(tema)
        self.assertEqual(len(tema), len(self.sample_data))
        
    def test_momentum_calculation(self):
        momentum = self.calc.calculate_momentum(self.sample_data)
        self.assertIn('5d', momentum)
        self.assertIn('14d', momentum)
        self.assertIn('30d', momentum)
```

### Integration Testing

```python
# tests/test_integration.py
def test_end_to_end_backtest():
    """Test complete backtest workflow."""
    
    # Setup components
    momentum_calc = NaturalMomentumCalculator()
    signal_gen = SignalGenerator(backtesting_mode=True)
    risk_manager = RiskManager(RiskParameters())
    engine = BacktestEngine(momentum_calc, signal_gen, risk_manager)
    
    # Fetch data
    provider = DataProvider()
    data = provider.fetch_historical_data(test_request)
    
    # Run backtest
    settings = BacktestSettings(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31)
    )
    results = engine.run_backtest(data, settings)
    
    # Validate results
    assert results.total_trades > 0
    assert results.final_capital > 0
    assert -1.0 <= results.total_return <= 5.0  # Reasonable bounds
```

## Performance Optimization

### Caching Strategy

```python
from functools import lru_cache
import pickle
import hashlib

class CacheManager:
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    @lru_cache(maxsize=128)
    def cached_momentum_calculation(self, price_data_hash: str, params: tuple):
        """Cache momentum calculations."""
        cache_file = self.cache_dir / f"momentum_{price_data_hash}_{hash(params)}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Calculate if not cached
        result = self._calculate_momentum_internal(price_data, params)
        
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result
```

### Memory Management

```python
# Efficient DataFrame operations
def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce DataFrame memory usage."""
    
    for col in df.select_dtypes(include=['float64']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    for col in df.select_dtypes(include=['int64']):
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    return df

# Streaming data processing for large datasets
def process_large_dataset(file_path: str, chunk_size: int = 10000):
    """Process large datasets in chunks."""
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunk = optimize_dataframe_memory(chunk)
        yield process_chunk(chunk)
```

## Deployment Guide

### Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Start development server
streamlit run src/dashboard/app.py
```

### Production Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

EXPOSE 8501

CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Environment-Specific Configuration

```python
# src/utils/config.py
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Config:
    environment: str
    debug: bool
    log_level: str
    database_url: str
    cache_enabled: bool

def load_config() -> Config:
    """Load configuration based on environment."""
    
    env = os.getenv('ENVIRONMENT', 'development')
    
    configs = {
        'development': Config(
            environment='development',
            debug=True,
            log_level='DEBUG',
            database_url='sqlite:///dev.db',
            cache_enabled=True
        ),
        'production': Config(
            environment='production',
            debug=False,
            log_level='INFO',
            database_url=os.getenv('DATABASE_URL'),
            cache_enabled=True
        )
    }
    
    return configs.get(env, configs['development'])
```
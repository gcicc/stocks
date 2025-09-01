"""
Configuration management for the Natural Momentum Indicator platform.
"""

import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class DataSourceConfig:
    """Configuration for data sources with fallback hierarchy."""
    primary_source: str = "yfinance"
    backup_sources: list = None
    cache_ttl_seconds: int = 300  # 5 minutes
    request_timeout: int = 10
    max_retries: int = 3
    
    def __post_init__(self):
        if self.backup_sources is None:
            self.backup_sources = ["alpha_vantage"]


@dataclass
class PerformanceConfig:
    """Performance-related configuration."""
    max_concurrent_requests: int = 10
    chunk_size: int = 20  # Process portfolios in chunks
    cache_size_mb: int = 100
    enable_numba_acceleration: bool = True


@dataclass
class UIConfig:
    """User interface configuration."""
    page_title: str = "Portfolio Intelligence Platform"
    max_upload_size_mb: int = 10
    default_chart_height: int = 500
    progress_update_interval: float = 0.5


@dataclass
class AlgorithmConfig:
    """Natural Momentum algorithm parameters."""
    tema_period: int = 14
    momentum_lookback: int = 20
    signal_threshold: float = 0.0  # Zero line crossover
    confidence_threshold: float = 0.4  # Lowered from 0.6 to 0.4 (40%)


class Config:
    """Main configuration class combining all settings."""
    
    def __init__(self):
        self.data_sources = DataSourceConfig()
        self.performance = PerformanceConfig()
        self.ui = UIConfig()
        self.algorithm = AlgorithmConfig()
        
        # Environment-specific overrides
        self._load_environment_overrides()
    
    def _load_environment_overrides(self):
        """Load configuration from environment variables."""
        
        # Data source overrides
        if os.getenv("ALPHA_VANTAGE_API_KEY"):
            self.data_sources.backup_sources.insert(0, "alpha_vantage")
        
        # Performance overrides
        max_concurrent = os.getenv("MAX_CONCURRENT_REQUESTS")
        if max_concurrent:
            self.performance.max_concurrent_requests = int(max_concurrent)
        
        # Algorithm overrides
        tema_period = os.getenv("TEMA_PERIOD")
        if tema_period:
            self.algorithm.tema_period = int(tema_period)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for debugging."""
        return {
            "data_sources": self.data_sources.__dict__,
            "performance": self.performance.__dict__,
            "ui": self.ui.__dict__,
            "algorithm": self.algorithm.__dict__
        }


# Global configuration instance
config = Config()
#!/usr/bin/env python
"""
Main launcher for the Portfolio Intelligence Platform.
Handles proper Python path setup and launches the integrated dashboard.
"""

import sys
import os
from pathlib import Path

# Ensure we're using the correct Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"

# Clear any conflicting paths and add our src directory first
sys.path = [str(src_dir)] + [p for p in sys.path if 'qtp' not in p]

if __name__ == "__main__":
    # Import and run dashboard with proper path setup
    import streamlit.web.cli as stcli
    
    # Set up Streamlit command
    sys.argv = [
        "streamlit",
        "run", 
        str(src_dir / "dashboard" / "app.py"),
        "--server.port",
        "8504",
        "--server.headless", 
        "true"
    ]
    
    # Launch the integrated Portfolio Intelligence Platform
    print("Launching Portfolio Intelligence Platform...")
    print("Features: CSV Upload, Portfolio Analysis, Enhanced Momentum Algorithm, Historical Backtesting")
    print("Access at: http://localhost:8504")
    
    stcli.main()
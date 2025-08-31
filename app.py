"""
Main application entry point for Portfolio Intelligence Platform.
Run this file to start the Streamlit dashboard.
"""

import sys
import os

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Import and run the dashboard
from dashboard.app import main

if __name__ == "__main__":
    main()
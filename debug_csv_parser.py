"""
Debug script to help identify CSV parsing issues with real portfolio files.
"""

import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from portfolio.csv_parser import CSVParser
import pandas as pd
from io import StringIO


def debug_csv_structure(file_path):
    """Debug the structure of a CSV file to understand parsing issues."""
    
    print(f"DEBUGGING: {file_path}")
    print("=" * 60)
    
    try:
        # Try to read raw file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("FILE SIZE:", len(content), "characters")
        print("FIRST 500 CHARACTERS:")
        print(repr(content[:500]))
        print("\n" + "=" * 60)
        
        # Try different parsing strategies
        strategies = [
            {'sep': ',', 'encoding': 'utf-8'},
            {'sep': '\t', 'encoding': 'utf-8'},
            {'sep': ';', 'encoding': 'utf-8'},
            {'sep': ',', 'encoding': 'latin-1'},
            {'sep': ',', 'encoding': 'cp1252'},
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                print(f"\nSTRATEGY {i+1}: {strategy}")
                df = pd.read_csv(file_path, **strategy, low_memory=False)
                
                print(f"SUCCESS: Shape {df.shape}")
                print("COLUMNS:", list(df.columns))
                
                if not df.empty:
                    print("FIRST FEW ROWS:")
                    print(df.head(3).to_string())
                    
                    # Look for potential symbol columns
                    potential_symbol_cols = []
                    for col in df.columns:
                        if any(keyword in str(col).lower() for keyword in ['symbol', 'ticker', 'security', 'instrument']):
                            potential_symbol_cols.append(col)
                            print(f"POTENTIAL SYMBOL COLUMN: '{col}'")
                            if col in df.columns:
                                sample_values = df[col].dropna().head(5).tolist()
                                print(f"  Sample values: {sample_values}")
                
                break  # Success, stop trying other strategies
                
            except Exception as e:
                print(f"FAILED: {str(e)}")
                continue
        else:
            print("ALL STRATEGIES FAILED")
            
            # Try auto-detect
            try:
                print("\nTRYING AUTO-DETECT:")
                df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8')
                print(f"AUTO-DETECT SUCCESS: Shape {df.shape}")
                print("COLUMNS:", list(df.columns))
                if not df.empty:
                    print("FIRST ROW:")
                    print(df.head(1).to_string())
            except Exception as e:
                print(f"AUTO-DETECT FAILED: {str(e)}")
        
    except Exception as e:
        print(f"FILE READ ERROR: {str(e)}")


def suggest_fixes(file_path):
    """Suggest fixes based on common issues."""
    
    print("\n" + "=" * 60)
    print("COMMON ISSUES & FIXES:")
    print("=" * 60)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_lines = [f.readline().strip() for _ in range(10)]
        
        # Check for empty file
        if not any(first_lines):
            print("ISSUE: File appears to be empty")
            print("FIX: Check if file downloaded correctly")
            return
        
        # Check for metadata rows
        data_start_row = -1
        for i, line in enumerate(first_lines):
            if any(keyword in line.lower() for keyword in ['symbol', 'ticker', 'security', 'position']):
                data_start_row = i
                break
        
        if data_start_row > 0:
            print(f"ISSUE: Data appears to start at row {data_start_row + 1}")
            print("FIX: File has metadata rows before actual data")
            print("METADATA ROWS:")
            for i in range(data_start_row):
                print(f"  Row {i+1}: {first_lines[i]}")
        
        # Check for common broker patterns
        content = '\n'.join(first_lines).lower()
        
        if 'schwab' in content:
            print("DETECTED: Charles Schwab format")
        elif 'fidelity' in content:
            print("DETECTED: Fidelity format")
        elif 'ameritrade' in content:
            print("DETECTED: TD Ameritrade format")
        
    except Exception as e:
        print(f"ANALYSIS ERROR: {str(e)}")


def main():
    """Main debug function."""
    
    # Look for portfolio files in current directory
    portfolio_files = []
    for file in os.listdir('.'):
        if file.lower().endswith('.csv') and 'portfolio' in file.lower():
            portfolio_files.append(file)
    
    if not portfolio_files:
        print("No portfolio CSV files found in current directory")
        print("Looking for any CSV files...")
        csv_files = [f for f in os.listdir('.') if f.lower().endswith('.csv')]
        if csv_files:
            print("CSV files found:", csv_files)
            file_to_debug = csv_files[0]
        else:
            print("No CSV files found")
            return
    else:
        file_to_debug = portfolio_files[0]
        print(f"Found portfolio file: {file_to_debug}")
    
    debug_csv_structure(file_to_debug)
    suggest_fixes(file_to_debug)


if __name__ == "__main__":
    main()
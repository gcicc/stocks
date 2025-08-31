"""
Fix the portfolio parser for the specific format with CASH and TOTAL rows.
Based on user feedback about structure in portfolio data folder.
"""

import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)


def create_specialized_parser():
    """Create parser specifically for the user's portfolio format."""
    
    parser_code = '''"""
Specialized CSV parser for portfolio formats that include CASH and TOTAL rows.
Enhanced to handle specific broker formats with metadata and summary rows.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from io import StringIO
import logging

logger = logging.getLogger(__name__)


@dataclass
class PortfolioPosition:
    """Represents a single position in the portfolio."""
    symbol: str
    quantity: float
    market_value: Optional[float] = None
    cost_basis: Optional[float] = None
    current_price: Optional[float] = None
    unrealized_gain_loss: Optional[float] = None
    percent_of_portfolio: Optional[float] = None
    sector: Optional[str] = None
    
    def __post_init__(self):
        # Clean up symbol (remove extra spaces, convert to uppercase)
        self.symbol = self.symbol.strip().upper()
        
        # Calculate derived fields if possible
        if self.market_value and self.quantity and self.quantity != 0:
            self.current_price = self.market_value / abs(self.quantity)


@dataclass
class Portfolio:
    """Represents a complete portfolio with metadata."""
    positions: List[PortfolioPosition]
    total_market_value: Optional[float] = None
    account_name: Optional[str] = None
    as_of_date: Optional[str] = None
    
    @property
    def symbols(self) -> List[str]:
        """Get list of unique symbols in portfolio, excluding CASH and TOTAL."""
        excluded_terms = ['CASH', 'TOTAL', 'GRAND TOTAL', 'PORTFOLIO TOTAL', 'ACCOUNT TOTAL']
        return [pos.symbol for pos in self.positions 
                if pos.symbol and pos.symbol.upper() not in excluded_terms]
    
    @property
    def position_count(self) -> int:
        """Get number of actual stock positions (excluding cash/totals)."""
        excluded_terms = ['CASH', 'TOTAL', 'GRAND TOTAL', 'PORTFOLIO TOTAL', 'ACCOUNT TOTAL']
        return len([pos for pos in self.positions 
                   if pos.symbol.upper() not in excluded_terms])


class CSVParser:
    """
    Specialized parser for portfolio formats with CASH and TOTAL rows.
    """
    
    def __init__(self):
        self.symbol_patterns = [
            r'^[A-Z]{1,6}$',  # Standard stock symbols (allow up to 6 chars)
            r'^[A-Z]{1,5}\\.[A-Z]{1,2}$',  # International symbols with exchange
            r'^[A-Z]+\\d+[CP]\\d+$',  # Options patterns
        ]
        
        # Exclude terms that shouldn't be treated as stock symbols
        self.excluded_symbols = {
            'CASH', 'TOTAL', 'GRAND TOTAL', 'PORTFOLIO TOTAL', 
            'ACCOUNT TOTAL', 'NET WORTH', 'BALANCE', 'SUMMARY'
        }
        
        # Enhanced column mappings
        self.column_mappings = {
            'symbol': ['symbol', 'ticker', 'security', 'instrument', 'description', 'stock symbol', 
                      'security description', 'security name', 'company', 'name', 'position'],
            'quantity': ['quantity', 'qty', 'shares', 'units', 'position', 'share quantity',
                        'shares owned', 'total shares', 'share balance'],
            'market_value': ['market_value', 'market value', 'current_value', 'current value', 
                           'total_value', 'total value', 'market_val', 'value', 'market val',
                           'total market value', 'current market value', 'balance'],
            'cost_basis': ['cost_basis', 'cost basis', 'total_cost', 'total cost', 'basis',
                          'original cost', 'purchase price', 'avg cost'],
            'current_price': ['current_price', 'current price', 'price', 'last_price', 'last price',
                            'share price', 'price per share', 'unit price'],
            'unrealized_pnl': ['unrealized_gain_loss', 'unrealized gain/loss', 'unrealized_pnl',
                              'gain_loss', 'gain/loss', 'pnl', 'unrealized', 'unrealized g/l',
                              'total gain/loss', 'net gain/loss'],
            'percent_portfolio': ['percent_of_portfolio', '% of portfolio', 'allocation', 
                                'weight', 'portfolio_weight', 'portfolio %', '% of account']
        }
    
    def parse_csv(self, file_content: Union[str, bytes], filename: str = "portfolio.csv") -> Portfolio:
        """Parse CSV with special handling for CASH and TOTAL rows."""
        try:
            # Convert bytes to string if needed
            if isinstance(file_content, bytes):
                file_content = file_content.decode('utf-8')
            
            # Check for empty content
            if not file_content or file_content.strip() == '':
                raise ValueError("File content is empty")
            
            logger.info(f"Parsing CSV file: {filename} ({len(file_content)} characters)")
            
            # Debug: Log first few lines
            lines = file_content.split('\\n')[:10]
            logger.info(f"First few lines of CSV: {lines}")
            
            # Try multiple parsing strategies with detailed logging
            df = self._read_csv_with_detailed_logging(file_content)
            
            if df is None or df.empty:
                raise ValueError("Could not parse CSV file - no readable data found")
            
            logger.info(f"Successfully read CSV with shape: {df.shape}")
            logger.info(f"Columns found: {list(df.columns)}")
            
            # Show first few rows for debugging
            logger.info(f"First 3 rows:\\n{df.head(3).to_string()}")
            
            # Parse the dataframe
            portfolio = self._parse_dataframe(df, filename)
            
            if not portfolio.positions:
                raise ValueError("No valid positions found in CSV file")
            
            # Count actual stock positions (excluding CASH/TOTAL)
            stock_positions = [p for p in portfolio.positions if p.symbol.upper() not in self.excluded_symbols]
            
            logger.info(f"Successfully parsed portfolio with {len(stock_positions)} stock positions (excluded CASH/TOTAL)")
            return portfolio
            
        except Exception as e:
            logger.error(f"Error parsing CSV file {filename}: {str(e)}")
            # Provide more helpful error messages based on content analysis
            if "empty" in str(e).lower():
                raise ValueError("Portfolio file appears to be empty. Please check that the file downloaded correctly.")
            elif "encoding" in str(e).lower():
                raise ValueError("File encoding issue. Try saving the CSV file with UTF-8 encoding.")
            elif len(file_content) < 100:
                raise ValueError("File seems too small. Please check if it downloaded correctly.")
            else:
                raise ValueError(f"Could not parse portfolio CSV: {str(e)}. File has {len(file_content)} characters.")
    
    def _read_csv_with_detailed_logging(self, content: str) -> Optional[pd.DataFrame]:
        """CSV reading with detailed logging for debugging."""
        
        # Log content analysis
        logger.info(f"Content analysis: {len(content)} chars, {content.count(chr(10))} lines")
        
        # Try to detect separator
        comma_count = content.count(',')
        tab_count = content.count('\\t')
        semicolon_count = content.count(';')
        
        logger.info(f"Separator analysis - Commas: {comma_count}, Tabs: {tab_count}, Semicolons: {semicolon_count}")
        
        # Strategy 1: Standard CSV parsing with multiple separators
        separators_to_try = [',', '\\t', ';', '|']
        
        for sep in separators_to_try:
            logger.info(f"Trying separator: '{sep}'")
            
            strategies = [
                {'sep': sep, 'encoding': 'utf-8'},
                {'sep': sep, 'encoding': 'utf-8', 'skiprows': 1},
                {'sep': sep, 'encoding': 'utf-8', 'skiprows': 2},
                {'sep': sep, 'encoding': 'latin-1'},
                {'sep': sep, 'encoding': 'cp1252'},
            ]
            
            for i, strategy in enumerate(strategies):
                try:
                    logger.info(f"  Strategy {i+1}: {strategy}")
                    df = pd.read_csv(StringIO(content), **strategy, low_memory=False)
                    
                    logger.info(f"  Result: Shape {df.shape}, Columns: {list(df.columns)}")
                    
                    if not df.empty and df.shape[1] > 1:
                        # Additional validation
                        if self._validate_dataframe_detailed(df):
                            logger.info(f"  SUCCESS with strategy: {strategy}")
                            return df
                        else:
                            logger.info(f"  Failed validation")
                    else:
                        logger.info(f"  Empty or single column result")
                        
                except Exception as e:
                    logger.info(f"  Failed: {str(e)}")
                    continue
        
        # Strategy 2: Try reading line by line for manual parsing
        logger.info("Attempting manual line-by-line parsing...")
        try:
            lines = content.split('\\n')
            logger.info(f"Found {len(lines)} lines")
            
            if len(lines) > 1:
                # Try to identify the structure
                for i, line in enumerate(lines[:10]):
                    logger.info(f"Line {i}: '{line}'")
                
                # Look for a line that looks like headers
                header_line = None
                data_start = None
                
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if line contains common portfolio headers
                    lower_line = line.lower()
                    if any(keyword in lower_line for keyword in ['symbol', 'ticker', 'quantity', 'shares', 'value']):
                        header_line = i
                        data_start = i
                        logger.info(f"Found potential header at line {i}: '{line}'")
                        break
                
                if header_line is not None:
                    # Try to parse from this point
                    csv_content = '\\n'.join(lines[data_start:])
                    logger.info(f"Attempting to parse from line {data_start}")
                    
                    df = pd.read_csv(StringIO(csv_content), sep=',', encoding='utf-8')
                    if not df.empty:
                        logger.info(f"Manual parsing successful: {df.shape}")
                        return df
            
        except Exception as e:
            logger.error(f"Manual parsing failed: {str(e)}")
        
        logger.error("All CSV reading strategies failed")
        return None
    
    def _validate_dataframe_detailed(self, df: pd.DataFrame) -> bool:
        """Detailed validation with logging."""
        
        if df.empty or df.shape[1] < 2:
            logger.info(f"Validation failed: Empty or too few columns ({df.shape})")
            return False
        
        # Look for symbol-like data
        symbol_columns = []
        for col in df.columns:
            col_name = str(col).lower()
            if any(keyword in col_name for keyword in ['symbol', 'ticker', 'security', 'name', 'description']):
                symbol_columns.append(col)
                logger.info(f"Found potential symbol column: '{col}'")
                
                # Sample some values
                sample_values = df[col].dropna().astype(str).head(5).tolist()
                logger.info(f"  Sample values: {sample_values}")
        
        # Look for numeric columns
        numeric_columns = []
        for col in df.columns:
            col_name = str(col).lower()
            if any(keyword in col_name for keyword in ['quantity', 'shares', 'value', 'price', 'amount']):
                numeric_columns.append(col)
                logger.info(f"Found potential numeric column: '{col}'")
        
        has_symbols = len(symbol_columns) > 0
        has_numbers = len(numeric_columns) > 0
        
        logger.info(f"Validation result: symbols={has_symbols}, numbers={has_numbers}")
        return has_symbols and has_numbers
    
    def _parse_dataframe(self, df: pd.DataFrame, filename: str) -> Portfolio:
        """Parse dataframe with special handling for CASH and TOTAL."""
        
        # Clean the dataframe
        df = self._clean_dataframe(df)
        
        # Map columns to standard names
        column_mapping = self._map_columns_enhanced(df.columns)
        df = df.rename(columns=column_mapping)
        
        logger.info(f"Column mapping applied: {column_mapping}")
        
        # Extract all positions (including CASH and TOTAL for portfolio value calculation)
        positions = self._extract_positions_with_exclusions(df)
        
        # Extract portfolio metadata
        metadata = self._extract_metadata_with_totals(df, filename)
        
        return Portfolio(
            positions=positions,
            total_market_value=metadata.get('total_value'),
            account_name=metadata.get('account_name'),
            as_of_date=metadata.get('as_of_date')
        )
    
    def _extract_positions_with_exclusions(self, df: pd.DataFrame) -> List[PortfolioPosition]:
        """Extract positions, keeping CASH and TOTAL for calculations but marking them."""
        positions = []
        
        logger.info(f"Extracting positions from {len(df)} rows")
        
        for idx, row in df.iterrows():
            try:
                # Get symbol from first column or mapped symbol column
                symbol_value = None
                
                # Try mapped symbol column first
                if 'symbol' in df.columns:
                    symbol_value = row.get('symbol')
                
                # If no symbol column mapped, try first column
                if pd.isna(symbol_value) or str(symbol_value).strip() == '':
                    symbol_value = row.iloc[0] if len(row) > 0 else None
                
                if pd.isna(symbol_value) or str(symbol_value).strip() == '':
                    logger.debug(f"Row {idx}: No symbol found")
                    continue
                
                # Clean the symbol
                symbol = self._clean_symbol(str(symbol_value))
                if not symbol:
                    logger.debug(f"Row {idx}: Symbol cleaning failed for '{symbol_value}'")
                    continue
                
                # Get quantity - be flexible about which column contains it
                quantity = self._safe_float(row.get('quantity', 0))
                if quantity == 0:
                    # Try other columns that might contain quantity
                    for col in df.columns:
                        if any(keyword in str(col).lower() for keyword in ['shar', 'unit', 'qty']):
                            alt_qty = self._safe_float(row.get(col))
                            if alt_qty and alt_qty != 0:
                                quantity = alt_qty
                                break
                
                # For CASH and TOTAL rows, quantity might be 0 - that's OK
                # We'll include them but they won't appear in the symbols list
                
                position = PortfolioPosition(
                    symbol=symbol,
                    quantity=quantity if quantity else 0,
                    market_value=self._safe_float(row.get('market_value')),
                    cost_basis=self._safe_float(row.get('cost_basis')),
                    current_price=self._safe_float(row.get('current_price')),
                    unrealized_gain_loss=self._safe_float(row.get('unrealized_pnl')),
                    percent_of_portfolio=self._safe_float(row.get('percent_portfolio'))
                )
                
                positions.append(position)
                
                # Log what we found
                if symbol.upper() in self.excluded_symbols:
                    logger.info(f"Added {symbol} row (excluded from trading): value=${position.market_value}")
                else:
                    logger.info(f"Added stock position: {symbol} ({quantity} shares, value=${position.market_value})")
                
            except Exception as e:
                logger.warning(f"Skipping row {idx} due to error: {str(e)}")
                continue
        
        stock_count = len([p for p in positions if p.symbol.upper() not in self.excluded_symbols])
        total_count = len(positions)
        
        logger.info(f"Extracted {stock_count} stock positions + {total_count - stock_count} cash/total rows")
        return positions
    
    def _extract_metadata_with_totals(self, df: pd.DataFrame, filename: str) -> Dict[str, any]:
        """Extract metadata, including total portfolio value from TOTAL rows."""
        metadata = {}
        
        # Look for TOTAL row to get portfolio value
        for idx, row in df.iterrows():
            first_col_value = str(row.iloc[0]).upper() if len(row) > 0 else ""
            
            if any(term in first_col_value for term in ['TOTAL', 'GRAND TOTAL', 'PORTFOLIO TOTAL']):
                # Look for a numeric value that represents total portfolio value
                for col in df.columns:
                    if 'symbol' in str(col).lower():
                        continue  # Skip symbol column
                    
                    val = self._safe_float(row.get(col))
                    if val and val > 1000:  # Likely a portfolio total
                        metadata['total_value'] = val
                        logger.info(f"Found portfolio total: ${val:,.2f}")
                        break
        
        # Account name from filename
        account_name = filename.replace('.csv', '').replace('_', ' ')
        metadata['account_name'] = account_name
        
        return metadata
    
    def _clean_symbol(self, symbol: str) -> str:
        """Clean symbol, handling descriptions and special cases."""
        if not symbol or symbol.lower() in ['nan', 'none', '', 'null']:
            return ''
        
        symbol = str(symbol).strip().upper()
        
        # Handle special cases first
        if any(term in symbol for term in ['CASH', 'TOTAL']):
            return symbol  # Keep as-is for CASH and TOTAL
        
        # For regular symbols, clean up
        # Remove parenthetical info: "AAPL (Apple Inc)" -> "AAPL"
        symbol = re.sub(r'\\([^)]*\\)', '', symbol).strip()
        
        # Remove descriptions: "AAPL - Apple Inc" -> "AAPL"  
        if ' - ' in symbol:
            symbol = symbol.split(' - ')[0].strip()
        
        # Remove non-alphanumeric except dots
        symbol = re.sub(r'[^A-Z0-9.]', '', symbol)
        
        return symbol
    
    def _map_columns_enhanced(self, columns: List[str]) -> Dict[str, str]:
        """Enhanced column mapping."""
        mapping = {}
        
        for col in columns:
            col_lower = str(col).lower().strip()
            
            # Find best match
            for standard_col, variations in self.column_mappings.items():
                for variation in variations:
                    if variation.lower() in col_lower or col_lower in variation.lower():
                        mapping[col] = standard_col
                        break
                if col in mapping:
                    break
        
        return mapping
    
    def _safe_float(self, value) -> Optional[float]:
        """Safe float conversion with enhanced cleaning."""
        if pd.isna(value) or value == '' or value is None:
            return None
        
        try:
            if isinstance(value, str):
                # Remove currency symbols, commas, spaces, etc.
                cleaned = re.sub(r'[$,()%\\s€£¥]', '', value.strip())
                
                # Handle negatives in parentheses
                if '(' in str(value) and ')' in str(value):
                    cleaned = '-' + cleaned.replace('(', '').replace(')', '')
                
                if cleaned == '' or cleaned == '-':
                    return None
                    
                return float(cleaned)
            
            return float(value)
            
        except (ValueError, TypeError):
            return None
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataframe while preserving CASH and TOTAL rows."""
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Strip whitespace from string columns  
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            df[col] = df[col].astype(str).str.strip()
        
        # DON'T remove TOTAL rows - we need them for portfolio value calculation
        
        return df


def parse_portfolio_csv(file_content: Union[str, bytes], filename: str = "portfolio.csv") -> Portfolio:
    """
    Specialized function to parse portfolio CSV with CASH and TOTAL handling.
    """
    parser = CSVParser()
    return parser.parse_csv(file_content, filename)
'''
    
    # Write the specialized parser
    parser_file = os.path.join(src_dir, 'portfolio', 'csv_parser.py')
    with open(parser_file, 'w', encoding='utf-8') as f:
        f.write(parser_code)
    
    print("SUCCESS: Specialized parser deployed!")
    print("FEATURES: Handles CASH and TOTAL rows correctly")
    print("EXCLUSIONS: CASH and TOTAL won't appear in symbols list")
    print("LOGGING: Detailed debugging information")
    print("RESTART: Please restart your Streamlit app")


if __name__ == "__main__":
    create_specialized_parser()
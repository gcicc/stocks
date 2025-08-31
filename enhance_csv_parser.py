"""
Enhanced CSV parser to handle more edge cases and broker formats.
Run this to update the parser with better error handling.
"""

import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)


def enhance_csv_parser():
    """Add enhanced parsing logic to handle more edge cases."""
    
    enhanced_parser_code = '''"""
Robust CSV parser for various portfolio formats from different brokers.
Enhanced version with better error handling and edge case support.
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
        """Get list of unique symbols in portfolio."""
        return [pos.symbol for pos in self.positions if pos.symbol]
    
    @property
    def position_count(self) -> int:
        """Get number of positions."""
        return len(self.positions)


class CSVParser:
    """
    Enhanced CSV parser that handles multiple portfolio formats with better error handling.
    """
    
    def __init__(self):
        self.symbol_patterns = [
            r'^[A-Z]{1,5}$',  # Standard stock symbols
            r'^[A-Z]{1,5}\\.[A-Z]{1,2}$',  # International symbols with exchange
            r'^[A-Z]+\\d+[CP]\\d+$',  # Options patterns
        ]
        
        # Enhanced column mappings with more variations
        self.column_mappings = {
            'symbol': ['symbol', 'ticker', 'security', 'instrument', 'description', 'stock symbol', 
                      'security description', 'security name', 'company', 'name'],
            'quantity': ['quantity', 'qty', 'shares', 'units', 'position', 'share quantity',
                        'shares owned', 'total shares', 'share balance'],
            'market_value': ['market_value', 'market value', 'current_value', 'current value', 
                           'total_value', 'total value', 'market_val', 'value', 'market val',
                           'total market value', 'current market value'],
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
        """Enhanced CSV parsing with better error handling."""
        try:
            # Convert bytes to string if needed
            if isinstance(file_content, bytes):
                file_content = file_content.decode('utf-8')
            
            # Check for empty content
            if not file_content or file_content.strip() == '':
                raise ValueError("File content is empty")
            
            logger.info(f"Parsing CSV file: {filename} ({len(file_content)} characters)")
            
            # Handle different CSV formats with enhanced strategies
            df = self._read_csv_flexible(file_content)
            
            if df is None or df.empty:
                raise ValueError("Could not parse CSV file - no readable data found")
            
            logger.info(f"Successfully read CSV with shape: {df.shape}")
            
            # Detect and parse format
            portfolio = self._parse_dataframe(df, filename)
            
            if not portfolio.positions:
                raise ValueError("No valid positions found in CSV file")
            
            logger.info(f"Successfully parsed portfolio with {len(portfolio.positions)} positions")
            return portfolio
            
        except Exception as e:
            logger.error(f"Error parsing CSV file {filename}: {str(e)}")
            # Provide more helpful error messages
            if "empty" in str(e).lower():
                raise ValueError("Portfolio file appears to be empty. Please check that the file downloaded correctly.")
            elif "encoding" in str(e).lower():
                raise ValueError("File encoding issue. Try saving the CSV file with UTF-8 encoding.")
            else:
                raise ValueError(f"Could not parse portfolio CSV: {str(e)}")
    
    def _read_csv_flexible(self, content: str) -> Optional[pd.DataFrame]:
        """Enhanced CSV reading with more strategies and better error handling."""
        
        # Strategy 1: Try standard approaches
        basic_strategies = [
            {'sep': ',', 'encoding': 'utf-8'},
            {'sep': '\t', 'encoding': 'utf-8'},
            {'sep': ';', 'encoding': 'utf-8'},
            {'sep': ',', 'encoding': 'latin-1'},
            {'sep': ',', 'encoding': 'cp1252'},
            {'sep': ',', 'encoding': 'iso-8859-1'},
        ]
        
        for strategy in basic_strategies:
            try:
                df = pd.read_csv(StringIO(content), **strategy, low_memory=False)
                
                # Check if we got meaningful data
                if not df.empty and df.shape[1] > 1:
                    # Further validation - check if we have reasonable data
                    if self._validate_dataframe(df):
                        logger.info(f"Successfully read CSV with strategy: {strategy}")
                        return df
                        
            except Exception as e:
                logger.debug(f"Strategy {strategy} failed: {str(e)}")
                continue
        
        # Strategy 2: Try with different parameters
        enhanced_strategies = [
            {'sep': ',', 'skiprows': 1, 'encoding': 'utf-8'},  # Skip potential header row
            {'sep': ',', 'skiprows': 2, 'encoding': 'utf-8'},  # Skip two header rows
            {'sep': ',', 'header': None, 'encoding': 'utf-8'},  # No header
            {'sep': ',', 'quotechar': '"', 'encoding': 'utf-8'},
            {'sep': ',', 'quotechar': "'", 'encoding': 'utf-8'},
        ]
        
        for strategy in enhanced_strategies:
            try:
                df = pd.read_csv(StringIO(content), **strategy, low_memory=False)
                if not df.empty and df.shape[1] > 1 and self._validate_dataframe(df):
                    logger.info(f"Enhanced strategy succeeded: {strategy}")
                    return df
            except Exception:
                continue
        
        # Strategy 3: Auto-detection as last resort
        try:
            df = pd.read_csv(StringIO(content), sep=None, engine='python', low_memory=False)
            if not df.empty and self._validate_dataframe(df):
                logger.info("Auto-detection strategy succeeded")
                return df
        except Exception:
            pass
        
        logger.error("All CSV reading strategies failed")
        return None
    
    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validate that dataframe contains reasonable portfolio data."""
        
        # Basic checks
        if df.empty or df.shape[1] < 2:
            return False
        
        # Check for potential symbol column
        has_symbols = False
        for col in df.columns:
            col_name = str(col).lower()
            if any(keyword in col_name for keyword in ['symbol', 'ticker', 'security', 'name']):
                # Check if this column has reasonable values
                sample_values = df[col].dropna().astype(str).str.strip().head(10)
                if any(self._looks_like_symbol(val) for val in sample_values):
                    has_symbols = True
                    break
        
        # Check for quantity/value columns
        has_numbers = False
        for col in df.columns:
            col_name = str(col).lower()
            if any(keyword in col_name for keyword in ['quantity', 'shares', 'value', 'price']):
                # Check if column has numeric data
                try:
                    numeric_values = pd.to_numeric(df[col].dropna(), errors='coerce').dropna()
                    if len(numeric_values) > 0:
                        has_numbers = True
                        break
                except:
                    continue
        
        return has_symbols and has_numbers
    
    def _looks_like_symbol(self, value: str) -> bool:
        """Check if a value looks like a stock symbol."""
        if not value or len(value) > 10:
            return False
        
        # Clean the value
        cleaned = re.sub(r'[^A-Z0-9.]', '', str(value).upper())
        
        # Check against patterns
        return any(re.match(pattern, cleaned) for pattern in self.symbol_patterns)
    
    def _parse_dataframe(self, df: pd.DataFrame, filename: str) -> Portfolio:
        """Enhanced dataframe parsing with better error handling."""
        
        # Clean up the dataframe
        df = self._clean_dataframe(df)
        
        # Find the data section (skip headers/metadata) - enhanced logic
        data_start_row = self._find_data_start_enhanced(df)
        if data_start_row > 0:
            logger.info(f"Skipping {data_start_row} metadata rows")
            df = df.iloc[data_start_row:].reset_index(drop=True)
        
        # Map columns to standard names - enhanced logic
        column_mapping = self._map_columns_enhanced(df.columns)
        df = df.rename(columns=column_mapping)
        
        logger.info(f"Column mapping: {column_mapping}")
        
        # Extract positions with better error handling
        positions = self._extract_positions_enhanced(df)
        
        # Extract portfolio metadata
        metadata = self._extract_metadata(df, filename)
        
        return Portfolio(
            positions=positions,
            total_market_value=metadata.get('total_value'),
            account_name=metadata.get('account_name'),
            as_of_date=metadata.get('as_of_date')
        )
    
    def _find_data_start_enhanced(self, df: pd.DataFrame) -> int:
        """Enhanced logic to find where actual position data starts."""
        
        for idx, row in df.iterrows():
            # Convert row to string for analysis
            row_str = ' '.join(str(val).lower() for val in row if pd.notna(val))
            
            # Look for rows that contain column headers
            header_keywords = ['symbol', 'ticker', 'quantity', 'shares', 'position', 'security', 'value']
            keyword_count = sum(1 for keyword in header_keywords if keyword in row_str)
            
            # If we find multiple keywords, this is likely the header row
            if keyword_count >= 2:
                return idx
            
            # Also check if row has reasonable data structure
            non_null_count = row.count()
            if non_null_count >= 3:  # At least 3 non-null values
                # Check if first few columns look like symbol data
                first_val = str(row.iloc[0]).strip().upper()
                if len(first_val) <= 10 and re.match(r'^[A-Z0-9.]+$', first_val):
                    return idx
        
        return 0
    
    def _map_columns_enhanced(self, columns: List[str]) -> Dict[str, str]:
        """Enhanced column mapping with fuzzy matching."""
        mapping = {}
        
        for col in columns:
            col_lower = str(col).lower().strip()
            best_match = None
            best_score = 0
            
            # Find the best match for each standard column
            for standard_col, variations in self.column_mappings.items():
                for variation in variations:
                    # Exact match
                    if variation.lower() == col_lower:
                        mapping[col] = standard_col
                        best_match = standard_col
                        break
                    
                    # Partial match
                    if variation.lower() in col_lower or col_lower in variation.lower():
                        score = len(variation)  # Prefer longer matches
                        if score > best_score:
                            best_match = standard_col
                            best_score = score
                
                if best_match:
                    mapping[col] = best_match
                    break
        
        return mapping
    
    def _extract_positions_enhanced(self, df: pd.DataFrame) -> List[PortfolioPosition]:
        """Enhanced position extraction with better error handling."""
        positions = []
        
        logger.info(f"Extracting positions from dataframe with columns: {list(df.columns)}")
        
        for idx, row in df.iterrows():
            try:
                # Skip rows without meaningful data
                if pd.isna(row.get('symbol')) or str(row.get('symbol')).strip() == '':
                    continue
                
                symbol = self._clean_symbol(str(row.get('symbol', '')))
                if not symbol:
                    logger.debug(f"Skipping row {idx}: no valid symbol")
                    continue
                
                # Be more lenient with symbol validation for enhanced parsing
                if not (self._is_valid_symbol(symbol) or self._looks_like_symbol(symbol)):
                    logger.debug(f"Skipping row {idx}: symbol '{symbol}' doesn't look valid")
                    continue
                
                # Extract numeric values safely
                quantity = self._safe_float(row.get('quantity', 0))
                if quantity == 0:
                    # Try to find quantity in other columns if main quantity is 0
                    for col in df.columns:
                        if 'shar' in str(col).lower() or 'unit' in str(col).lower():
                            alt_qty = self._safe_float(row.get(col))
                            if alt_qty and alt_qty != 0:
                                quantity = alt_qty
                                break
                
                if quantity == 0:
                    logger.debug(f"Skipping row {idx}: no valid quantity for {symbol}")
                    continue
                
                position = PortfolioPosition(
                    symbol=symbol,
                    quantity=quantity,
                    market_value=self._safe_float(row.get('market_value')),
                    cost_basis=self._safe_float(row.get('cost_basis')),
                    current_price=self._safe_float(row.get('current_price')),
                    unrealized_gain_loss=self._safe_float(row.get('unrealized_pnl')),
                    percent_of_portfolio=self._safe_float(row.get('percent_portfolio'))
                )
                
                positions.append(position)
                logger.debug(f"Added position: {symbol} ({quantity} shares)")
                
            except Exception as e:
                logger.warning(f"Skipping row {idx} due to parsing error: {str(e)}")
                continue
        
        logger.info(f"Successfully extracted {len(positions)} positions")
        return positions
    
    def _clean_symbol(self, symbol: str) -> str:
        """Enhanced symbol cleaning."""
        if not symbol or symbol.lower() in ['nan', 'none', '', 'null']:
            return ''
        
        # Remove common prefixes/suffixes that aren't part of the symbol
        symbol = str(symbol).strip().upper()
        
        # Remove parenthetical information and other noise
        symbol = re.sub(r'\\([^)]*\\)', '', symbol).strip()
        symbol = re.sub(r'\\s*-\\s*.*$', '', symbol).strip()  # Remove "AAPL - Apple Inc"
        symbol = re.sub(r'[^A-Z0-9.]', '', symbol)  # Keep only alphanumeric and dots
        
        # Extract just the symbol part if it's mixed with description
        match = re.match(r'^([A-Z]{1,5})', symbol)
        if match:
            symbol = match.group(1)
        
        return symbol
    
    def _is_valid_symbol(self, symbol: str) -> bool:
        """Enhanced symbol validation."""
        if not symbol or len(symbol) > 6:  # Allow slightly longer symbols
            return False
        
        return any(re.match(pattern, symbol) for pattern in self.symbol_patterns)
    
    def _safe_float(self, value) -> Optional[float]:
        """Enhanced safe float conversion."""
        if pd.isna(value) or value == '' or value is None:
            return None
        
        try:
            # Handle string values with various formats
            if isinstance(value, str):
                # Remove various currency symbols, commas, spaces, etc.
                cleaned = re.sub(r'[$,()%\\s€£¥]', '', value.strip())
                
                # Handle negative values in parentheses
                if '(' in str(value) and ')' in str(value):
                    cleaned = '-' + cleaned.replace('(', '').replace(')', '')
                
                # Handle percentages
                if '%' in str(value):
                    cleaned = cleaned.replace('%', '')
                    if cleaned and cleaned != '-':
                        return float(cleaned) / 100.0
                
                if cleaned == '' or cleaned == '-':
                    return None
                    
                return float(cleaned)
            
            return float(value)
            
        except (ValueError, TypeError):
            return None
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced dataframe cleaning."""
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Strip whitespace from string columns
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            df[col] = df[col].astype(str).str.strip()
        
        # Remove rows that are clearly not data (like "Total" rows)
        df = df[~df.astype(str).apply(lambda x: x.str.contains('Total|TOTAL|Grand Total', na=False)).any(axis=1)]
        
        return df
    
    def _extract_metadata(self, df: pd.DataFrame, filename: str) -> Dict[str, any]:
        """Enhanced metadata extraction."""
        metadata = {}
        
        # Try to find total value with more patterns
        total_candidates = ['Total', 'TOTAL', 'Grand Total', 'Portfolio Total', 'Account Total', 'Net Worth']
        for _, row in df.iterrows():
            for col in df.columns:
                cell_value = str(row.get(col, ''))
                if any(candidate in cell_value for candidate in total_candidates):
                    # Look for numeric value in the same row or nearby cells
                    for other_col in df.columns:
                        val = self._safe_float(row.get(other_col))
                        if val and val > 1000:  # Likely a portfolio total
                            metadata['total_value'] = val
                            break
        
        # Extract account name from filename with better parsing
        account_name = filename.replace('.csv', '').replace('_', ' ').replace('(', '').replace(')', '')
        metadata['account_name'] = account_name
        
        return metadata


def parse_portfolio_csv(file_content: Union[str, bytes], filename: str = "portfolio.csv") -> Portfolio:
    """
    Enhanced convenience function to parse a portfolio CSV file.
    """
    parser = CSVParser()
    return parser.parse_csv(file_content, filename)
'''
    
    # Write enhanced parser
    parser_file = os.path.join(src_dir, 'portfolio', 'csv_parser.py')
    with open(parser_file, 'w', encoding='utf-8') as f:
        f.write(enhanced_parser_code)
    
    print("✅ Enhanced CSV parser has been deployed!")
    print("🚀 Restart your Streamlit app to use the improved parser")
    print("📊 The new parser handles:")
    print("   - More broker formats")
    print("   - Better error messages")
    print("   - Metadata row detection")
    print("   - Enhanced symbol cleaning")
    print("   - Fuzzy column matching")


if __name__ == "__main__":
    enhance_csv_parser()
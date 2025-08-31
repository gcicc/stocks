"""
Multi-section CSV parser for portfolio formats with account summary and positions sections.
Handles broker formats with metadata sections followed by position data.
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
    Multi-section CSV parser for complex broker formats.
    Handles files with account summaries followed by position data.
    """
    
    def __init__(self):
        # Exclude terms that shouldn't be treated as stock symbols
        self.excluded_symbols = {
            'CASH', 'TOTAL', 'GRAND TOTAL', 'PORTFOLIO TOTAL', 
            'ACCOUNT TOTAL', 'NET WORTH', 'BALANCE', 'SUMMARY'
        }
    
    def parse_csv(self, file_content: Union[str, bytes], filename: str = "portfolio.csv") -> Portfolio:
        """Parse multi-section CSV with account summary and positions."""
        try:
            # Convert bytes to string if needed
            if isinstance(file_content, bytes):
                file_content = file_content.decode('utf-8')
            
            # Check for empty content
            if not file_content or file_content.strip() == '':
                raise ValueError("File content is empty")
            
            logger.info(f"Parsing multi-section CSV: {filename} ({len(file_content)} characters)")
            
            # Split into lines and analyze structure
            lines = file_content.split('\n')
            logger.info(f"File has {len(lines)} lines")
            
            # Find the positions section
            positions_start = self._find_positions_section(lines)
            if positions_start == -1:
                raise ValueError("Could not find positions section in CSV")
            
            logger.info(f"Found positions section starting at line {positions_start}")
            
            # Extract positions data
            positions_lines = lines[positions_start:]
            positions_csv = '\n'.join(positions_lines)
            
            logger.info(f"Positions section has {len(positions_lines)} lines")
            logger.info(f"First position line: {positions_lines[0] if positions_lines else 'None'}")
            
            # Parse positions section as CSV
            df = self._parse_positions_csv(positions_csv)
            
            if df is None or df.empty:
                raise ValueError("Could not parse positions data")
            
            logger.info(f"Successfully parsed positions: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Extract positions
            positions = self._extract_positions_from_df(df)
            
            # Extract metadata from the beginning of the file
            metadata = self._extract_metadata_from_lines(lines, filename)
            
            # Count stock positions (excluding CASH/TOTAL)
            stock_positions = [p for p in positions if p.symbol.upper() not in self.excluded_symbols]
            
            logger.info(f"Successfully parsed {len(stock_positions)} stock positions (excluded CASH/TOTAL)")
            
            return Portfolio(
                positions=positions,
                total_market_value=metadata.get('total_value'),
                account_name=metadata.get('account_name'),
                as_of_date=metadata.get('as_of_date')
            )
            
        except Exception as e:
            logger.error(f"Error parsing multi-section CSV {filename}: {str(e)}")
            raise ValueError(f"Could not parse portfolio CSV: {str(e)}")
    
    def _find_positions_section(self, lines: List[str]) -> int:
        """Find where the positions data starts."""
        
        # Look for a line that looks like position headers
        position_keywords = ['symbol', 'last price', 'quantity', 'value', 'shares']
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Check if this line contains multiple position-related keywords
            keyword_count = sum(1 for keyword in position_keywords if keyword in line_lower)
            
            if keyword_count >= 2:  # At least 2 position keywords
                logger.info(f"Found position header at line {i}: '{line}'")
                return i
        
        # If no clear header found, look for lines with stock symbols
        for i, line in enumerate(lines):
            if ',' in line:
                parts = line.split(',')
                if len(parts) > 3:  # At least 4 columns
                    first_part = parts[0].strip().upper()
                    # Check if first part looks like a stock symbol
                    if len(first_part) >= 2 and len(first_part) <= 6 and first_part.isalpha():
                        logger.info(f"Found potential position data at line {i}: '{line}'")
                        # Look back for header
                        if i > 0:
                            prev_line = lines[i-1].lower()
                            if any(keyword in prev_line for keyword in position_keywords):
                                return i - 1
                        return i
        
        return -1
    
    def _parse_positions_csv(self, positions_csv: str) -> Optional[pd.DataFrame]:
        """Parse the positions section as CSV."""
        
        try:
            # Try standard CSV parsing first
            df = pd.read_csv(StringIO(positions_csv), encoding='utf-8')
            
            if not df.empty and df.shape[1] > 3:
                logger.info(f"Standard CSV parsing successful: {df.shape}")
                return df
                
        except Exception as e:
            logger.info(f"Standard parsing failed: {str(e)}")
        
        # Try with error handling for inconsistent columns
        try:
            df = pd.read_csv(StringIO(positions_csv), encoding='utf-8', error_bad_lines=False, warn_bad_lines=False)
            
            if not df.empty:
                logger.info(f"Error-tolerant parsing successful: {df.shape}")
                return df
                
        except Exception as e:
            logger.info(f"Error-tolerant parsing failed: {str(e)}")
        
        # Manual line-by-line parsing as fallback
        try:
            lines = positions_csv.strip().split('\n')
            if len(lines) < 2:
                return None
            
            # Get headers
            header_line = lines[0]
            headers = [col.strip() for col in header_line.split(',')]
            
            logger.info(f"Manual parsing headers: {headers}")
            
            # Parse data rows
            data_rows = []
            for i, line in enumerate(lines[1:], 1):
                if not line.strip():
                    continue
                
                # Split and pad/trim to match header count
                row_data = [col.strip() for col in line.split(',')]
                
                # Handle inconsistent column counts
                if len(row_data) > len(headers):
                    # Too many columns - trim
                    row_data = row_data[:len(headers)]
                elif len(row_data) < len(headers):
                    # Too few columns - pad with empty strings
                    row_data.extend([''] * (len(headers) - len(row_data)))
                
                data_rows.append(row_data)
                logger.debug(f"Row {i}: {row_data}")
            
            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=headers)
            logger.info(f"Manual parsing successful: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Manual parsing failed: {str(e)}")
            return None
    
    def _extract_positions_from_df(self, df: pd.DataFrame) -> List[PortfolioPosition]:
        """Extract positions from the parsed DataFrame."""
        positions = []
        
        # Map common column names
        column_map = self._create_column_mapping(df.columns)
        logger.info(f"Column mapping: {column_map}")
        
        for idx, row in df.iterrows():
            try:
                # Get symbol (first column or mapped symbol column)
                symbol = None
                if 'symbol' in column_map:
                    symbol = str(row[column_map['symbol']]).strip()
                elif len(df.columns) > 0:
                    symbol = str(row.iloc[0]).strip()
                
                if not symbol or symbol.lower() in ['nan', '', 'none']:
                    continue
                
                symbol = symbol.upper()
                
                # Get quantity
                quantity = 0
                if 'quantity' in column_map:
                    quantity = self._safe_float(row[column_map['quantity']])
                
                # Get market value  
                market_value = None
                if 'value' in column_map:
                    market_value = self._safe_float(row[column_map['value']])
                
                # Get current price
                current_price = None
                if 'price' in column_map:
                    current_price = self._safe_float(row[column_map['price']])
                
                # Get gain/loss
                total_gain = None
                if 'total_gain' in column_map:
                    total_gain = self._safe_float(row[column_map['total_gain']])
                
                # Create position
                position = PortfolioPosition(
                    symbol=symbol,
                    quantity=quantity if quantity else 0,
                    market_value=market_value,
                    current_price=current_price,
                    unrealized_gain_loss=total_gain
                )
                
                positions.append(position)
                
                # Log what we found
                if symbol.upper() in self.excluded_symbols:
                    logger.info(f"Added {symbol} (excluded from trading): value=${market_value}")
                else:
                    logger.info(f"Added {symbol}: {quantity} shares, value=${market_value}")
                
            except Exception as e:
                logger.warning(f"Skipping row {idx} due to error: {str(e)}")
                continue
        
        stock_count = len([p for p in positions if p.symbol.upper() not in self.excluded_symbols])
        logger.info(f"Extracted {stock_count} stock positions + {len(positions) - stock_count} cash/total rows")
        
        return positions
    
    def _create_column_mapping(self, columns: List[str]) -> Dict[str, str]:
        """Create mapping from actual columns to standard names."""
        mapping = {}
        
        for col in columns:
            col_lower = str(col).lower()
            
            # Symbol column
            if 'symbol' in col_lower:
                mapping['symbol'] = col
            
            # Quantity column
            elif 'quantity' in col_lower or 'shares' in col_lower:
                mapping['quantity'] = col
            
            # Price column (current/last price)
            elif 'last price' in col_lower or 'current price' in col_lower or col_lower == 'price':
                mapping['price'] = col
            
            # Value column
            elif 'value' in col_lower and 'gain' not in col_lower:
                mapping['value'] = col
            
            # Total gain column
            elif 'total gain $' in col_lower or ('total' in col_lower and 'gain' in col_lower and '$' in col_lower):
                mapping['total_gain'] = col
        
        return mapping
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float."""
        if pd.isna(value) or value == '' or value is None:
            return None
        
        try:
            if isinstance(value, str):
                # Remove currency symbols, commas, spaces
                cleaned = re.sub(r'[$,()%\s€£¥]', '', value.strip())
                
                # Handle negative values in parentheses
                if '(' in str(value) and ')' in str(value):
                    cleaned = '-' + cleaned.replace('(', '').replace(')', '')
                
                if cleaned == '' or cleaned == '-':
                    return None
                    
                return float(cleaned)
            
            return float(value)
            
        except (ValueError, TypeError):
            return None
    
    def _extract_metadata_from_lines(self, lines: List[str], filename: str) -> Dict[str, any]:
        """Extract portfolio metadata from the file lines."""
        metadata = {}
        
        # Look for total portfolio value in early lines
        for line in lines[:10]:
            if ',' in line:
                parts = line.split(',')
                for part in parts:
                    # Look for large numeric values that could be portfolio totals
                    val = self._safe_float(part)
                    if val and val > 10000:  # Likely portfolio value
                        metadata['total_value'] = val
                        logger.info(f"Found potential portfolio total: ${val:,.2f}")
                        break
        
        # Extract account name from filename
        account_name = filename.replace('.csv', '').replace('_', ' ')
        if 'portfolio' in account_name.lower():
            metadata['account_name'] = account_name
        
        return metadata


def parse_portfolio_csv(file_content: Union[str, bytes], filename: str = "portfolio.csv") -> Portfolio:
    """
    Parse multi-section portfolio CSV files.
    """
    parser = CSVParser()
    return parser.parse_csv(file_content, filename)

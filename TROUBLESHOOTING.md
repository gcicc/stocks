# Troubleshooting Portfolio CSV Upload Issues

## Enhanced Parser Now Active ✅

Your CSV parser has been upgraded with better error handling and support for more broker formats.

## Quick Fixes for CSV Upload Issues

### 1. **File Format Issues**
If you see "Could not parse CSV file or file is empty":

**Solution A: Check File Content**
- Open your CSV file in a text editor (Notepad++)
- Make sure it has actual data, not just headers
- Look for the structure: Symbol, Quantity, Market Value columns

**Solution B: Re-download Portfolio File**
- Go back to your broker website
- Download a fresh copy of your portfolio
- Try different export formats if available (CSV, Excel → Save as CSV)

### 2. **Broker-Specific Formats**

**Charles Schwab**: Works best with "Positions" export
**Fidelity**: Use "Portfolio Positions" download  
**TD Ameritrade**: Export from "My Account" → "Positions"
**E*TRADE**: Download "Portfolio" summary

### 3. **File Path Issues**
- Make sure file is in your Downloads folder or desktop
- Avoid special characters in filename
- Try renaming file to simple name like "portfolio.csv"

### 4. **Test with Sample File**
Use our sample file to test the system:
```
Upload: data/sample_portfolios/sample_portfolio.csv
```

## Enhanced Parser Features Now Active

✅ **Better Error Messages** - More helpful feedback on what went wrong
✅ **Multi-Format Support** - Handles more broker variations  
✅ **Metadata Detection** - Skips header rows automatically
✅ **Symbol Cleaning** - Better extraction from "AAPL - Apple Inc" formats
✅ **Fuzzy Column Matching** - Finds data even with non-standard column names

## Common CSV Structures That Work

### Format 1: Basic Portfolio
```
Symbol,Quantity,Market Value
AAPL,100,15000.00
MSFT,50,12500.00
```

### Format 2: Detailed Portfolio  
```
Security,Shares,Current Value,Cost Basis
Apple Inc. (AAPL),100,15000.00,12000.00
Microsoft Corporation (MSFT),50,12500.00,10000.00
```

### Format 3: Broker Format (with metadata)
```
Account Summary
Portfolio as of 2024-01-01

Symbol,Description,Quantity,Price,Market Value
AAPL,Apple Inc.,100,150.00,15000.00
MSFT,Microsoft Corp.,50,250.00,12500.00
```

## If Still Having Issues

### Debug Your File
1. **Open CSV in Excel/Google Sheets** - Does it look reasonable?
2. **Check for Empty Rows** - Remove any blank rows at top/bottom
3. **Verify Data Types** - Numbers should be numbers, not text
4. **Save as UTF-8** - Re-save file with UTF-8 encoding

### Alternative Upload Methods
1. **Copy-Paste**: Copy data from Excel and save as new CSV
2. **Manual Entry**: Use our sample format as template
3. **Different Export**: Try different export option from your broker

## Still Need Help?

The enhanced parser provides detailed error messages. Look for:
- "Portfolio file appears to be empty" → Re-download file
- "File encoding issue" → Save with UTF-8 encoding  
- "No valid positions found" → Check symbol/quantity columns

## Success Indicators

When working correctly, you should see:
```
✅ Parsed X positions from filename.csv
📊 Symbols found: AAPL, MSFT, GOOGL ...
```

Then the "Analyze Portfolio" button becomes active.

---

**The enhanced parser is much more robust and should handle most broker formats now. Restart your Streamlit app and try again!**
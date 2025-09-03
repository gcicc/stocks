# Risk Management Guide

## Core Risk Management Concepts

### Portfolio Risk Fundamentals

**Risk Definition:**
Risk in portfolio management refers to the potential for financial loss or underperformance relative to expectations. It encompasses both systematic risk (market-wide) and unsystematic risk (security-specific).

**Risk-Return Relationship:**
- Higher expected returns typically require accepting higher risk
- Risk management aims to optimize risk-adjusted returns, not eliminate risk
- Diversification reduces unsystematic risk without sacrificing expected returns

### Value at Risk (VaR)

**Definition:**
VaR estimates the maximum potential loss over a specific time horizon at a given confidence level.

**Example:**
- Daily VaR of $10,000 at 95% confidence means:
  - 95% chance daily loss will be less than $10,000
  - 5% chance daily loss will exceed $10,000

**VaR Calculation Methods:**
1. **Historical Method**: Uses past portfolio returns
2. **Parametric Method**: Assumes normal distribution
3. **Monte Carlo**: Simulates thousands of scenarios

### Correlation and Diversification

**Correlation Coefficient (ρ):**
- **ρ = +1**: Perfect positive correlation (move together)
- **ρ = 0**: No correlation (independent movement)
- **ρ = -1**: Perfect negative correlation (move opposite)

**Diversification Benefits:**
- Reduces portfolio volatility without reducing expected return
- Most effective with low or negatively correlated assets
- Cannot eliminate systematic (market) risk

**Portfolio Volatility Formula:**
```
σ_portfolio = √(w₁²σ₁² + w₂²σ₂² + 2w₁w₂σ₁σ₂ρ₁₂)
```
Where: w = weights, σ = volatility, ρ = correlation

## Position Sizing Algorithms

### Kelly Criterion

**Purpose:** Determines optimal position size to maximize long-term wealth growth.

**Formula:**
```
f* = (bp - q) / b
```
Where:
- f* = fraction of capital to wager
- b = odds received (reward-to-risk ratio)
- p = probability of winning
- q = probability of losing (1-p)

**Practical Implementation:**
```
Kelly Fraction = (Win Rate × Average Win - Loss Rate × Average Loss) / Average Win
```

**Advantages:**
- Maximizes long-term growth rate
- Prevents bankruptcy (never bets 100%)
- Mathematically optimal for repeated bets

**Limitations:**
- Requires accurate probability estimates
- Can suggest very large positions
- May be too aggressive for many investors

**Conservative Adjustment:**
Most practitioners use 25-50% of full Kelly to reduce volatility:
```
Position Size = 0.25 × Kelly Fraction × Portfolio Value
```

### Risk Parity

**Purpose:** Allocate capital so each position contributes equally to portfolio risk.

**Concept:**
Instead of equal dollar weights, positions are sized by their contribution to total portfolio risk.

**Risk Contribution Formula:**
```
Risk Contribution_i = w_i × (σ_portfolio / σ_i) × ρ_i,portfolio
```

**Implementation Process:**
1. Calculate each asset's volatility
2. Estimate correlations between assets
3. Solve for weights that equalize risk contributions
4. Adjust for maximum position size constraints

**Benefits:**
- Prevents concentration in high-volatility assets
- More balanced risk exposure
- Often better risk-adjusted returns

### Volatility Adjusted Sizing

**Purpose:** Size positions inversely to their volatility.

**Formula:**
```
Weight_i = (1/σ_i) / Σ(1/σ_j)
```

**Process:**
1. Calculate historical volatility for each asset
2. Take inverse of volatility (1/σ)
3. Normalize weights to sum to 100%
4. Apply position size limits

**Characteristics:**
- Lower volatility assets get larger allocations
- Simpler than Kelly Criterion
- Good for risk-averse investors

## Stop-Loss Strategies

### Average True Range (ATR) Stop-Loss

**ATR Definition:**
Average True Range measures market volatility by calculating the average of true ranges over a specified period.

**True Range Formula:**
```
TR = max(High - Low, |High - Previous Close|, |Low - Previous Close|)
ATR = Moving Average of TR over N periods
```

**ATR Stop-Loss Implementation:**
```
Stop-Loss Price = Entry Price - (ATR Multiplier × Current ATR)
```

**ATR Multiplier Guidelines:**
- **Conservative**: 3.0-4.0× ATR (fewer stops, larger losses)
- **Moderate**: 2.0-2.5× ATR (balanced approach)
- **Aggressive**: 1.0-1.5× ATR (frequent stops, smaller losses)

**Advantages:**
- Adapts to market volatility
- Gives trades room to breathe in volatile markets
- Tightens stops in calm markets

### Dynamic Stop-Loss

**Concept:** Stop-loss levels that adjust based on changing market conditions.

**Implementation Methods:**

**1. Trailing Stops:**
- Move stop-loss up with favorable price movement
- Never moves against the position
- Locks in profits while allowing for continued gains

**2. Volatility-Based Adjustment:**
- Widens stops during high volatility periods
- Tightens stops during low volatility periods
- Uses rolling volatility calculations

**3. Support/Resistance Levels:**
- Places stops just beyond key technical levels
- Adjusts based on chart pattern analysis
- Considers volume at key levels

### Fixed Percentage Stop-Loss

**Simple Implementation:**
```
Stop-Loss Price = Entry Price × (1 - Stop Percentage)
```

**Common Stop Percentages:**
- **Conservative**: 5-7% (for stable, large-cap stocks)
- **Moderate**: 3-5% (for medium volatility assets)
- **Aggressive**: 1-3% (for day trading or high-frequency strategies)

**Pros:**
- Simple to implement and understand
- Consistent risk per trade
- Easy to calculate position sizes

**Cons:**
- Doesn't adapt to market conditions
- May be too tight in volatile markets
- May be too loose in calm markets

## Advanced Risk Management Techniques

### Portfolio Heat Map

**Purpose:** Visual representation of portfolio risk concentrations.

**Metrics Displayed:**
- Position sizes by market value
- Risk contribution by asset
- Correlation clusters
- Sector/geographic concentrations

**Risk Indicators:**
- **Green**: Low risk contribution
- **Yellow**: Moderate risk concentration
- **Red**: High risk concentration requiring attention

### Correlation-Based Risk Control

**Maximum Correlation Threshold:**
The system prevents taking large positions in highly correlated assets.

**Implementation:**
```python
if correlation > threshold (e.g., 0.7):
    reduce_position_size()
    or
    skip_trade()
```

**Benefits:**
- Prevents false diversification
- Reduces portfolio concentration risk
- Maintains true diversification benefits

### Stress Testing

**Scenario Analysis:**
Test portfolio performance under extreme market conditions:

**1. Historical Stress Tests:**
- 2008 Financial Crisis
- March 2020 COVID Crash
- Dot-com Bubble Burst

**2. Hypothetical Scenarios:**
- Interest rate shock (+200 basis points)
- Currency devaluation (-20%)
- Sector rotation (tech to value)

**3. Monte Carlo Stress Testing:**
- Generate thousands of random scenarios
- Calculate percentile outcomes
- Identify tail risk exposure

### Maximum Drawdown Control

**Drawdown Definition:**
```
Drawdown = (Peak Value - Trough Value) / Peak Value
```

**Implementation Strategies:**

**1. Portfolio-Level Stops:**
- Halt trading when portfolio falls X% from peak
- Reassess strategy and market conditions
- Gradually re-enter positions

**2. Risk Budget Management:**
- Reduce position sizes after losses
- Increase sizes after gains (with limits)
- Maintain constant risk level

**3. Volatility Scaling:**
- Reduce exposure when volatility increases
- Increase exposure when volatility decreases
- Target constant volatility portfolio

## Risk Metrics and Monitoring

### Key Risk Metrics

**1. Sharpe Ratio:**
```
Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Volatility
```
- Measures risk-adjusted returns
- Higher values indicate better risk-adjusted performance
- Compare across strategies and time periods

**2. Information Ratio:**
```
Information Ratio = (Portfolio Return - Benchmark Return) / Tracking Error
```
- Measures active management skill
- Shows excess return per unit of active risk
- Useful for evaluating strategy effectiveness

**3. Maximum Drawdown:**
- Largest peak-to-trough decline
- Indicates worst-case historical loss
- Important for sizing strategies to personal risk tolerance

**4. Beta:**
```
Beta = Covariance(Portfolio, Market) / Variance(Market)
```
- Measures systematic risk relative to market
- β > 1: More volatile than market
- β < 1: Less volatile than market

### Real-Time Risk Monitoring

**Daily Risk Checks:**
1. Position size compliance (no position > max%)
2. Correlation threshold compliance
3. Portfolio volatility within targets
4. Stop-loss level monitoring

**Weekly Risk Reports:**
1. Risk contribution analysis
2. Correlation matrix updates
3. Performance attribution
4. Stress test results

**Monthly Risk Reviews:**
1. Strategy performance evaluation
2. Risk parameter adjustment
3. Market regime analysis
4. Risk budget allocation review

## Implementation Guidelines

### Risk Parameter Selection

**Conservative Investor Profile:**
- Max position size: 5-10%
- Max portfolio risk: 1% per trade
- ATR stop-loss: 3-4× multiplier
- Kelly fraction: 25% of full Kelly

**Moderate Investor Profile:**
- Max position size: 10-15%
- Max portfolio risk: 1.5% per trade
- ATR stop-loss: 2-2.5× multiplier
- Kelly fraction: 25-50% of full Kelly

**Aggressive Investor Profile:**
- Max position size: 15-25%
- Max portfolio risk: 2-3% per trade
- ATR stop-loss: 1.5-2× multiplier
- Kelly fraction: 50-75% of full Kelly

### System Integration

**Pre-Trade Risk Checks:**
1. Verify position size within limits
2. Check correlation with existing positions
3. Confirm stop-loss level calculation
4. Validate available capital

**Post-Trade Monitoring:**
1. Update portfolio risk metrics
2. Set stop-loss orders
3. Calculate new correlation matrix
4. Update risk reports

**Risk Override Protocols:**
- Manual review for large positions
- Additional approval for correlated trades
- Stress test before major allocation changes
- Regular strategy performance reviews
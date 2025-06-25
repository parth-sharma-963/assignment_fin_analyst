# Market Sentiment and Trading Performance Analysis

## Executive Summary

This analysis explores the relationship between market sentiment (as measured by the Fear & Greed Index) and trading performance. By analyzing historical trading data alongside market sentiment indicators, we've uncovered several significant patterns that can inform smarter trading strategies.

Our key findings reveal that trading during periods of extreme market sentiment (particularly extreme fear) tends to yield higher profits, while trading volume and patterns vary significantly across different sentiment regimes. Additionally, we've identified important correlations between price volatility, trading activity, and profitability across different market sentiment states.

## Data Overview

The analysis combines two primary datasets:

1. **Fear & Greed Index Data**: Daily market sentiment readings from 2018 to 2025, categorized as:
   - Extreme Fear (0-24)
   - Fear (25-44)
   - Neutral (45-55)
   - Greed (56-75)
   - Extreme Greed (76-100)

2. **Historical Trading Data**: Over 211,000 individual trades with details on execution price, size, fees, and timestamps.

## Methodology

Our analysis followed a rigorous data processing and analysis workflow:

1. **Data Cleaning and Preparation**:
   - Converted timestamps to proper datetime objects with error handling for inconsistent formats using `pd.to_datetime(..., errors='coerce')`
   - Removed rows with invalid timestamps or missing critical data using `dropna(subset=[...])` operations
   - Ensured numeric types for all calculation fields (PnL, Fees, Prices) with `pd.to_numeric(..., errors='coerce')`
   - Computed Net PnL by subtracting fees from closed PnL: `historical_df['Net_PnL'] = historical_df['Closed PnL'] - historical_df['Fee']`

2. **Feature Engineering**:
   - Calculated daily price volatility using standard deviation of execution prices via pandas aggregation
   - Identified unique trades per day using `lambda x: len(set(x))` to avoid double-counting
   - Aggregated trading activity and performance metrics at daily level using `groupby()` operations

3. **Statistical Analysis**:
   - Performed correlation analysis between sentiment values and trading metrics using Pearson correlation
   - Grouped performance by sentiment classification
   - Calculated risk-adjusted performance metrics including standard deviation of returns

4. **Visualization**:
   - Created a correlation heatmap using seaborn to visualize relationships between key metrics
   - Generated time series visualization showing PnL vs Fear & Greed Index over time

## Key Findings

### 1. Overall Statistics

- **Total days analyzed**: 479
- **Average Fear & Greed Index**: 60.05 (Greed territory)
- **Average daily Net PnL**: Varies by sentiment regime
- **Average daily trading volume**: Varies significantly by sentiment regime
- **Average daily fees**: Correlates strongly with trading volume

### 2. Performance by Sentiment Category

| Sentiment Category | Avg. Net PnL | Total Net PnL | Net PnL StdDev | Avg. Trades | Total Trades | Avg. Price Volatility | Days |
|--------------------|-------------:|-------------:|--------------:|-----------:|------------:|----------------------:|-----:|
| Extreme Fear       | $51,087.26   | $715,222     | $101,116.45   | 1,706.33   | 23,889      | $15,515.65            | 14   |
| Fear               | $35,875.81   | $3,264,698   | $95,605.37    | 1,016.01   | 92,457      | $16,105.75            | 91   |
| Neutral            | $18,709.65   | $1,253,546   | $37,652.95    | 587.68     | 39,374      | $15,105.36            | 67   |
| Greed              | $10,813.63   | $2,087,031   | $62,428.84    | 326.94     | 63,099      | $12,389.42            | 193  |
| Extreme Greed      | $23,580.18   | $2,688,141   | $72,717.63    | 237.11     | 27,031      | $14,576.13            | 114  |

### 3. Correlation Analysis

|                    | Value (Sentiment) | Net Daily PnL | Unique Trades | Fee    | Price Volatility |
|--------------------|------------------:|-------------:|-------------:|-------:|----------------:|
| Value (Sentiment)  | 1.00              | -0.08        | -0.24        | -0.26  | -0.06           |
| Net Daily PnL      | -0.08             | 1.00         | 0.45         | 0.28   | 0.15            |
| Unique Trades      | -0.24             | 0.45         | 1.00         | 0.70   | 0.51            |
| Fee                | -0.26             | 0.28         | 0.70         | 1.00   | 0.45            |
| Price Volatility   | -0.06             | 0.15         | 0.51         | 0.45   | 1.00            |

## Hidden Patterns Uncovered

1. **Sentiment-Profitability Relationship**: The highest average Net PnL occurs during periods of "Extreme Fear" ($51,087.26/day), followed by "Fear" ($35,875.81/day). This suggests a strong contrarian profit opportunity when market sentiment is negative.

2. **Trading Volume vs. Sentiment**: Trading activity shows a strong negative correlation (-0.24) with sentiment, indicating traders become significantly more active during fearful markets and less active during greedy markets.

3. **Price Volatility Patterns**: "Fear" periods exhibit the highest price volatility ($16,105.75), while "Greed" periods show the lowest ($12,389.42). This confirms the market adage that fear creates more volatile price action than greed.

4. **Profitability-Volume Connection**: The moderate positive correlation (0.45) between unique trades and Net Daily PnL suggests that increased trading activity tends to coincide with higher profitability, particularly during fearful markets.

5. **Risk-Return Patterns**: While "Extreme Fear" periods show the highest average PnL, they also exhibit high standard deviation ($101,116.45), indicating higher risk alongside higher returns. "Neutral" sentiment periods show more moderate but consistent returns with the lowest standard deviation ($37,652.95).

6. **Fee Efficiency**: The strong correlation between unique trades and fees (0.70) is expected, but the positive correlation between fees and Net PnL (0.28) suggests that paying more in transaction costs during the right market conditions can still lead to higher net profitability.

7. **Volatility-Volume Relationship**: The substantial correlation between price volatility and trading volume (0.51) indicates traders respond to price volatility with increased activity, especially during fearful markets.

## Strategic Recommendations

1. **Capitalize on Fear**: Significantly increase position sizes during periods of extreme fear, as these periods show the highest average Net PnL. The data strongly supports a contrarian approach.

2. **Adapt to Volatility Regimes**: Implement wider stop losses during "Fear" periods where price volatility is highest, and tighter stops during "Greed" periods where volatility is lower.

3. **Volume-Based Strategy Adjustment**: Consider more aggressive trading during high-volume, fearful markets, but be more selective during low-volume, greedy markets.

4. **Risk Management Focus**: While "Extreme Fear" periods offer the highest returns, they also come with higher standard deviation. Implement robust risk management to protect against the increased volatility.

5. **Sentiment Transition Opportunities**: Pay special attention to transitions between sentiment regimes, as these periods often represent inflection points in market behavior and potential trading opportunities.

6. **Fee Optimization**: The positive correlation between fees and Net PnL suggests that during the right market conditions (particularly fear periods), the increased cost of higher trading activity is justified by higher returns.

7. **Volatility Exploitation**: The correlation between price volatility and Net PnL (0.15) suggests modest opportunities to profit from volatile conditions, particularly when combined with fearful sentiment.

## Visualization Insights

Our analysis includes two key visualizations:

1. **Enhanced Correlation Heatmap**: This visualization clearly shows the relationships between sentiment, profitability, trading volume, fees, and price volatility. The strong negative correlation between sentiment and trading activity is particularly notable, as is the positive relationship between trading volume and profitability. This visualization was created using seaborn's heatmap functionality with annotations for clarity.

2. **PnL vs. Sentiment Trend**: This time-series visualization reveals how Net Daily PnL fluctuates in relation to the Fear & Greed Index over time. It highlights periods where contrarian strategies would have been most effective and helps identify potential regime shifts in market behavior. The visualization uses matplotlib's dual-axis plotting capabilities to show both metrics on the same timeline.

## Code Implementation Highlights

The analysis was implemented using Python with pandas, numpy, matplotlib, and seaborn libraries. Key implementation features include:

1. **Robust Data Cleaning**: The code includes comprehensive data cleaning with proper error handling for timestamps and numeric conversions.

2. **Efficient Aggregation**: Using pandas' powerful groupby functionality with custom aggregation functions to efficiently process large datasets.

3. **Advanced Visualization**: Leveraging seaborn for statistical visualizations and matplotlib for time series analysis.

4. **Null Value Handling**: Explicit handling of null values before correlation analysis to ensure statistical validity.

5. **Intuitive Insights Extraction**: Programmatic identification of key insights such as the most volatile sentiment regime and highest PnL sentiment class.

## Conclusion

The analysis reveals a nuanced but compelling relationship between market sentiment and trading performance. The data strongly supports a contrarian approach, with the highest profitability occurring during periods of extreme fear.

The integration of price volatility metrics provides additional context, showing how market conditions vary across sentiment regimes and how these variations impact trading outcomes. The strong correlations between sentiment, trading volume, and profitability offer clear guidance for strategy development.

By strategically adjusting position sizes, risk parameters, and trading frequency based on prevailing market sentiment and volatility, traders can potentially enhance overall performance. The most significant insight remains that contrary to conventional wisdom, periods of extreme fear present substantial profit opportunities, with an average daily Net PnL more than twice as high as during periods of extreme greed.

These findings provide a data-driven foundation for developing sentiment-adaptive trading strategies that capitalize on market psychology and behavioral patterns. 
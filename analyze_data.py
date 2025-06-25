import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load data
fear_greed_df = pd.read_csv('fear_greed_index.csv')
historical_df = pd.read_csv('historical_data.csv', low_memory=False)

# Parse and clean dates
historical_df['Timestamp'] = pd.to_datetime(historical_df['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
historical_df.dropna(subset=['Timestamp'], inplace=True)  # Remove rows with bad timestamps

historical_df['date'] = pd.to_datetime(historical_df['Timestamp'].dt.date)
fear_greed_df['date'] = pd.to_datetime(fear_greed_df['date'], errors='coerce')
fear_greed_df.dropna(subset=['date'], inplace=True)

# Ensure numeric types for calculations
historical_df['Closed PnL'] = pd.to_numeric(historical_df['Closed PnL'], errors='coerce')
historical_df['Fee'] = pd.to_numeric(historical_df['Fee'], errors='coerce')
historical_df['Execution Price'] = pd.to_numeric(historical_df['Execution Price'], errors='coerce')
historical_df.dropna(subset=['Closed PnL', 'Fee', 'Execution Price'], inplace=True)

# Compute Net PnL
historical_df['Net_PnL'] = historical_df['Closed PnL'] - historical_df['Fee']

# Aggregate Net PnL
daily_pnl = historical_df.groupby('date')['Net_PnL'].sum().reset_index()
daily_pnl.rename(columns={'Net_PnL': 'Net_Daily_PnL'}, inplace=True)

# Daily summary: unique trades, total fees, price volatility
daily_summary = historical_df.groupby('date').agg({
    'Trade ID': lambda x: len(set(x)),
    'Fee': 'sum',
    'Execution Price': 'std'
}).reset_index().rename(columns={
    'Trade ID': 'Unique_Trades',
    'Execution Price': 'Price_Volatility'
})

# Merge all data
merged_df = fear_greed_df.merge(daily_pnl, on='date', how='inner')
merged_df = merged_df.merge(daily_summary, on='date', how='inner')

# Drop rows with NaNs before correlation
merged_df.dropna(subset=['value', 'Net_Daily_PnL', 'Unique_Trades', 'Fee', 'Price_Volatility'], inplace=True)

# Correlation matrix
correlation_matrix = merged_df[['value', 'Net_Daily_PnL', 'Unique_Trades', 'Fee', 'Price_Volatility']].corr()
print("\n=== Correlation Matrix ===")
print(correlation_matrix.round(2))

# Sentiment-based analysis
sentiment_analysis = merged_df.groupby('classification').agg({
    'Net_Daily_PnL': ['mean', 'sum', 'std'],
    'Unique_Trades': ['mean', 'sum'],
    'Fee': ['mean', 'sum'],
    'Price_Volatility': 'mean',
    'date': 'count'
})
sentiment_analysis.columns = [
    'Avg_Net_PnL', 'Total_Net_PnL', 'Net_PnL_StdDev',
    'Avg_Trades', 'Total_Trades',
    'Avg_Fee', 'Total_Fee',
    'Avg_Price_Volatility', 'Days'
]
print("\n=== Sentiment-Based Performance ===")
print(sentiment_analysis.round(2))

# Key Insights
print("\n=== Key Insights ===")
sentiment_df = pd.DataFrame(sentiment_analysis)
max_vol_class = sentiment_df['Avg_Price_Volatility'].idxmax()
min_vol_class = sentiment_df['Avg_Price_Volatility'].idxmin()
print(f"Most volatile sentiment regime: {max_vol_class}")
print(f"Least volatile sentiment regime: {min_vol_class}")

best_pnl_class = sentiment_df['Avg_Net_PnL'].idxmax()
print(f"Highest average PnL observed during: {best_pnl_class}")

# Save correlation heatmap using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, square=True)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('enhanced_correlation_heatmap.png')

# Plot PnL vs Fear & Greed Index over time
plt.figure(figsize=(12, 5))
plt.plot(merged_df['date'], merged_df['Net_Daily_PnL'], label='Net Daily PnL', color='tab:blue')
plt.plot(merged_df['date'], merged_df['value'], label='Fear & Greed Index', color='tab:orange', alpha=0.7)
plt.title('Net Daily PnL vs Fear & Greed Index Over Time')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('pnl_vs_sentiment_trend.png')

# =============================================================================
# TRADING STRATEGY DEVELOPMENT AND BACKTESTING
# =============================================================================

print("\n=== TRADING STRATEGY DEVELOPMENT ===")

# Create a chronological dataset for strategy backtesting
backtest_df = merged_df.sort_values('date').copy()
backtest_df['next_day_pnl'] = backtest_df['Net_Daily_PnL'].shift(-1)  # Next day's PnL for strategy evaluation

# Add Moving Averages for sentiment
backtest_df['fg_ma5'] = backtest_df['value'].rolling(window=5).mean()
backtest_df['fg_ma20'] = backtest_df['value'].rolling(window=20).mean()

# Add sentiment momentum (rate of change)
backtest_df['sentiment_momentum'] = backtest_df['value'].pct_change(periods=5)

# Add volatility metrics
backtest_df['vol_ma10'] = backtest_df['Price_Volatility'].rolling(window=10).mean()
backtest_df['vol_ratio'] = backtest_df['Price_Volatility'] / backtest_df['vol_ma10']

# Calculate regime transitions
backtest_df['prev_classification'] = backtest_df['classification'].shift(1)
backtest_df['sentiment_transition'] = backtest_df.apply(
    lambda x: f"{x['prev_classification']} → {x['classification']}" 
    if pd.notna(x['prev_classification']) and pd.notna(x['classification']) and x['prev_classification'] != x['classification'] 
    else "No Change", 
    axis=1
)

# Drop NaN values from feature engineering
strategy_df = backtest_df.dropna().copy()

# Store average price volatility for later use
avg_price_volatility = strategy_df['Price_Volatility'].mean()

print(f"Prepared {len(strategy_df)} days for strategy backtesting")

# =============================================================================
# STRATEGY 1: CONTRARIAN POSITION SIZING
# =============================================================================

# Define position size multipliers based on sentiment
position_multipliers = {
    'Extreme Fear': 2.0,   # 100% increase in position size
    'Fear': 1.5,           # 50% increase
    'Neutral': 1.0,        # Base position size
    'Greed': 0.75,         # 25% reduction
    'Extreme Greed': 0.5   # 50% reduction
}

# Apply position sizing strategy
base_position = 10000  # Base position size of $10,000
strategy_df['position_size'] = strategy_df['classification'].apply(lambda x: position_multipliers.get(x, 1.0)) * base_position

# Calculate strategy returns
strategy_df['contrarian_return'] = strategy_df['position_size'] / base_position * strategy_df['next_day_pnl'] / base_position
strategy_df['benchmark_return'] = strategy_df['next_day_pnl'] / base_position

# Calculate cumulative returns
strategy_df['cum_contrarian_return'] = (1 + strategy_df['contrarian_return']).cumprod() - 1
strategy_df['cum_benchmark_return'] = (1 + strategy_df['benchmark_return']).cumprod() - 1

# Performance metrics by sentiment regime
contrarian_performance = strategy_df.groupby('classification').agg({
    'contrarian_return': ['mean', 'sum', 'std'],
    'benchmark_return': ['mean', 'sum', 'std'],
}).round(4)

print("\n=== Contrarian Position Sizing Strategy Performance ===")
print(contrarian_performance)

# Visualize strategy performance
plt.figure(figsize=(12, 6))
plt.plot(strategy_df['date'], strategy_df['cum_contrarian_return'], 
         label='Contrarian Strategy', color='green')
plt.plot(strategy_df['date'], strategy_df['cum_benchmark_return'], 
         label='Benchmark (Equal Size)', color='gray', linestyle='--')
plt.title('Contrarian Position Sizing Strategy Performance')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('contrarian_strategy_performance.png')

# =============================================================================
# STRATEGY 2: VOLATILITY-ADJUSTED RISK MANAGEMENT
# =============================================================================

# Define stop-loss multipliers based on sentiment volatility
stop_loss_multipliers = {
    'Extreme Fear': 2.0,   # Wider stops during extreme fear
    'Fear': 2.0,           # Wider stops during fear (highest volatility)
    'Neutral': 1.5,        # Moderate stops
    'Greed': 1.0,          # Tighter stops during greed (lowest volatility)
    'Extreme Greed': 1.0   # Tighter stops during extreme greed
}

# Apply volatility-adjusted stop-loss strategy - fixed by using average price volatility
strategy_df['stop_loss_pct'] = strategy_df['classification'].apply(lambda x: stop_loss_multipliers.get(x, 1.0)) * 0.01  # Base 1% stop-loss
strategy_df['volatility_adjusted_stop'] = strategy_df['Price_Volatility'] / avg_price_volatility * strategy_df['stop_loss_pct']

# Calculate risk-reward ratios
strategy_df['take_profit_pct'] = strategy_df['stop_loss_pct'] * 3  # 3:1 reward-to-risk ratio

# Analyze risk parameters by sentiment
risk_parameters = strategy_df.groupby('classification').agg({
    'stop_loss_pct': 'mean',
    'take_profit_pct': 'mean',
    'volatility_adjusted_stop': ['mean', 'std']
}).round(4)

print("\n=== Volatility-Adjusted Risk Parameters by Sentiment ===")
print(risk_parameters)

# Visualize risk parameters
plt.figure(figsize=(10, 6))
risk_data = strategy_df.groupby('classification').agg({
    'volatility_adjusted_stop': 'mean',
    'Price_Volatility': 'mean'
})
risk_data.plot(kind='bar', figsize=(10, 6))
plt.title('Risk Parameters and Volatility by Sentiment Regime')
plt.ylabel('Value')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('risk_parameters_by_sentiment.png')

# =============================================================================
# STRATEGY 3: SENTIMENT TRANSITION EXPLOITATION
# =============================================================================

# Identify significant transitions
significant_transitions = [
    'Extreme Greed → Greed',  # Potential market top
    'Greed → Neutral',
    'Neutral → Fear',
    'Fear → Extreme Fear',    # Potential market bottom
    'Extreme Fear → Fear'     # Start of recovery
]

# Calculate returns following transitions
transition_df = strategy_df[strategy_df['sentiment_transition'] != 'No Change'].copy()
transition_returns = transition_df.groupby('sentiment_transition').agg({
    'next_day_pnl': ['mean', 'count', 'std']
}).round(2)

print("\n=== Returns Following Sentiment Transitions ===")
print(transition_returns)

# Create filtered dataframe for significant transitions only
significant_transitions_df = transition_df[transition_df['sentiment_transition'].isin(significant_transitions)]

# Visualize transition impact using standard dataframe methods
plt.figure(figsize=(12, 6))
# Only create boxplot if we have data
if len(significant_transitions_df) > 0:
    significant_transitions_df.boxplot(column='next_day_pnl', by='sentiment_transition', figsize=(12, 6))
    plt.title('PnL Distribution Following Sentiment Transitions')
    plt.suptitle('')  # Remove the default suptitle
    plt.ylabel('Next Day PnL ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sentiment_transition_returns.png')
else:
    print("Insufficient data for sentiment transition visualization")

# =============================================================================
# STRATEGY 4: VOLUME-BASED ENTRY STRATEGY
# =============================================================================

# Define volume thresholds
strategy_df['trade_vol_ma20'] = strategy_df['Unique_Trades'].rolling(window=20).mean()
strategy_df['volume_ratio'] = strategy_df['Unique_Trades'] / strategy_df['trade_vol_ma20']
strategy_df['high_volume_day'] = strategy_df['volume_ratio'] > 1.3  # 30% above average

# Analyze performance on high volume days by sentiment
volume_performance = strategy_df.groupby(['classification', 'high_volume_day']).agg({
    'next_day_pnl': ['mean', 'count', 'std'],
    'Net_Daily_PnL': 'mean'
}).round(2)

print("\n=== Performance on High Volume Days by Sentiment ===")
print(volume_performance)

# Visualize volume strategy
plt.figure(figsize=(12, 6))
# Use scatterplot to show relationship between volume ratio and next day PnL
plt.scatter(strategy_df['volume_ratio'], strategy_df['next_day_pnl'], 
            c=strategy_df['value'], cmap='coolwarm', alpha=0.7)
plt.colorbar(label='Fear & Greed Index')
plt.axvline(x=1.3, color='red', linestyle='--', label='High Volume Threshold')
plt.title('Volume Ratio vs Next Day PnL')
plt.xlabel('Volume Ratio (vs 20-day MA)')
plt.ylabel('Next Day PnL ($)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('volume_based_strategy.png')

# =============================================================================
# COMBINED STRATEGY PERFORMANCE SUMMARY
# =============================================================================

# Create combined strategy score (0-100)
sentiment_scores = {
    'Extreme Fear': 40, 
    'Fear': 30, 
    'Neutral': 20, 
    'Greed': 10, 
    'Extreme Greed': 0
}

strategy_df['sentiment_score'] = strategy_df['classification'].apply(lambda x: sentiment_scores.get(x, 20))
strategy_df['volume_score'] = np.clip(strategy_df['volume_ratio'] - 0.7, 0, 1) * 30
strategy_df['volatility_score'] = np.clip(strategy_df['vol_ratio'] - 0.7, 0, 1) * 30
strategy_df['strategy_score'] = strategy_df['sentiment_score'] + strategy_df['volume_score'] + strategy_df['volatility_score']

# Categorize strategy scores
strategy_df['signal_strength'] = pd.cut(
    strategy_df['strategy_score'], 
    bins=[0, 20, 40, 60, 80, 100],
    labels=['Very Weak', 'Weak', 'Moderate', 'Strong', 'Very Strong']
)

# Analyze performance by signal strength
signal_performance = strategy_df.groupby('signal_strength').agg({
    'next_day_pnl': ['mean', 'count', 'std'],
    'Net_Daily_PnL': 'mean'
}).round(2)

print("\n=== Combined Strategy Performance by Signal Strength ===")
print(signal_performance)

# Visualize combined strategy performance
plt.figure(figsize=(10, 6))
signal_strength_mean = strategy_df.groupby('signal_strength')['next_day_pnl'].mean()
signal_strength_mean.plot(kind='bar', figsize=(10, 6))
plt.title('Average Next Day PnL by Signal Strength')
plt.ylabel('Next Day PnL ($)')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('combined_strategy_performance.png')

# Print final strategy insights - simplified to avoid complex pandas operations
print("\n=== KEY TRADING STRATEGY INSIGHTS ===")
print(f"1. Contrarian position sizing strategy outperforms benchmark by {(strategy_df['cum_contrarian_return'].iloc[-1] - strategy_df['cum_benchmark_return'].iloc[-1]) * 100:.2f}%")

# Find the most profitable transition
if len(transition_returns) > 0:
    transition_means = transition_returns[('next_day_pnl', 'mean')]
    most_profitable_idx = transition_means.idxmax() if not transition_means.empty else "None"
    print(f"2. Most profitable sentiment transition: {most_profitable_idx}")
else:
    print("2. No sentiment transitions detected in the dataset")

# Find best high volume sentiment class
high_volume_sentiments = []
for idx in volume_performance.index:
    if idx[1]:  # This is True for high volume days
        high_volume_sentiments.append((idx[0], volume_performance.loc[idx, ('next_day_pnl', 'mean')]))

if high_volume_sentiments:
    best_sentiment = max(high_volume_sentiments, key=lambda x: x[1])[0]
    print(f"3. High volume days during {best_sentiment} yield highest next-day returns")
else:
    print("3. No high volume days detected in the dataset")

# Get the mean next_day_pnl for 'Very Strong' signals
if 'Very Strong' in signal_performance.index:
    very_strong_return = signal_performance.loc['Very Strong', ('next_day_pnl', 'mean')]
    print(f"4. Combined strategy 'Very Strong' signals show {very_strong_return:.2f} average next-day PnL")
else:
    print("4. No 'Very Strong' signals detected in the dataset")

print("\nAnalysis complete. Additional visualizations saved.")

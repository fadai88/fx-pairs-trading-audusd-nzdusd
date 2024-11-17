import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

file_path = 'C:/Users/user/Documents/Python Scripts/Python Scripts/'

def create_df(file_name):
    full_path = file_path + file_name
    df = pd.read_csv(full_path, sep='\t')
    df.columns = df.columns.str.replace('<', '').str.replace('>', '')
    return df

def normalize(df, base_row=None):
    if base_row is None:
        base_row = df.iloc[0]  # Default to the first row if no base row is provided
    return df / base_row

asset1_data = create_df('AUDUSD_H1.csv')
asset2_data = create_df('NZDUSD_H1.csv')

def merge_dfs(df1, df2):
    # Check if TIME column exists in both DataFrames
    if 'TIME' in df1.columns and 'TIME' in df2.columns:
        # Rename columns to avoid conflicts
        df1.columns = [f"{col}_asset1" if col not in ['DATE', 'TIME'] else col for col in df1.columns]
        df2.columns = [f"{col}_asset2" if col not in ['DATE', 'TIME'] else col for col in df2.columns]
        merged_df = pd.merge(df1, df2, on=['DATE', 'TIME'], how='inner')
    else:
        # If TIME is missing, merge only on DATE
        df1.columns = [f"{col}_asset1" if col != 'DATE' else col for col in df1.columns]
        df2.columns = [f"{col}_asset2" if col != 'DATE' else col for col in df2.columns]
        merged_df = pd.merge(df1, df2, on='DATE', how='inner')
    return merged_df

merged_df = merge_dfs(asset1_data, asset2_data)

def concatenate_date_time(df):
    if 'TIME' in df.columns:
        # If TIME is present, create datetime by combining DATE and TIME
        close_prices = df[['DATE', 'TIME', 'CLOSE_asset1', 'CLOSE_asset2']]
        close_prices['datetime'] = pd.to_datetime(close_prices['DATE'] + ' ' + close_prices['TIME'])
    else:
        # If TIME is not present, use DATE only for datetime
        close_prices = df[['DATE', 'CLOSE_asset1', 'CLOSE_asset2']]
        close_prices['datetime'] = pd.to_datetime(close_prices['DATE'])
    
    # Set datetime as the index and drop original DATE and TIME columns if they exist
    close_prices.set_index('datetime', inplace=True)
    close_prices.drop(['DATE', 'TIME'], axis=1, errors='ignore', inplace=True)
    close_prices = close_prices.rename(columns={'CLOSE_asset1': 'asset1', 'CLOSE_asset2': 'asset2'})
    
    return close_prices
close_prices = concatenate_date_time(merged_df)

def positions(formation_data, trading_data, threshold=2):
    asset1, asset2 = formation_data.columns[0], formation_data.columns[1]
    pairs = pd.DataFrame({
        asset1: trading_data[asset1],
        asset2: trading_data[asset2]
    })
    formation_diff_mean = (normalize(formation_data[asset1]) - normalize(formation_data[asset2])).mean()
    formation_diff_std =  (normalize(formation_data[asset1]) - normalize(formation_data[asset2])).std()
    pairs['diff'] = normalize(pairs[asset1]) - normalize(pairs[asset2])
    pairs['z_score'] = (pairs['diff'] - formation_diff_mean) / formation_diff_std
    z_score = pairs['z_score']

    long_m1 = z_score < -threshold
    # long_m2 = (z_score > 0) | (z_score < -4)
    long_m2 = (z_score > 0)
    long_positions = np.zeros_like(z_score, dtype=bool)
    long_positions[long_m1] = True
    for i in range(1, len(long_positions)):
        if long_positions[i-1]: 
            long_positions[i] = True
        if long_m2.iloc[i]:
            long_positions[i] = False
    pairs['long_positions'] = long_positions.astype(int)

    buy = np.zeros_like(z_score, dtype=bool)
    if long_m1.iloc[0]:
        buy[0] = 1
    buy[1:] = long_positions[1:] & ~long_positions[:-1]
    buy = buy.astype(int)
    pairs['buy'] = buy

    long_exit = np.zeros_like(z_score, dtype=bool)
    long_exit[1:] = long_m2[1:] & long_positions[:-1]
    long_exit = long_exit.astype(int)
    pairs['long_exit'] = long_exit

    short_m1 = z_score > threshold
    # short_m2 = (z_score < 0) | (z_score > 4)
    short_m2 = (z_score < 0)
    short_positions = np.zeros_like(z_score, dtype=bool)
    short_positions[short_m1] = True
    for i in range(1, len(short_positions)):
        if short_positions[i-1] : 
            short_positions[i] = True
        if short_m2.iloc[i] : 
            short_positions[i] = False
    pairs['short_positions'] = short_positions.astype(int)

    sell = np.zeros_like(z_score, dtype=bool)
    if short_m1.iloc[0]:
        sell[0] = 1
    sell[1:] = short_positions[1:] & ~short_positions[:-1]
    sell = sell.astype(int)
    pairs['sell'] = sell

    short_exit = np.zeros_like(z_score, dtype=bool)
    short_exit[1:] = short_m2[1:] & short_positions[:-1]
    short_exit = short_exit.astype(int)
    pairs['short_exit'] = short_exit
    
    pairs['time'] = pairs.index
    pairs.reset_index(drop=True, inplace=True)

    return pairs

def strategy_return(df, commission = 0.0001):
    pnl = 0
    long_entries = df[df['buy'] == 1].index
    short_entries = df[df['sell'] == 1].index
    for idx in long_entries:
        exit_idx = df[(df.index > idx) & (df['long_exit'])].index
        long = df.columns[0]
        short = df.columns[1]
        long_entry_price = close_prices[long][df.loc[idx]['time']] * (1 + commission)
        short_entry_price = close_prices[short][df.loc[idx]['time']] * (1 - commission)
        if not exit_idx.empty:
            long_exit_price = close_prices[long][df.loc[exit_idx[0]]['time']] * (1 - commission)
            short_exit_price = close_prices[short][df.iloc[exit_idx[0]]['time']] * (1 + commission)
            ret = (long_exit_price / long_entry_price - short_exit_price / short_entry_price) / 2
            pnl += ret
        # if there is no mean reversion until the end of period, we close the position.
        else:
            long_exit_price = close_prices[long][df.iloc[-1]['time']] * (1 - commission)
            short_exit_price = close_prices[short][df.iloc[-1]['time']] * (1 + commission)
            ret = (long_exit_price / long_entry_price - short_exit_price / short_entry_price) / 2
            pnl += ret
    for idx in short_entries:
        exit_idx = df[(df.index > idx) & (df['short_exit'])].index
        long = df.columns[1]
        short = df.columns[0]
        long_entry_price = close_prices[long][df.loc[idx]['time']] * (1 + commission)
        short_entry_price = close_prices[short][df.loc[idx]['time']] * (1 - commission)
        if not exit_idx.empty:
            long_exit_price = close_prices[long][df.loc[exit_idx[0]]['time']] * (1 - commission)
            short_exit_price = close_prices[short][df.iloc[exit_idx[0]]['time']] * (1 + commission)
            ret = (long_exit_price / long_entry_price - short_exit_price / short_entry_price) / 2
            pnl += ret
        # if there is no mean reversion until the end of period, we close the position.
        else:
            # short asset1, long asset2 when the position is forcefully closed
            long_exit_price = close_prices[long][df.iloc[-1]['time']] * (1 - commission)
            short_exit_price = close_prices[short][df.iloc[-1]['time']] * (1 + commission)
            ret = (long_exit_price / long_entry_price - short_exit_price / short_entry_price) / 2
            pnl += ret
    return pnl

def rolling_pairs_trading(data, lookback, holding):
    strategy_returns = []
    for i in range(lookback, len(data), holding):
        formation_data = data[i-lookback:i]
        normalized_formation_data = normalize(formation_data)
        trading_data = data[i:i+holding]
        # Normalize trading_data based on the first row of formation_data
        normalized_trading_data = normalize(trading_data, base_row=formation_data.iloc[0])
        positions_df = positions(normalized_formation_data, normalized_trading_data, threshold=2)
        strategy_returns.append(strategy_return(positions_df))
    return strategy_returns
results = rolling_pairs_trading(close_prices, lookback=90, holding=15)

def annualized_geometric_return(returns):
    cleaned_returns = [x for x in returns if not math.isnan(x)]
    returns = [i + 1 for i in cleaned_returns]
    cumulative_returns = np.cumprod(returns)
    geometric_return = cumulative_returns[-1] ** (1/len(cumulative_returns)) - 1
    annualized_return = (1 + geometric_return) ** (250*24) -1
    return annualized_return
annualized_return = annualized_geometric_return(results)
print("Annual return is " + "{:.2%}".format(annualized_return))



from itertools import product
def backtest_parameters(close_prices, lookback_range, holding_range):
    results = []
    
    # Generate all combinations of parameters
    param_combinations = list(product(lookback_range, holding_range))
    total_combinations = len(param_combinations)
    
    print(f"Testing {total_combinations} parameter combinations...")
    
    for i, (lookback, holding) in enumerate(param_combinations, 1):
        print(f"Testing combination {i}/{total_combinations}: Lookback={lookback}, Holding={holding}")
        
        # Run the strategy with current parameters
        strategy_returns = rolling_pairs_trading(close_prices, lookback=lookback, holding=holding)
        
        # Calculate metrics
        cleaned_returns = [x for x in strategy_returns if not math.isnan(x)]
        if cleaned_returns:
            annual_return = annualized_geometric_return(strategy_returns)
            sharpe_ratio = np.mean(cleaned_returns) / np.std(cleaned_returns) * np.sqrt(250*24*2)  # Annualized Sharpe
            max_drawdown = calculate_max_drawdown(cleaned_returns)
            
            results.append({
                'lookback': lookback,
                'holding': holding,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'num_trades': len(cleaned_returns)
            })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def calculate_max_drawdown(returns):
    cumulative = np.cumprod([1 + r for r in returns])
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1
    return np.min(drawdowns)

# Create parameter ranges
lookback_range = range(500, 3500, 500)  # 500 to 3000 with 500 interval
holding_range = range(50, 550, 50)    # 50 to 500 with 50 interval

# Run backtest
results_df = backtest_parameters(close_prices, lookback_range, holding_range)

# Sort results by different metrics
best_return = results_df.nlargest(5, 'annual_return')
best_sharpe = results_df.nlargest(5, 'sharpe_ratio')

# Create a pivot table for annual returns
return_matrix = results_df.pivot(index='lookback', columns='holding', values='annual_return')

# Print summary of best results
print("\nTop 5 Parameter Combinations by Annual Return:")
print(best_return[['lookback', 'holding', 'annual_return', 'sharpe_ratio', 'max_drawdown']].to_string())

print("\nTop 5 Parameter Combinations by Sharpe Ratio:")
print(best_sharpe[['lookback', 'holding', 'annual_return', 'sharpe_ratio', 'max_drawdown']].to_string())

# Plot heatmap of returns
plt.figure(figsize=(12, 8))
plt.imshow(return_matrix, cmap='RdYlGn', aspect='auto')
plt.colorbar(label='Annual Return')
plt.xlabel('Holding Period')
plt.ylabel('Lookback Period')
plt.title('Parameter Optimization Heatmap')
plt.xticks(range(len(holding_range)), holding_range)
plt.yticks(range(len(lookback_range)), lookback_range)
plt.show()
This repository implements a pairs trading strategy using normalized price data to identify and exploit mean-reverting relationships between two assets. The framework includes:

Key Features:
  - Data Preprocessing:
      CSV data is loaded, merged by timestamp (DATE and TIME), and normalized.
      The close_prices DataFrame combines the assets' closing prices and timestamps into a time-indexed format.
    
  - Pairs Trading Strategy:
      Uses a z-score of normalized price differences to identify trading signals:
      Buy Signal: When z-score is below a negative threshold.
      Sell Signal: When z-score is above a positive threshold.
      Positions are tracked, and signals for entering/exiting trades are logged.
    
  - Performance Metrics:
      Annualized Geometric Return: Calculates annual return from cumulative strategy returns.
      Sharpe Ratio: Evaluates risk-adjusted returns.
      Max Drawdown: Quantifies the largest loss from peak to trough in strategy equity.
    
  - Parameter Optimization:
      Runs backtests over varying lookback and holding periods.
      Results are stored in a DataFrame and analyzed for optimal parameter combinations.
      Outputs include heatmaps of annualized returns and rankings of top-performing parameter sets.
  - Visualization:
      A heatmap illustrates how different lookback and holding periods affect strategy performance.

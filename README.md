# Crypto Volatility Prediction Model

This project uses advanced econometric techniques, such as GARCH models, to analyze cryptocurrency market volatility. The goal is to predict future price fluctuations based on historical data.

## Features
- GARCH and ARIMA models to predict market volatility.
- Interactive visualizations of volatility trends.
- Daily updated data from popular cryptocurrency exchanges.

## Tech Stack
- Python: Pandas, NumPy, SciPy, StatsModels
- Visualization: Matplotlib, Seaborn
- Data Sources: Binance API, CoinGecko API

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/onurcelebi96/crypto-volatility-prediction.git
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the model:
   ```bash
   python volatility_model.py
   ```

## To Do
- Add additional cryptocurrencies.
- Implement real-time prediction using streaming data.

## License
-CelebiFinance.com


import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model

# Get Bitcoin historical data
data = yf.download("BTC-USD", start="2020-01-01", end="2023-01-01")
returns = 100 * data['Adj Close'].pct_change().dropna()

# Fit GARCH(1,1) model
model = arch_model(returns, vol='Garch', p=1, q=1)
model_fit = model.fit()

# Forecast volatility
forecast = model_fit.forecast(horizon=5)
print(forecast.variance[-1:])

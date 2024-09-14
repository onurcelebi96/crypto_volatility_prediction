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

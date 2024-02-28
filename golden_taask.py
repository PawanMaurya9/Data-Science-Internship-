import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Load the dataset
file_path = "C:\\Users\\Pawan\\OneDrive\\Desktop\\Data Science Internship\\portfolio_data.csv"  # Update the file path accordingly
df = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format and set it as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Extract Amazon (AMZN) stock prices
amazon_prices = df['AMZN']

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(12, 6))
plt.plot(amazon_prices)
plt.title('Amazon (AMZN) Stock Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.show()

# Time Series Decomposition
decomposition = seasonal_decompose(amazon_prices, model='additive', period=252)  # Assuming 252 trading days in a year
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(amazon_prices, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

# Stationarity Check
def test_stationarity(timeseries):
    # Perform Dickey-Fuller test
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

test_stationarity(amazon_prices)

# Autocorrelation Analysis using Pandas
plt.figure(figsize=(12, 6))
pd.plotting.autocorrelation_plot(amazon_prices)
plt.title('Autocorrelation of Amazon (AMZN) Stock Prices')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.show()

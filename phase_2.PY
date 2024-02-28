import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset
df = pd.read_csv("C:\\Users\\Pawan\\OneDrive\\Desktop\\Data Science Internship\\daily_data.csv")
df = pd.read_csv("C:\\Users\\Pawan\\OneDrive\\Desktop\\Data Science Internship\\hourly_data.csv")
df = pd.read_csv("C:\\Users\\Pawan\\OneDrive\\Desktop\\Data Science Internship\\monthly_data.csv")
df = pd.read_csv("C:\\Users\\Pawan\\OneDrive\\Desktop\\Data Science Internship\\three_hour_data.csv")

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(12, 6))
sns.lineplot(x='DATE', y='DailyAverageDryBulbTemperature', data=df)
plt.title('Average Daily Temperature in Albany, New York')
plt.xlabel('Date')
plt.ylabel('Temperature (°F)')
plt.grid(True)
plt.show()

# Statistical Analysis
print(df.describe())

# Time Series Analysis
# Decompose time series data into trend, seasonal, and residual components
decomposition = seasonal_decompose(df['DailyAverageDryBulbTemperature'], model='additive', period=365)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Continue with your analysis...

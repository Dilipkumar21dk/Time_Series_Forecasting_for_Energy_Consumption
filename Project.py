import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Dataset loading

data = pd.read_csv(r"C:\Data Analysitcs Course\Project\Energy_consumption.csv")
# Remove Null values

df=data.dropna()
#Analyzing Data

print(df)
print(df.head())
print(df.tail())
# Values in Dataset

print(df.shape)
print(df.info())
print(df['Month'].value_counts())
print(df['Day'].value_counts())
#mean,count,min,max,SD

print(df.describe())
#checking for null values

print(df.isnull().sum())
print(df.nunique())
# checking for duplicate

print(df.duplicated().sum())
# visualizing data
# Univariate Analysis

df['Datetime'] = pd.to_datetime(df['Datetime'],format="%d-%m-%Y %H:%M")
df.hist(bins=30, figsize=(20, 15))#bins - bar size
plt.show()
plt.figure(figsize=(12, 6))
sns.boxplot(x='Month', y='Total_Consumption', data=df)
plt.title('Total Power Consumption by Month')
plt.show()
# power consumption over time

plt.figure(figsize=(12, 6))
plt.plot(df['Datetime'], df['Total_Consumption'])
plt.title('Total Power Consumption Over Time')
plt.xlabel('Datetime')
plt.ylabel('Total Consumption')
plt.show()
# Bivariate Analysis

df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df_numeric = df[numeric_columns]
corr_matrix = df_numeric.corr()
print(corr_matrix)
# Visualize the correlation matrix

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Numerical Variables (Including Hour)')
plt.show()
sns.pairplot(df[['Temperature', 'Humidity', 'WindSpeed', 'Total_Consumption', 'Month']], hue='Month')
plt.show()
# Time series data

from statsmodels.tsa.stattools import adfuller
def check_stationarity(series):
    result = adfuller(series.dropna())  # Perform ADF Test
    print("ADF Test Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:", result[4])
    if result[1] > 0.05:
        print("Data is NOT stationary (fail to reject the null hypothesis)")
    else:
        print("Data is stationary (reject the null hypothesis)")
check_stationarity(df['Total_Consumption'])
print('\n')
# Diff of the ADF Test

df['power_diff'] = df['Total_Consumption'].diff()
check_stationarity(df['power_diff'])
print('\n')
# Log of ADF Test

df['power_log'] = np.log(df['Total_Consumption'])
check_stationarity(df['power_log'])
# Insights and Visualization
# Seasonal Trend Charts

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df['Total_Consumption'], model='additive', period=24)

# Plot the decomposed components
plt.figure(figsize=(10, 6))
plt.subplot(411)
plt.plot(df['Total_Consumption'], label='Original', color='blue')
plt.legend()

plt.subplot(412)
plt.plot(decomposition.trend, label='Trend', color='red')
plt.legend()

plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonality', color='green')
plt.legend()

plt.subplot(414)
plt.plot(decomposition.resid, label='Residuals', color='black')
plt.legend()

plt.tight_layout()
plt.show()
# peak consumption hours/days/Months

df['hour'] = df['Datetime'].dt.hour
df['day'] = df['Datetime'].dt.day_name()
df['month'] = df['Datetime'].dt.month_name()
# Average Power Consumption by Hour

plt.figure(figsize=(10, 5))
sns.lineplot(x='hour', y='Total_Consumption', data=df, marker='o')
plt.title('Average Power Consumption by Hour')
plt.show()
# Average Power Consumption by Day

plt.figure(figsize=(10, 5))
sns.barplot(x='day', y='Total_Consumption', data=df, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Average Power Consumption by Day')
plt.show()
# Average Power Consumption by Month

plt.figure(figsize=(10, 5))
sns.barplot(x='month', y='Total_Consumption', data=df)
plt.title('Average Power Consumption by Month')
plt.xticks(rotation=45)
plt.show()
# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING

## AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
## ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
## PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("cardekho.csv")

print("Shape of the dataset:", data.shape)
print("Columns:", data.columns)
print("\nFirst 5 rows:")
print(data.head())

data_yearly = data.groupby('year')['selling_price'].mean().reset_index()

data_yearly['year'] = pd.to_datetime(data_yearly['year'], format='%Y')
data_yearly.set_index('year', inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(data_yearly['selling_price'], marker='o', label='Average Selling Price')
plt.title('Average Car Selling Price by Year')
plt.xlabel('Year')
plt.ylabel('Selling Price (in lakhs)')
plt.legend()
plt.grid()
plt.show()

rolling_mean_2 = data_yearly['selling_price'].rolling(window=2).mean()
rolling_mean_3 = data_yearly['selling_price'].rolling(window=3).mean()

print("\nFirst 10 values of rolling mean (window=2):")
print(rolling_mean_2.head(10))
print("\nFirst 10 values of rolling mean (window=3):")
print(rolling_mean_3.head(10))

plt.figure(figsize=(12, 6))
plt.plot(data_yearly['selling_price'], label='Original Data', color='blue')
plt.plot(rolling_mean_2, label='Moving Average (window=2)', color='orange')
plt.plot(rolling_mean_3, label='Moving Average (window=3)', color='green')
plt.title('Moving Average of Average Selling Price')
plt.xlabel('Year')
plt.ylabel('Selling Price (in lakhs)')
plt.legend()
plt.grid()
plt.show()

scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_yearly['selling_price'].values.reshape(-1, 1)).flatten(),
    index=data_yearly.index
)
scaled_data = scaled_data + 1

split_index = int(len(scaled_data) * 0.8)
train_data = scaled_data[:split_index]
test_data = scaled_data[split_index:]

model = ExponentialSmoothing(train_data, trend='add', seasonal=None)
fit_model = model.fit()

test_pred = fit_model.forecast(len(test_data))


plt.figure(figsize=(12, 6))
train_data.plot(label='Train Data')
test_data.plot(label='Test Data')
test_pred.plot(label='Predictions')
plt.title('Exponential Smoothing Forecast (Cardekho Dataset)')
plt.xlabel('Year')
plt.ylabel('Scaled Selling Price')
plt.legend()
plt.grid()
plt.show()

rmse = np.sqrt(mean_squared_error(test_data, test_pred))
print(f"RMSE: {rmse:.4f}")

model_full = ExponentialSmoothing(scaled_data, trend='add', seasonal=None).fit()
future_steps = 3 
forecast = model_full.forecast(steps=future_steps)

plt.figure(figsize=(12, 6))
scaled_data.plot(label='Original Data')
forecast.plot(label='Forecast', style='--')
plt.title('Future Forecast of Selling Price (Next 3 Years)')
plt.xlabel('Year')
plt.ylabel('Scaled Selling Price')
plt.legend()
plt.grid()
plt.show()

forecast_original = scaler.inverse_transform((forecast - 1).values.reshape(-1, 1))
print("\nForecasted Selling Prices for Next 3 Years:")
for i, val in enumerate(forecast_original.flatten(), 1):
    print(f"Year +{i}: {val:.2f} Lakhs")


```
## OUTPUT:

### Original Data:

<img width="992" height="650" alt="image" src="https://github.com/user-attachments/assets/3ced125a-e44d-4a9b-a9b9-5316a496fa5a" />

### Moving Average:

<img width="731" height="675" alt="image" src="https://github.com/user-attachments/assets/d5ea4dfd-232f-458f-97ff-b38e3673c471" />

### Plot Transform Dataset

<img width="1292" height="685" alt="image" src="https://github.com/user-attachments/assets/96853068-0f78-46bf-b384-e861e97a9976" />

### Exponential Smoothing

<img width="1280" height="626" alt="image" src="https://github.com/user-attachments/assets/c272dfb2-2d80-4d83-8944-60175aecf699" />

## RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.

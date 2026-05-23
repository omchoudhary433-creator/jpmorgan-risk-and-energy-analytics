import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("C:\\Users\\user\\OneDrive\\Desktop\\Nat_Gas (1).csv")
print(df.head(4))
# Convert dates
df['Dates'] = pd.to_datetime(df['Dates'])
print(df.dtypes)
print(df.head(4))

# Sort values
df = df.sort_values('Dates')
print(df)

# Plot historical data
plt.figure(figsize=(12,6))

plt.plot(
    df['Dates'],
    df['Prices'],
    marker='o',
    label='Historical Prices'
)

plt.title("Natural Gas Prices")

plt.xlabel("Date")
plt.ylabel("Price")

plt.grid(True)

plt.legend()

plt.show()

# Convert dates to numerical format
df['Date_Ordinal'] = df['Dates'].map(pd.Timestamp.toordinal)
print(df['Date_Ordinal'])
# Interpolation function
interp_function = interp1d(
    df['Date_Ordinal'],
    df['Prices'],
    kind='linear',
    fill_value='extrapolate'
)

# Historical estimation function
def estimate_price(date):

    date = pd.to_datetime(date)

    date_ordinal = date.toordinal()

    predicted_price = interp_function(date_ordinal)

    return float(predicted_price)
#test historical prediction

print(estimate_price("2022-06-01"))



# Test interpolation
print("Estimated Historical Price:")
print(estimate_price("2022-06-15"))

# Prepare data for forecasting
X = df[['Date_Ordinal']]
y = df['Prices']

# Train Linear Regression model
model = LinearRegression()

model.fit(X, y)

# Future dates
future_dates = pd.date_range(
    start=df['Dates'].max() + pd.Timedelta(days=1),
    periods=365,
    freq='D'
)
print(future_dates)

# Convert future dates
future_ordinals = future_dates.map(pd.Timestamp.toordinal)
print(future_ordinals)

# Predict future prices
future_prices = model.predict(
    np.array(future_ordinals).reshape(-1,1)
)

# Plot forecast
plt.figure(figsize=(12,6))

# Historical data
plt.plot(
    df['Dates'],
    df['Prices'],
    marker='o',
    label='Historical Prices'
)

# Forecast
plt.plot(
    future_dates,
    future_prices,
    marker='x',
    linestyle='--',
    label='Forecasted Prices'
)

plt.title("Natural Gas Price Forecast")

plt.xlabel("Date")
plt.ylabel("Price")

plt.grid(True)

plt.legend()

plt.show()

# Final combined function
last_date = df['Dates'].max()

def get_price_estimate(input_date):

    input_date = pd.to_datetime(input_date)

    if input_date <= last_date:
          return float(
            interp_function(input_date.toordinal())
        )

    else:

        return float(
            model.predict(
                [[input_date.toordinal()]]
            )[0]
        )

# Final testing
print("\nFinal Price Estimates:")

print("2023-07-15 : ",
      get_price_estimate("2023-07-15"))

print("2025-08-31 : ",
      get_price_estimate("2025-08-31"))
#add moving average

df['Moving_Avg'] = df['Prices'].rolling(window=3).mean()

print(df[['Dates', 'Prices', 'Moving_Avg']])
      



#export results to CSV file
results_df = pd.DataFrame({
    "Date": future_dates,
    "Forecasted_Price": future_prices
})
results_df.to_csv("natural_gas_forecast.csv", index=False)


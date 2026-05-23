import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

# LOAD DATA

df = pd.read_csv(r"C:\Users\user\OneDrive\Desktop\Nat_Gas (1).csv")
print(df.head(4))

# Convert Dates column to datetime
df['Dates'] = pd.to_datetime(df['Dates'])

# Sort data by date
df = df.sort_values('Dates')

# VISUALIZE HISTORICAL DATA

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

# CONVERT DATES TO NUMBERS

df['Date_Ordinal'] = df['Dates'].map(
    pd.Timestamp.toordinal
)

# INTERPOLATION FUNCTION

interp_function = interp1d(
    df['Date_Ordinal'],
    df['Prices'],
    kind='linear',
    fill_value='extrapolate'
)

# HISTORICAL PRICE ESTIMATION

def estimate_price(date):

    date = pd.to_datetime(date)

    date_ordinal = date.toordinal()

    predicted_price = interp_function(
        date_ordinal
    )

    return float(predicted_price)

print("Historical Price Estimate:")
print(
    estimate_price("2022-06-15")
)

# TRAIN FORECAST MODEL

X = df[['Date_Ordinal']]

y = df['Prices']

model = LinearRegression()

model.fit(X, y)

# GENERATE FUTURE DATES

future_dates = []

last_date = df['Dates'].max()

for i in range(1, 13):

    next_month = (
        last_date
        + pd.DateOffset(months=i)
    )

    future_dates.append(next_month)

# PREDICT FUTURE PRICES

future_ordinals = [
    date.toordinal()
    for date in future_dates
]

future_prices = model.predict(
    np.array(
        future_ordinals
    ).reshape(-1,1)
)

# PLOT FORECAST

plt.figure(figsize=(12,6))

# Historical prices
plt.plot(
    df['Dates'],
    df['Prices'],
    marker='o',
    label='Historical Prices'
)

# Forecasted prices
plt.plot(
    future_dates,
    future_prices,
    marker='x',
    linestyle='--',
    label='Forecasted Prices'
)

plt.title(
    "Natural Gas Price Forecast"
)

plt.xlabel("Date")

plt.ylabel("Price")

plt.grid(True)

plt.legend()

plt.show()

# FINAL PRICE ESTIMATION FUNCTION

last_historical_date = df['Dates'].max()

def get_price_estimate(input_date):

    input_date = pd.to_datetime(
        input_date
    )

    # Historical estimation
    if input_date <= last_historical_date:

        return float(
            interp_function(
                input_date.toordinal()
            )
        )

    # Future prediction
    else:

        return float(
            model.predict(
                [[input_date.toordinal()]]
            )[0]
        )

# TEST FINAL FUNCTION

print("\nFinal Price Estimates:")

print(
    "2023-07-15 : ",
    get_price_estimate("2023-07-15")
)

print(
    "2025-08-31 : ",
    get_price_estimate("2025-08-31")
)

# STORAGE CONTRACT PRICING MODEL

def price_gas_contract(
    injection_dates,
    withdrawal_dates,
    volume,
    storage_cost_per_month,
    max_volume
):

    # Check storage limit
    if volume > max_volume:
        raise ValueError(
            "Volume exceeds maximum storage capacity."
        )
    total_profit = 0
    for inj_date, wth_date in zip(
        injection_dates,
        withdrawal_dates
    ):

          # Buy price
        buy_price = get_price_estimate(
            inj_date
        )

        # Sell price
        sell_price = get_price_estimate(
            wth_date
        )

        # Gross profit
        profit = (
            sell_price - buy_price
        ) * volume

        # Convert dates
        inj = pd.to_datetime(
            inj_date
        )

        wd = pd.to_datetime(
            wth_date
        )

        # Storage duration
        months_stored = (
            (wd.year - inj.year) * 12
            + (wd.month - inj.month)
        )

        # Storage cost
        storage_cost = (
            months_stored
            * storage_cost_per_month
        )

        # Net profit
        total_profit += (
            profit - storage_cost
        )

    return total_profit

# TEST STORAGE CONTRACT

injection_dates = [
    "2023-01-01",
    "2023-03-01"
]

withdrawal_dates = [
    "2023-06-01",
    "2023-09-01"
]

volume = 1000

storage_cost_per_month = 500

max_volume = 5000

contract_value = price_gas_contract(
    injection_dates,
    withdrawal_dates,
    volume,
    storage_cost_per_month,
    max_volume
)

# FINAL OUTPUT

print("\nStorage Contract Value:")

print(contract_value)

print("\nAnalysis completed successfully.")

#export in csv format

df.to_csv("storage_contract_results.csv", index=False)


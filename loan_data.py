import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# LOAD DATA

df = pd.read_csv(r"C:\Users\user\OneDrive\Desktop\Task 3 and 4_Loan_Data.csv")
   
# VIEW DATA

print("First 5 Rows:\n")

print(df.head())

print("\nColumn Names:\n")

print(df.columns)

# HANDLE MISSING VALUES

df = df.dropna()

print(df.columns)

# Replace column names below
# according to your dataset

X = df[[
    'income',
    'total_debt_outstanding',
'fico_score'
]]

# Target column
y = df['default']

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TRAIN MODEL

model = LogisticRegression()

model.fit(X_train, y_train)

# PREDICTIONS

y_pred = model.predict(X_test)

# MODEL EVALUATION

print("\nAccuracy:\n")

print(
    accuracy_score(y_test, y_pred)
)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(y_test, y_pred)
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# DEFAULT PROBABILITY FUNCTION

def predict_default_probability(
    income,
    total_debt_outstanding,
    fico_score
):

    input_data = pd.DataFrame({
        'income': [income],
        'total_debt_outstanding': [
            total_debt_outstanding
        ],
        'fico_score': [fico_score]
    })

    probability = model.predict_proba(
        input_data
    )[0][1]

    return probability

# EXPECTED LOSS FUNCTION

def calculate_expected_loss(
    loan_amount,
    income,
    total_loans_outstanding,
    fico_score,
    recovery_rate=0.10
):

    # Probability of Default
    pd_value = predict_default_probability(
        income,
        total_loans_outstanding,
        fico_score
    )

    # Loss Given Default
    lgd = 1 - recovery_rate

    # Expected Loss
    expected_loss = (
        pd_value
        * lgd
        * loan_amount
    )

    return expected_loss

# TEST THE FUNCTIONS

loan_amount = 50000

income = 75000

total_debt_outstanding = 20000

fico_score = 700

# Predict probability of default

pd_result = predict_default_probability(
    income,
    total_debt_outstanding,
    fico_score
)

print("\nProbability of Default:\n")

print(pd_result)

# Calculate expected loss

expected_loss_result = calculate_expected_loss(
    loan_amount,
    income,
    total_debt_outstanding,
    fico_score
)

print("\nExpected Loss:\n")

print(expected_loss_result)

# VISUALIZATION

plt.figure(figsize=(8,5))

plt.hist(
    df['fico_score'],
    bins=20
)

plt.title("FICO Score Distribution")

plt.xlabel("FICO Score")

plt.ylabel("Frequency")

plt.grid(True)

plt.show()

#export file into csv

df.to_csv( "processed_loan_data.csv",
    index=False
)
  #final print statement

print("\nCredit Risk Analysis Complete!")



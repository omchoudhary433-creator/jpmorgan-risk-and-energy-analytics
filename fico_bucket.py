import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# LOAD CUSTOMER LOAN DATA

df = pd.read_csv(r"C:\Users\user\OneDrive\Desktop\Task 3 and 4_Loan_Data.csv")

print(df.head())

# CHECK COLUMN NAMES

print("\nColumns in Dataset:")

print(df.columns)

# SELECT FICO SCORE COLUMN

# Replace column name if needed
fico_scores = df['fico_score']

# VISUALIZE FICO SCORES

plt.figure(figsize=(10,5))

plt.hist(
    fico_scores,
    bins=30
)

plt.title("FICO Score Distribution")

plt.xlabel("FICO Score")

plt.ylabel("Frequency")

plt.grid(True)

plt.show()

# CREATE BUCKETS USING KMEANS

# Number of rating buckets
num_buckets = 5

# Reshape data for model
X = np.array(fico_scores).reshape(-1,1)

# Train KMeans
kmeans = KMeans(
    n_clusters=num_buckets,
    random_state=42
)

kmeans.fit(X)

# Assign bucket labels
df['Bucket'] = kmeans.labels_

# SORT BUCKETS PROPERLY

# Get cluster centers
centers = kmeans.cluster_centers_.flatten()

# Sort centers
sorted_centers = np.sort(centers)

print("\nBucket Centers:")

print(sorted_centers)

# CREATE RATING MAP
# Lower rating = Better credit score

bucket_map = {}

for i, center in enumerate(sorted_centers):

    bucket_map[center] = i + 1

print("\nRating Map:")

print(bucket_map)

# ASSIGN RATINGS

def assign_rating(score):

    distances = abs(
        sorted_centers - score
    )

    closest_bucket = np.argmin(
        distances
    )

    rating = closest_bucket + 1

    return rating

# Apply ratings
df['Rating'] = df[
    'fico_score'
].apply(assign_rating)

# DISPLAY RESULTS

print("\nSample Results:")

print(
    df[
        ['fico_score', 'Rating']
    ].head(20)
)

# VISUALIZE BUCKETS

plt.figure(figsize=(10,5))

plt.scatter(
    df['fico_score'],
    df['Rating']
)

plt.title("FICO Score Buckets")

plt.xlabel("FICO Score")

plt.ylabel("Rating Bucket")

plt.grid(True)

plt.show()

# TEST WITH SAMPLE SCORES

test_scores = [
    450,
    550,
    650,
    720,
    800
]

print("\nTest Predictions:")

for score in test_scores:

    rating = assign_rating(score)

    print(
        f"FICO Score: {score}"
        f" --> Rating: {rating}"
    )
    #export file into csv
df.to_csv(r"C:\Users\user\OneDrive\Desktop\fico_buckets.csv", index=False)


# FINAL MESSAGE

print("\nBucketing completed successfully.")



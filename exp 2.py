import math
from collections import Counter

# -----------------------------------
# Step 1: Hardcoded Retail Dataset
# -----------------------------------
# Attributes: Weather, Promotion, Weekend, Demand
dataset = [
    ["Sunny", "Yes", "No", "High"],
    ["Sunny", "No", "No", "Medium"],
    ["Overcast", "Yes", "No", "High"],
    ["Rainy", "Yes", "Yes", "High"],
    ["Rainy", "No", "Yes", "Low"],
    ["Rainy", "No", "No", "Low"],
    ["Overcast", "No", "Yes", "Medium"],
    ["Sunny", "Yes", "Yes", "High"],
    ["Sunny", "No", "Yes", "Medium"],
    ["Rainy", "Yes", "No", "Medium"]
]

# -----------------------------------
# Step 2: Encode Categorical Data
# -----------------------------------
# Manual encoding dictionary
encoding = {
    "Weather": {"Sunny": 0, "Overcast": 1, "Rainy": 2},
    "Promotion": {"Yes": 1, "No": 0},
    "Weekend": {"Yes": 1, "No": 0}
}

def encode_record(record):
    return [
        encoding["Weather"][record[0]],
        encoding["Promotion"][record[1]],
        encoding["Weekend"][record[2]]
    ]

# Encode dataset
encoded_data = []
labels = []

for row in dataset:
    encoded_data.append(en_

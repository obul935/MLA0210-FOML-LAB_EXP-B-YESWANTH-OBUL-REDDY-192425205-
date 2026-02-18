import math
import random

# --------------------------------------
# Step 1: Hardcoded Retail Dataset
# --------------------------------------
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

# --------------------------------------
# Step 2: Encode Categorical Data
# --------------------------------------
encoding = {
    "Weather": {"Sunny": 0, "Overcast": 1, "Rainy": 2},
    "Promotion": {"Yes": 1, "No": 0},
    "Weekend": {"Yes": 1, "No": 0}
}

def encode_features(row):
    return [
        encoding["Weather"][row[0]],
        encoding["Promotion"][row[1]],
        encoding["Weekend"][row[2]]
    ]

# Encode dataset
X = []
y = []

for row in dataset:
    X.append(encode_features(row))
    # Binary Target: High = 1, Others = 0
    y.append(1 if row[3] == "High" else 0)

# --------------------------------------
# Step 3: Sigmoid Function
# --------------------------------------
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# --------------------------------------
# Step 4: Initialize Parameters
# --------------------------------------
weights = [0.0] * 3
bias = 0.0
learning_rate = 0.1
epochs = 1000

# --------------------------------------
# Step 5: Train using Gradient Descent
# --------------------------------------
for _ in range(epochs):
    for i in range(len(X)):
        linear_model = sum(weights[j] * X[i][j] for j in range(3)) + bias
        prediction = sigmoid(linear_model)
        
        error = prediction - y[i]
        
        # Update weights and bias
        for j in range(3):
            weights[j] -= learning_rate * error * X[i][j]
        bias -= learning_rate * error

# --------------------------------------
# Step 6: Prediction Function
# --------------------------------------
def predict(new_sample):
    encoded = encode_features(new_sample)
    linear_model = sum(weights[j] * encoded[j] for j in range(3)) + bias
    probability = sigmoid(linear_model)
    
    classification = "High" if probability >= 0.5 else "Low/Medium"
    
    return probability, classification

# --------------------------------------
# Step 7: Take User Input
# --------------------------------------
print("Enter New Retail Scenario:")

weather = input("Weather (Sunny/Overcast/Rainy): ")
promotion = input("Promotion (Yes/No): ")
weekend = input("Weekend (Yes/No): ")

new_sample = [weather, promotion, weekend]

prob, result = predict(new_sample)

print("\nLearned Weights:", weights)
print("Learned Bias:", bias)
print("\nPredicted Probability of High Demand:", round(prob, 4))
print("Final Predicted Demand Level:", result)

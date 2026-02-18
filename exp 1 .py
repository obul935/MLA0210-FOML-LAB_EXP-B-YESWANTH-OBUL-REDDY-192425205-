import math
from collections import Counter

# -----------------------------
# Sample Retail Demand Dataset
# -----------------------------
# Attributes: Weather, Promotion, Weekend, Demand
dataset = [
    {"Weather": "Sunny", "Promotion": "Yes", "Weekend": "No", "Demand": "High"},
    {"Weather": "Sunny", "Promotion": "No", "Weekend": "No", "Demand": "Medium"},
    {"Weather": "Overcast", "Promotion": "Yes", "Weekend": "No", "Demand": "High"},
    {"Weather": "Rainy", "Promotion": "Yes", "Weekend": "Yes", "Demand": "High"},
    {"Weather": "Rainy", "Promotion": "No", "Weekend": "Yes", "Demand": "Low"},
    {"Weather": "Rainy", "Promotion": "No", "Weekend": "No", "Demand": "Low"},
    {"Weather": "Overcast", "Promotion": "No", "Weekend": "Yes", "Demand": "Medium"},
    {"Weather": "Sunny", "Promotion": "Yes", "Weekend": "Yes", "Demand": "High"},
    {"Weather": "Sunny", "Promotion": "No", "Weekend": "Yes", "Demand": "Medium"},
    {"Weather": "Rainy", "Promotion": "Yes", "Weekend": "No", "Demand": "Medium"}
]

# -----------------------------
# Function to calculate entropy
# -----------------------------
def entropy(data):
    labels = [row["Demand"] for row in data]
    label_count = Counter(labels)
    total = len(data)
    
    ent = 0
    for count in label_count.values():
        probability = count / total
        ent -= probability * math.log2(probability)
    
    return ent

# ------------------------------------
# Function to calculate Information Gain
# ------------------------------------
def information_gain(data, attribute):
    total_entropy = entropy(data)
    values = set(row[attribute] for row in data)
    weighted_entropy = 0
    
    for value in values:
        subset = [row for row in data if row[attribute] == value]
        probability = len(subset) / len(data)
        weighted_entropy += probability * entropy(subset)
    
    return total_entropy - weighted_entropy

# ------------------------------------
# ID3 Algorithm Implementation
# ------------------------------------
def id3(data, attributes):
    labels = [row["Demand"] for row in data]
    
    # If all examples have same class
    if len(set(labels)) == 1:
        return labels[0]
    
    # If no attributes left
    if not attributes:
        return Counter(labels).most_common(1)[0][0]
    
    # Choose attribute with highest information gain
    gains = {attr: information_gain(data, attr) for attr in attributes}
    best_attr = max(gains, key=gains.get)
    
    tree = {best_attr: {}}
    values = set(row[best_attr] for row in data)
    
    for value in values:
        subset = [row for row in data if row[best_attr] == value]
        if not subset:
            tree[best_attr][value] = Counter(labels).most_common(1)[0][0]
        else:
            remaining_attrs = [attr for attr in attributes if attr != best_attr]
            tree[best_attr][value] = id3(subset, remaining_attrs)
    
    return tree

# ------------------------------------
# Prediction Function
# ------------------------------------
def predict(tree, sample):
    if isinstance(tree, str):
        return tree
    
    root = next(iter(tree))
    value = sample[root]
    
    if value in tree[root]:
        return predict(tree[root][value], sample)
    else:
        return "Unknown"

# ------------------------------------
# Build Decision Tree
# ------------------------------------
attributes = ["Weather", "Promotion", "Weekend"]
decision_tree = id3(dataset, attributes)

print("\nDecision Tree:")
print(decision_tree)

# ------------------------------------
# Predict for New Retail Scenario
# ------------------------------------
print("\nEnter New Retail Scenario Details:")

weather = input("Weather (Sunny/Overcast/Rainy): ")
promotion = input("Promotion (Yes/No): ")
weekend = input("Weekend (Yes/No): ")

new_sample = {
    "Weather": weather,
    "Promotion": promotion,
    "Weekend": weekend
}

prediction = predict(decision_tree, new_sample)

print("\nPredicted Demand Level:", prediction)

from collections import Counter

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
# Step 2: Calculate Prior Probabilities
# --------------------------------------
def calculate_prior():
    labels = [row[3] for row in dataset]
    total = len(labels)
    label_count = Counter(labels)
    
    priors = {}
    print("\nPrior Probabilities:")
    for label in label_count:
        priors[label] = label_count[label] / total
        print(f"P({label}) = {label_count[label]}/{total} = {priors[label]:.4f}")
    
    return priors

# --------------------------------------
# Step 3: Calculate Likelihood
# --------------------------------------
def calculate_likelihood(attribute_index, value, label):
    total_label_count = sum(1 for row in dataset if row[3] == label)
    match_count = sum(1 for row in dataset if row[attribute_index] == value and row[3] == label)
    
    likelihood = match_count / total_label_count if total_label_count != 0 else 0
    
    print(f"P({value} | {label}) = {match_count}/{total_label_count} = {likelihood:.4f}")
    return likelihood

# --------------------------------------
# Step 4: Naïve Bayes Prediction
# --------------------------------------
def naive_bayes_predict(new_sample):
    priors = calculate_prior()
    probabilities = {}
    
    print("\nCalculating Posterior Probabilities:\n")
    
    for label in priors:
        print(f"For Class = {label}")
        prob = priors[label]
        
        for i in range(3):  # 3 attributes
            likelihood = calculate_likelihood(i, new_sample[i], label)
            prob *= likelihood
        
        probabilities[label] = prob
        print(f"Posterior Probability for {label} = {prob:.6f}\n")
    
    # Choose class with highest probability
    predicted_class = max(probabilities, key=probabilities.get)
    return predicted_class

# --------------------------------------
# Step 5: User Input
# --------------------------------------
print("Enter New Retail Scenario:")

weather = input("Weather (Sunny/Overcast/Rainy): ")
promotion = input("Promotion (Yes/No): ")
weekend = input("Weekend (Yes/No): ")

new_sample = [weather, promotion, weekend]

prediction = naive_bayes_predict(new_sample)

print("Final Predicted Demand Level:", prediction)

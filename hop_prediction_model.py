# Import libraries:
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

# Read data:
read_data = pd.read_csv('data.csv')

# Convert hops to list:
read_data['Hops'] = read_data['Hops'].apply(lambda x: [hop.strip() for hop in x.split(',')])

# Log progress:
print("### Initializing model data...\n")
print(read_data)

# Prepare features:
features = read_data[["Style", "ABV", "Rating"]]

# Convert style feature into one-hot encoding:
converted_features = pd.get_dummies(features, columns=["Style"])

# Prepare MultiLabelBinarizer:
multi_label_binarizer = MultiLabelBinarizer()

# Convert hops into a binary target matrix using MultiLabelBinarizer:
target = multi_label_binarizer.fit_transform(read_data["Hops"])

# Train One-vs-Rest multi-label classification model:
model_full = OneVsRestClassifier(LogisticRegression())
model_full.fit(converted_features, target)

# Analyze hop combinations:
def analyze_hop_combinations(data, top_n=10):
    
    # Prepare hop combinations:
    hop_combinations = []

    # Loop through hops:
    for hops in data['Hops']:
        
        # Sort hops to ensure consistent ordering:
        sorted_hops = sorted(hops)

        # Append sorted hops to list:
        hop_combinations.append(tuple(sorted_hops))
    
    # Count combinations:
    combination_counts = Counter(hop_combinations)
    
    # Get top N combinations:
    top_combinations = combination_counts.most_common(top_n)
    
    # Log progress:
    print("\n### Most common hop combinations:\n")

    # Loop through top combinations:
    for i, (combo, count) in enumerate(top_combinations, 1):
        print(f"{i}. {', '.join(combo)} (used in {count} beers)")

# Analyze highly-rated hop combinations:
def analyze_highly_rated_combinations(data, rating_threshold=4.0, top_n=10):
    
    # Filter highly-rated beers:
    high_rated_beers = data[data['Rating'] >= rating_threshold]
    
    # Prepare hop combinations:
    hop_combinations = []

    # Loop through hops:
    for hops in high_rated_beers['Hops']:
        
        # Sort hops:
        sorted_hops = sorted(hops)

        # Append sorted hops:
        hop_combinations.append(tuple(sorted_hops))
    
    # Count combinations
    combination_counts = Counter(hop_combinations)
    
    # Get top N combinations:
    top_combinations = combination_counts.most_common(top_n)
    
    # Log progress:
    print(f"\n### Top hop combinations in highly-rated beers (Rating >= {rating_threshold}):\n")

    # Loop through top combinations:
    for i, (combo, count) in enumerate(top_combinations, 1):
        avg_rating = high_rated_beers[high_rated_beers['Hops'].apply(lambda x: sorted(x) == list(combo))]['Rating'].mean()
        print(f"{i}. {', '.join(combo)} (used in {count} beers, average rating: {avg_rating:.2f})")

# Get user input:
def get_user_input():

    # Log progress:
    print("\n### Ready to analyze beer data!\n")
    print("1. Predict hops for a new beer")
    print("2. Show most common hop combinations")
    print("3. Show hop combinations in highly-rated beers")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    # Predict hops:
    if choice == "1":

        # Get user input:
        style = input("Beer style:\t\t").strip()
        abv = float(input("ABV:\t\t\t").strip())
        minimum_rating = float(input("Minimum rating:\t\t").strip())
        confidence_level = float(input("Confidence level:\t").strip())

        # Return user input:
        return choice, (style, abv, minimum_rating, confidence_level)
    
    # Analyze hop combinations:
    elif choice in ["2", "3"]:
        return choice, None
    
    # Exit:
    else:
        return "4", None

# Predict hops:
def predict_hops(style, abv, minimum_rating, confidence_level):

    # Create a range of ratings slightly above the minimum:
    ratings_range = np.arange(minimum_rating, minimum_rating + 0.2, 0.01)

    # Prepare predictions list:
    predictions_list = []

    # Loop through ratings range:
    for r in ratings_range:

        # Prepare data frame:
        data_frame = pd.DataFrame({
            "Style": [style],
            "ABV": [abv],
            "Rating": [r]
        })

        # Convert style to one-hot encoding:
        converted_data_frame = pd.get_dummies(data_frame, columns=["Style"])

        # Reindex data frame:
        converted_data_frame = converted_data_frame.reindex(columns=converted_features.columns, fill_value=0)

        # Predict probabilities:
        predicted_probabilities = model_full.predict_proba(converted_data_frame)

        # Append probabilities to list:
        predictions_list.append(predicted_probabilities[0])

    # Calculate average probabilities:
    average_probabilities = np.mean(predictions_list, axis=0)

    # Predict labels:
    predicted_labels = (average_probabilities >= confidence_level).astype(int)

    # Return inverse transform of predicted labels:
    return multi_label_binarizer.inverse_transform(np.array([predicted_labels]))

# Loop main interaction:
while True:

    # Get user input:
    choice, params = get_user_input()
    
    # Predict hops:
    if choice == "1":

        # Prepare user input:
        style, abv, minimum_rating, confidence_level = params

        # Predict hops:
        predicted_hops = predict_hops(style, abv, minimum_rating, confidence_level)

        # Log progress:
        print(f"\nPredicted hops for {style} (ABV {abv}%) with rating equal to or above {minimum_rating}:\n")
        print(predicted_hops[0])

    # Analyze hop combinations:
    elif choice == "2":

        # Analyze hop combinations:
        analyze_hop_combinations(read_data)

    # Analyze highly-rated hop combinations:
    elif choice == "3":

        # Analyze highly-rated hop combinations:
        analyze_highly_rated_combinations(read_data)

    # Exit:
    elif choice == "4":
        break
    
    # Ask user to try again:
    if choice != "4" and input("\nTry another analysis? (y/n): ").lower() != 'y':
        break


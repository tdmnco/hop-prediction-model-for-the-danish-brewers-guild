# Import libraries:
import pandas as pd
import numpy as np

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

# Log progress:
print("\n### Initializing hop varieties...\n")
print(multi_label_binarizer.classes_)

# Train model:
model_full = OneVsRestClassifier(LogisticRegression())
model_full.fit(converted_features, target)

# Get user input:
def get_user_input():

    # Log progress:
    print("\n### Ready to predict hops for your beer!\n")

    # Get user input:
    style = input("Beer style: ").strip()
    abv = float(input("ABV: ").strip())
    minimum_rating = float(input("Minimum rating: ").strip())
    confidence_level = float(input("Confidence level: ").strip())

    return style, abv, minimum_rating, confidence_level

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

# Main interaction loop:
while True:

    # Get user input:
    style, abv, minimum_rating, confidence_level = get_user_input()

    # Predict hops:
    predicted_hops = predict_hops(style, abv, minimum_rating, confidence_level)

    # Log progress:
    print(f"\nRecommended hops for {style} (ABV {abv}%) with rating above {minimum_rating}:")
    print(predicted_hops[0])

    # Ask user to try again:
    if input("\nTry another prediction? (y/n): ").lower() != 'y':
        break


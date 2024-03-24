import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load the cleaned data
data = pd.read_csv('cleaned_data.csv')

# Set 'GenHealth' is your target variable
X = data.drop('GenHealth', axis=1)
y = data['GenHealth']

# Ensure 'GenHealth' is encoded as categorical
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Initialize the models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)

# Train the models
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test)
gb_predictions = gb_model.predict(X_test)

# Evaluate performance
print("Random Forest Performance:")
print(f"Accuracy: {accuracy_score(y_test, rf_predictions):.4f}")
print(f"Precision (Weighted): {precision_score(y_test, rf_predictions, average='weighted'):.4f}")
print(f"Recall (Weighted): {recall_score(y_test, rf_predictions, average='weighted'):.4f}")
print(f"F1 Score (Weighted): {f1_score(y_test, rf_predictions, average='weighted'):.4f}")

print("\nGradient Boosting Performance:")
print(f"Accuracy: {accuracy_score(y_test, gb_predictions):.4f}")
print(f"Precision (Weighted): {precision_score(y_test, gb_predictions, average='weighted'):.4f}")
print(f"Recall (Weighted): {recall_score(y_test, gb_predictions, average='weighted'):.4f}")
print(f"F1 Score (Weighted): {f1_score(y_test, gb_predictions, average='weighted'):.4f}")


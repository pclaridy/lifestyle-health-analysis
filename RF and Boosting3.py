import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint, uniform

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

# Expanded Random Forest parameter grid
rf_param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 11),
    'max_features': ['sqrt', 'log2', None]
}

# Expanded Gradient Boosting parameter grid
gb_param_dist = {
    'n_estimators': randint(100, 500),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(3, 10),
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 11),
    'max_features': ['sqrt', 'log2', None]
}

# Random Forest Randomized Search
rf_random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=rf_param_dist,
    n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1, scoring='accuracy'
)
rf_random_search.fit(X_train, y_train)
best_rf_model = rf_random_search.best_estimator_

# Gradient Boosting Randomized Search
gb_random_search = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_distributions=gb_param_dist,
    n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1, scoring='accuracy'
)
gb_random_search.fit(X_train, y_train)
best_gb_model = gb_random_search.best_estimator_

# Evaluate the tuned models
rf_predictions = best_rf_model.predict(X_test)
gb_predictions = best_gb_model.predict(X_test)

print("Tuned Random Forest Performance:")
print(f"Accuracy: {accuracy_score(y_test, rf_predictions):.4f}")
print(f"Precision (Weighted): {precision_score(y_test, rf_predictions, average='weighted'):.4f}")
print(f"Recall (Weighted): {recall_score(y_test, rf_predictions, average='weighted'):.4f}")
print(f"F1 Score (Weighted): {f1_score(y_test, rf_predictions, average='weighted'):.4f}")

print("\nTuned Gradient Boosting Performance:")
print(f"Accuracy: {accuracy_score(y_test, gb_predictions):.4f}")
print(f"Precision (Weighted): {precision_score(y_test, gb_predictions, average='weighted'):.4f}")
print(f"Recall (Weighted): {recall_score(y_test, gb_predictions, average='weighted'):.4f}")
print(f"F1 Score (Weighted): {f1_score(y_test, gb_predictions, average='weighted'):.4f}")

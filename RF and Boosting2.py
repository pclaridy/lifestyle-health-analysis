import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Hyperparameter tuning for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf = RandomForestClassifier(random_state=42)
rf_grid_search = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
rf_grid_search.fit(X_train, y_train)
best_rf_model = rf_grid_search.best_estimator_

# Hyperparameter tuning for Gradient Boosting
gb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
gb = GradientBoostingClassifier(random_state=42)
gb_grid_search = GridSearchCV(estimator=gb, param_grid=gb_param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
gb_grid_search.fit(X_train, y_train)
best_gb_model = gb_grid_search.best_estimator_

# Make predictions with the best models
rf_predictions = best_rf_model.predict(X_test)
gb_predictions  = best_gb_model.predict(X_test)

# Evaluate performance
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

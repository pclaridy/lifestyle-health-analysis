import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# Load the cleaned data
data = pd.read_csv('cleaned_data.csv')

# Set 'BMI' is your target variable
X = data.drop('BMI', axis=1)
y = data['BMI']

# Ensure 'BMI' is encoded as categorical
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize baseline models
knn = KNeighborsClassifier()
logreg = LogisticRegression()
decision_tree = DecisionTreeClassifier()

# Train baseline models
knn.fit(X_train_scaled, y_train)
logreg.fit(X_train_scaled, y_train)
decision_tree.fit(X_train, y_train) # No scaling needed for decision trees

# Evaluate baseline models using cross-validation
knn_cv_accuracy = cross_val_score(knn, scaler.transform(X), y_encoded, cv=5).mean()
logreg_cv_accuracy = cross_val_score(logreg, scaler.transform(X), y_encoded, cv=5).mean()
decision_tree_cv_accuracy = cross_val_score(decision_tree, X, y_encoded, cv=5).mean()

# Print cross-validation results
print("Baseline Model Performance (Cross-Validation):")
print(f"KNN Accuracy: {knn_cv_accuracy}")
print(f"Logistic Regression Accuracy: {logreg_cv_accuracy}")
print(f"Decision Tree Accuracy: {decision_tree_cv_accuracy}")

# Print test accuracy for comparison, using scaled data for KNN and Logistic Regression
print("\nBaseline Model Performance (Test Set):")
print(f"KNN Accuracy: {knn.score(X_test_scaled, y_test)}")
print(f"Logistic Regression Accuracy: {logreg.score(X_test_scaled, y_test)}")
print(f"Decision Tree Accuracy: {decision_tree.score(X_test, y_test)}")




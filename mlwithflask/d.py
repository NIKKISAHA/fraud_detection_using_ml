# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load your data (replace with your actual dataset path)
data = pd.read_csv('fraud_detection.csv')

# Assume the target variable is 'is_fraud' and features are 'Transaction_Amount','Amount_paid','Vehicle_Speed'
X = data[['Transaction_Amount','Amount_paid','Vehicle_Speed']]
data['Fraud_indicator'].replace({'Fraud':1, 'Not Fraud':-1},inplace=True)
y = data['Fraud_indicator']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions)}")

# Save the model
joblib_file = "random_forest_model.pkl"
joblib.dump(model, joblib_file)
print(f"Model saved as {joblib_file}")
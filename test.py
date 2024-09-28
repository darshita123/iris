import json
import pickle
from sklearn.metrics import accuracy_score

# Load the model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the test data
with open("data.json", "r") as data_file:
    test_data = json.load(data_file)

X_test = test_data["data"]
y_expected = test_data["expected"]

# Make predictions using the loaded model
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_expected, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Assert that the model passes a threshold accuracy
assert accuracy > 0.9, "Model accuracy is below acceptable threshold!"

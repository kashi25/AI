# model_builder.py
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
import joblib  # for saving model

# Load dataset
digits = load_digits()
X = digits.data
y = digits.target

# Train logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X, y)

# Save model
joblib.dump(model, "digit_classifier_model.pkl")

print("✅ Model trained and saved as digit_classifier_model.pkl")

import joblib
import numpy as np

# Load the trained model and the scaler from Joblib files
svm = joblib.load('svm_model.joblib')
scaler = joblib.load('scaler.joblib')

# New data to predict
new_data = [
    [0.062357219,0.0278,0.019132322,0.138319637,4.80078991,25.83713617]  # Example input 2
]

# Convert new data to a NumPy array
new_data = np.array(new_data, dtype=float)

# Standardize the new data using the previously fitted scaler
new_data = scaler.transform(new_data)

# Make predictions using the loaded model
predictions = svm.predict(new_data)

# Print the predictions
print(f'Predictions: {predictions}')

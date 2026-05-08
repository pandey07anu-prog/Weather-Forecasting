import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv("weather.csv")

# Features and target
X = data[['Humidity', 'WindSpeed', 'Pressure']]
y = data['Temperature']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained successfully!")
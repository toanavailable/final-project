import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np

# Load and preprocess the dataset
file_path = r'C:\Users\OS\Desktop\Workspace\FPT\Housing.csv'
housing_data = pd.read_csv(file_path)

label_encoders = {}
categorical_columns = housing_data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    housing_data[col] = label_encoders[col].fit_transform(housing_data[col])

# Splitting the dataset
X = housing_data.drop('price', axis=1)
y = housing_data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Function to make predictions based on user input
def predict_house_price(model, scaler):
    inputs = []
    for col in X.columns:
        if col in categorical_columns:
            value = input(f"Enter {col} (Available options: {housing_data[col].unique()}): ")
            # Encoding the categorical input
            value = label_encoders[col].transform([value])[0]
        else:
            value = float(input(f"Enter {col}: "))
        inputs.append(value)

    # Scaling the input
    inputs_scaled = scaler.transform([inputs])
    # Making prediction
    predicted_price = model.predict(inputs_scaled)
    return predicted_price[0]

# User input and prediction
predicted_price = predict_house_price(linear_model, scaler)
print(f"The predicted house price is: {predicted_price}")

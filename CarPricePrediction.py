import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('car_features_and_msrm.csv')

# Display the first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Get dataset summary
print(df.describe())

# Fill missing values (if any)
df = df.fillna(method='ffill')

# Encode categorical variables
label_encoder = LabelEncoder()
df['Make'] = label_encoder.fit_transform(df['Make'])
df['Model'] = label_encoder.fit_transform(df['Model'])
df['Transmission Type'] = label_encoder.fit_transform(df['Transmission Type'])
df['Driven_Wheels'] = label_encoder.fit_transform(df['Driven_Wheels'])
df['Market Category'] = label_encoder.fit_transform(df['Market Category'].astype(str))
df['Vehicle Size'] = label_encoder.fit_transform(df['Vehicle Size'])
df['Vehicle Style'] = label_encoder.fit_transform(df['Vehicle Style'])

# Define the features and target
X = df[['Year', 'Make', 'Model', 'Engine HP', 'Engine Cylinders', 'Transmission Type', 'Driven_Wheels', 'Number of Doors', 'Market Category', 'Vehicle Size', 'Vehicle Style', 'highway MPG', 'city mpg']]
y = df['MSRP']

# Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('R-squared:', r2)

# Visualize Results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Car Prices')
plt.show()

# Save the Model
joblib.dump(model, 'car_price_prediction_model.pkl')
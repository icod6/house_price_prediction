import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load dataset
df = pd.read_csv('dataset/california_housing.csv')

# Fill missing values with the mean of the column
df.fillna(df.mean(), inplace=True)

# Create proxy features
df['total_bedrooms'] = df['total_bedrooms']  # Number of bedrooms
df['total_bathrooms'] = df['population'] / 500  # Approximate number of bathrooms (proxy)
df['size_in_sqft'] = df['total_rooms'] / df['households'] * 250  # Approximate size in square feet

# Select features and target
X = df[['total_bedrooms', 'total_bathrooms', 'size_in_sqft', 'longitude', 'latitude']]
y = df['median_house_value']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R²): {r2}')

# Save the trained model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")

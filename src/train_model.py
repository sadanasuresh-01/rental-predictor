import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, '..', 'data', 'openrent.xlsx')

df = pd.read_excel(data_path)

df.columns = df.columns.str.strip()

print("Available columns:", df.columns)

df = df.dropna(subset=['bedrooms', 'bathrooms', 'squareft', 'price'])

X = df[['bedrooms', 'bathrooms', 'squareft']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

model_path = os.path.join(base_dir, '..', 'models', 'rental_model.pkl')
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)

print("✅ Model trained and saved at:", model_path)

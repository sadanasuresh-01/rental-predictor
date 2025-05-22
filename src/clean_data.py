import pandas as pd
import os

raw_data_path = os.path.join('data', 'raw_openrent.csv')
clean_data_path = os.path.join('data', 'cleaned_openrent.csv')

try:
    df = pd.read_csv(raw_data_path)
except FileNotFoundError:
    print("❌ raw_openrent.csv not found. Please check the path.")
    exit()

print("📊 Columns in dataset:", df.columns.tolist())

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

required_columns = ['bedrooms', 'bathrooms', 'squareft','price']
missing_cols = [col for col in required_columns if col not in df.columns]

if missing_cols:
    print(f"❌ Missing columns in dataset: {missing_cols}")
    exit()

df_cleaned = df.dropna(subset=required_columns)

df_cleaned = df_cleaned[
    (df_cleaned['price'] > 100) &
    (df_cleaned['price'] < 10000) &
    (df_cleaned['bedrooms'] > 0) &
    (df_cleaned['bathrooms'] > 0)
]

df_cleaned.to_csv(clean_data_path, index=False)
print(f"✅ Cleaned data saved to {clean_data_path}")
print(f"✅ Final rows: {len(df_cleaned)}")

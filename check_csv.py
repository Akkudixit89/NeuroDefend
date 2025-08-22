import pandas as pd

try:
    df = pd.read_csv("url_features.csv")
    print("CSV loaded successfully!\n")
    print("Columns:\n", df.columns)
except FileNotFoundError:
    print("❌ File not found. Please ensure 'url_features.csv' exists in the current directory.")
except Exception as e:
    print("❌ An error occurred while loading the CSV:\n", e)

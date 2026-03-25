# create_data.py
import pandas as pd
from sklearn.datasets import load_iris
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Load iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Save to CSV
df.to_csv('data/iris.csv', index=False)
print("Dataset created at data/iris.csv")
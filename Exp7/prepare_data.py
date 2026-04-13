"""Data preparation stage for DVC pipeline"""
import json
import seaborn as sns
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Create data directory if it doesn't exist
Path("data").mkdir(exist_ok=True)

# Load dataset
iris = sns.load_dataset('iris')

# Remove duplicates
iris.drop_duplicates(inplace=True)

# Encode target variable
label_encoder = LabelEncoder()
iris['species_encoded'] = label_encoder.fit_transform(iris['species'])

# Save full dataset
iris.to_csv("data/iris_full.csv", index=False)

# Split data into train and test
X = iris.drop(['species', 'species_encoded'], axis=1)
y = iris['species_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save splits
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

# Save label encoding mapping
encoding_map = {i: label for i, label in enumerate(label_encoder.classes_)}
with open("data/label_encoding.json", "w") as f:
    json.dump(encoding_map, f)

print("✓ Data preparation completed")
print(f"  - Full dataset: {len(iris)} samples")
print(f"  - Train set: {len(X_train)} samples")
print(f"  - Test set: {len(X_test)} samples")
print(f"  - Classes: {list(label_encoder.classes_)}")

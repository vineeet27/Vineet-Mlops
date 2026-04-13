
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# %%
# Load Dataset
df = sns.load_dataset('iris')
df.head()

# %%
# Dataset Info
df.info()

# %%
# Statistical Summary
df.describe()

# %% [markdown]
# ## Exploratory Data Analysis (EDA)

# %%
# Pairplot
sns.pairplot(df, hue='species')
plt.show()

# %%
# Distribution plots
for col in df.columns[:-1]:
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# %%
# Correlation Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title('Correlation Matrix')
plt.show()

# %% [markdown]
# ## Model Training

# %%
# Prepare Data
X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Train Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# %% [markdown]
# ## Evaluation

# %%
# Predictions
y_pred = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nClassification Report:\n', classification_report(y_test, y_pred))
print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred))

# %%
# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# %% [markdown]
# ## Conclusion
# 
# Logistic Regression model successfully trained and evaluated on Iris dataset.



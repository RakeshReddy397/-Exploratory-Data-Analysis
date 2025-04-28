# -Exploratory-Data-Analysis # Step 1: Upload the CSV File
from google.colab import files
uploaded = files.upload()  # Select and upload 'train.csv'

# Step 2: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 3: Load Dataset (After Uploading)
data = pd.read_csv('/content/train.csv')

# Step 4: Basic Information
print("✅ Shape of the dataset:", data.shape)
print("\n✅ Data Types:\n", data.dtypes)
print("\n✅ First 5 Rows:\n", data.head())
print("\n✅ Statistical Summary:\n", data.describe())

# Step 5: Check Missing Values
print("\n✅ Missing Values:\n", data.isnull().sum())

# Step 6: Histograms for All Numerical Features
data.hist(figsize=(15, 10))
plt.suptitle('Feature Distributions', fontsize=20)
plt.show()

# Step 7: Boxplots for Numerical Columns
for column in data.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=data[column])
    plt.title(f'Boxplot of {column}', fontsize=16)
    plt.show()

# Step 8: Correlation Heatmap (Corrected)
# Only select numeric columns
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Calculate correlation matrix
correlation_matrix = numeric_data.corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap', fontsize=20)
plt.show()

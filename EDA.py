# Import necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the dataset
df = pd.read_csv("C:/Users/Pawan/OneDrive/Desktop/Data Science Internship/Hackathon_Ideal_Data.csv")

# Display the first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Summary statistics
print(df.describe())

# Visualize distribution of numerical columns
plt.figure(figsize=(10, 6))
sns.histplot(df['QTY'], bins=20, kde=True, color='blue')
plt.title('Distribution of Quantity')
plt.xlabel('Quantity')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['VALUE'], bins=20, kde=True, color='green')
plt.title('Distribution of Value')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Visualize categorical columns
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='GRP')
plt.title('Distribution of GRP')
plt.xlabel('GRP')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='STORECODE')
plt.title('Distribution of Store Codes')
plt.xlabel('Store Code')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('/mnt/data/Walmart_sales.csv')

# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# 1. Summary statistics for numerical features
summary_statistics = df.describe()

# 2. Sales distribution across stores
plt.figure(figsize=(10, 6))
sns.boxplot(x='Store', y='Weekly_Sales', data=df)
plt.title('Sales Distribution Across Stores')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Impact of holidays on weekly sales
plt.figure(figsize=(10, 6))
sns.boxplot(x='Holiday_Flag', y='Weekly_Sales', data=df)
plt.title('Impact of Holidays on Weekly Sales')
plt.xticks([0, 1], ['Non-Holiday', 'Holiday'])
plt.tight_layout()
plt.show()

# 4. Trends in sales over time (sampled across all stores)
plt.figure(figsize=(10, 6))
df.groupby('Date')['Weekly_Sales'].mean().plot()
plt.title('Trend of Average Weekly Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Average Weekly Sales')
plt.tight_layout()
plt.show()

# 5. Correlation between numerical features
correlation_matrix = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.show()

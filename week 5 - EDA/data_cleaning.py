import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('your_dataset.csv')  # replace with your actual dataset path/filename

# Summary stats
df.describe(include='all')

# Distribution of a numeric column
sns.histplot(df['numeric_col'], kde=True)
plt.show()

# Correlation heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.show()

# Boxplot to spot outliers
sns.boxplot(x=df['numeric_col'])
plt.show()

# Categorical value counts
df['category_col'].value_counts().plot(kind='bar')
plt.show()

# Pairwise relationships
sns.pairplot(df[['col1', 'col2', 'col3']])
plt.show()
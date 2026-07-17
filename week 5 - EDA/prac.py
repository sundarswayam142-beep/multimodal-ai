import pandas as pd
import numpy as np

df = pd.read_csv('your_dataset.csv')

# Rename columns for consistency
df.rename(columns={'Old Name': 'new_name'}, inplace=True)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Check unique values in a column (spot typos/inconsistent categories)
df['category_col'].unique()
df['category_col'].nunique()

# Replace inconsistent category labels
df['category_col'] = df['category_col'].replace({'usa': 'USA', 'U.S.A': 'USA'})

# Convert a column to category dtype (saves memory, useful for grouping)
df['category_col'] = df['category_col'].astype('category')

# Reset index after dropping rows
df.reset_index(drop=True, inplace=True)

# Apply a custom function across a column
df['numeric_col'] = df['numeric_col'].apply(lambda x: x if x > 0 else np.nan)

# Check for negative values where they shouldn't exist
(df['numeric_col'] < 0).sum()
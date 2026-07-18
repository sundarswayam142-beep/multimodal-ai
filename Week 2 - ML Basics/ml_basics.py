# Pandas Wrangling and Pre-processing Simulation
import pandas as pd
import numpy as np

# Create synthetic Industrial Quality Data matching the tutorial concepts
raw_data = {

    'Defect_Area_mm': [1.2, np.nan, 0.5, 3.4, np.nan],
    'Inspection_Status': ['Pass', 'Fail', 'Pass', 'Fail', 'Pass']
}

df = pd.DataFrame(raw_data)
print("Original Frame Content:\n", df)

# Clean missing entries using concepts from CodeWithHarry [00:20:15]
df['Defect_Area_mm'] = df['Defect_Area_mm'].fillna(0.0)

# Rename Columns for styling [00:21:47]
df = df.rename(columns={'Defect_Area_mm': 'Defect_Size'})

# Export processing without including indexing trackers [00:28:50, 00:29:58]
df.to_csv("cleaned_quality_data.csv", index=False)
print("\nProcessed Frame Content:\n", df)

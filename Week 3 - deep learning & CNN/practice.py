import pandas as pd
import numpy as np

sensor_data = {
    'Timestamp': pd.date_range(start='2026-03-01', periods=5, freq='h'),
    'Vibration_Hz': [45.2, 48.9, 120.5, 46.1, 135.2],
    'Temperature_C': [32.1, 34.5, 55.2, 33.0, 58.1]
}

df = pd.DataFrame(sensor_data)

df['Is_Overheating'] = np.where(df['Temperature_C'] > 50.0, 1, 0)
df['Vibration_Normalized'] = (df['Vibration_Hz'] - df['Vibration_Hz'].mean()) / df['Vibration_Hz'].std()

print(df)

import numpy as np
import pandas as pd

readings = {
    'Device_ID': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6'],
    'Voltage': [220.1, 219.8, 220.4, 350.2, 219.9, 220.2]
}

df = pd.DataFrame(readings)

voltage_mean = df['Voltage'].mean()
voltage_std = df['Voltage'].std()

cutoff = voltage_std * 2
lower_bound = voltage_mean - cutoff
upper_bound = voltage_mean + cutoff

df['Is_Anomaly'] = np.where((df['Voltage'] < lower_bound) | (df['Voltage'] > upper_bound), 1, 0)

print(df)

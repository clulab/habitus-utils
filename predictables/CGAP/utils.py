import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#given a pandas series return its values as z-scaled.
def get_z_scaled(data):
    assert type(data) is pd.Series
    data_scaled=data.copy()
    data_scaled = (data_scaled - (data_scaled.mean())) / (data_scaled.std())
    return data_scaled

def scale_min_max(data):
    scaler = MinMaxScaler(feature_range=(1,2))
    assert type(data) is pd.Series
    df_min_max_scaled = data.copy()
    df_min_max_scaled=np.asarray(df_min_max_scaled).reshape(-1, 1)
    scaler.fit(df_min_max_scaled)
    df_min_max_scaled=scaler.transform(df_min_max_scaled)
    return df_min_max_scaled

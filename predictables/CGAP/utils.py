import math
import pandas as pd

#given a pandas series return its values as z-scaled.
def get_z_scaled(data):
    assert type(data) is pd.Series
    data_scaled=data.copy()
    data_scaled = (data_scaled - (data_scaled.mean())) / (data_scaled.std())
    return data_scaled

def scale_min_max(data):
    assert type(data) is pd.Series
    df_min_max_scaled = data.copy()
    df_min_max_scaled = (df_min_max_scaled - df_min_max_scaled.min()) / (
                df_min_max_scaled.max() - df_min_max_scaled.min())
    return df_min_max_scaled

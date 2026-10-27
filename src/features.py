import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait les composantes calendaires utiles pour anticiper le trafic.
    """
    df = df.copy()
    df['hour'] = df['date_time'].dt.hour
    df['day'] = df['date_time'].dt.day
    df['month'] = df['date_time'].dt.month
    df['dayofweek'] = df['date_time'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    return df

def add_lag_features(df: pd.DataFrame, target_col: str = 'traffic_volume') -> pd.DataFrame:
    """
    Crée des variables de décalage (lags) : ex. trafic à la même heure la veille (lag 24h).
    """
    df = df.copy()
    # Lag de 24h (veille même heure) et de 168h (semaine précédente même heure)
    df['lag_24h'] = df[target_col].shift(24)
    df['lag_168h'] = df[target_col].shift(168)
    
    # Suppression des lignes contenant des NaN générés par les shifts
    df = df.dropna(subset=[target_col, 'lag_24h', 'lag_168h']).reset_index(drop=True)
    return df
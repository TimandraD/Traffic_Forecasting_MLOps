import pandas as pd
from pathlib import Path

def load_and_clean_data(file_path: str | Path) -> pd.DataFrame:
    """
    Charge le fichier CSV du trafic, convertit la colonne date_time,
    trie les données par ordre chronologique et supprime les doublons.
    """
    df = pd.read_csv(file_path)
    
    # Conversion de la date en datetime
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    # Tri chronologique indispensable pour une série temporelle
    df = df.sort_values('date_time').reset_index(drop=True)
    
    # Suppression d'éventuels doublons sur le timestamp
    df = df.drop_duplicates(subset=['date_time'])
    
    return df

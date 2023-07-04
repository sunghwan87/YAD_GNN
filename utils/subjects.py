import pandas as pd

def load_YAD_labels(file_path: str) -> pd.DataFrame:
    subj_data = pd.read_csv(file_path, encoding='CP949')
    return subj_data

def load_HCP_labels(file_path: str) -> pd.DataFrame:
    subj_data = pd.read_csv(file_path, encoding='CP949')
    return subj_data

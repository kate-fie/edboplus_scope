
import numpy as np
import pandas as pd
from chemUtils.utils import utils


class EDBOStandardScaler:
    """
    Custom standard scaler for EDBO.
    """
    def __init__(self):
        pass

    def fit(self, x):
        self.mu  = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

    def transform(self, x):
        for obj in range(0, len(self.std)):
            if self.std[obj] == 0.0:
                self.std[obj] = 1e-6
        return (x-[self.mu])/[self.std]

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

        for obj in range(0, len(self.std)):
            if self.std[obj] == 0.0:
                self.std[obj] = 1e-6
        return (x-[self.mu])/[self.std]

    def inverse_transform(self, x):
        return x * [self.std] + [self.mu]

    def inverse_transform_var(self, x):
        return x * [self.std]

def append_max_peak_height(row, objective_name: str, max_peak_height_path: str):
    """
    Append max peak height to row of dataframe
    """
    # Load max peak height.
    df_max_peak_height = pd.read_csv(max_peak_height_path, index_col=False)

    # Append max peak height to row.
    if row[objective_name] != 'PENDING' and row[objective_name] != None:
        if type(row[objective_name]) == str:
            row[objective_name] = float(row[objective_name])
        max_peak = df_max_peak_height[(df_max_peak_height['A'] == row['A']) & (df_max_peak_height['B'] == row['B'])][
            objective_name].values[0]

        # Compute new columns
        row[f'max_{objective_name}'] = max_peak
        row[f'percent_diff_{objective_name}'] = (max_peak - row[objective_name]) / max_peak

        # Reorder columns to ensure new columns are at the end
        new_columns = [f'max_{objective_name}', f'percent_diff_{objective_name}']
        all_columns = [col for col in row.index if col not in new_columns] + new_columns
        row = row.reindex(all_columns)

    return row

def append_avg_tanimoto_sim(df: pd.DataFrame, smiles_cols: list) -> pd.DataFrame:
    """
    For each batch compute avg tanimoto sim within each reactant and append as columns.
    """
    batches = np.unique(np.array([x for x in df['batch_num']]))
    batches = batches[~np.isnan(batches)]
    for i in batches:
        A_smiles = list(df[df['batch_num']==i]['A_smiles'])
        B_smiles = list(df[df['batch_num']==i]['B_smiles'])
        A_avg_tani = utils.average_tanimoto(A_smiles)
        B_avg_tani = utils.average_tanimoto(B_smiles)
        condition = (df['batch_num'] == i)
        df.loc[condition, 'A_avg_tani'] = A_avg_tani
        df.loc[condition, 'B_avg_tani'] = B_avg_tani

    return df

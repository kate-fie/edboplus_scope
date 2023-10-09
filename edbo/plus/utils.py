
import numpy as np
import pandas as pd
from chemUtils.utils import utils
from rdkit import Chem
from rdkit.Chem import AllChem

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

def get_ecfp(row: pd.Series, smiles_cols: list) -> pd.Series:
    """
    Get ECFP representation of reactants.
    """
    # Get ECFP representation of reactants.
    A_ecfp = list(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(row[smiles_cols[0]]), 2, nBits=1024))
    B_ecfp = list(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(row[smiles_cols[1]]), 2, nBits=1024))

    return pd.Series(A_ecfp + B_ecfp)

def use_ecfp_representation(df: pd.DataFrame, smiles_cols: list, columns_regression: list) -> pd.DataFrame:
    """
    Use ECFP representation of reactants.
    """
    # Generate column names for the ECFP representation
    A_ecfp_cols = [f'A_ecfp_{i}' for i in range(1024)]
    B_ecfp_cols = [f'B_ecfp_{i}' for i in range(1024)]

    # Apply get_ecfp and concatenate results with original dataframe
    ecfp_df = df[smiles_cols].apply(get_ecfp, axis=1, result_type='expand')
    ecfp_df.columns = A_ecfp_cols + B_ecfp_cols

    # Concatenate the ECFP columns to the original dataframe
    df = pd.concat([df, ecfp_df], axis=1)

    # Extend columns_regression list
    columns_regression.extend(A_ecfp_cols + B_ecfp_cols)

    return df, columns_regression



import shutil
from edbo.plus.benchmark.multiobjective_benchmark import Benchmark
import os
import numpy as np
import pandas as pd
import argparse
import json

# Command-line input
parser = argparse.ArgumentParser(description="Benchmark script")
parser.add_argument('--rxn_components', type=str, help='Path to reaction components that you want to explore.', required=True)
parser.add_argument('--data', type=str, help='Path to data file.', required=True)
parser.add_argument('--batch', type=int, default=96, help='Batch size.', required=True)
parser.add_argument('--max_path', type=str, default=None, help='Path to max peak height file.', required=True)
parser.add_argument('--init_exp', action='store_true', help='Initialize with exp samples chosen randomly.')
parser.add_argument('--init_size', type=int, default=96, help='Number of initial samples.')
parser.add_argument('--train_choose', action='store_true', help='You already chose the train samples.')
parser.add_argument("--seed", type=int, default=1, help="Seed value for random initialization.")
parser.add_argument('--rounds', type=int, default=3, help='Number of rounds to run. Default is 3.')
parser.add_argument('--ecfp', action='store_true', help='Use ECFP encoding.')
args = parser.parse_args()

# Use parsed arguments
seed = args.seed
rxn_components_path = args.rxn_components
data_path = args.data
batch = args.batch
init_exp = args.init_exp
init_size = args.init_size
train_choose = args.train_choose
rounds = args.rounds
max_path = args.max_path
ecfp = args.ecfp

#######################
# Benchmark inputs
acq = 'EHVI'
budget = batch * rounds
sampling_method = 'lhs'

for embedding in ['OHE', 'ECFP']: # TODO: add 'ECFP'
    columns_regression = ['A', 'B']
    if embedding == 'ECFP':
        columns_regression = ['A', 'B']
        # TODO: ADD ECFP COLUMNS FOR REGRESSION to FILENAME
    # Select the features for the model.
    columns_regression += ['solvent',
                           'base',
                           'T']
    for scaling in ['raw', 'log']: # TODO: add 'decile' for classification.
        df_exp = pd.read_csv(data_path)
        df_exp['new_index'] = np.arange(0, len(df_exp.values))
        sort_column = 'new_index'

        # Add values for train column if init_exp is True.
        if init_exp:
            if train_choose is False and df_exp.train is False: # column does not exist and samples have not already been chosen.
                random_indicies = np.random.choice(df_exp.index, size=init_size, replace=False, seed=seed) # Choose random training samples to include.
                df_exp.loc[random_indicies, 'train'] = True
            columns_train = ['train']
            assert df_exp.train is not None, "There is no 'train' column."
        else:
            columns_train = None

        # Select objectives.
        if scaling == 'raw':
            objectives = ['peak_height']
            objective_modes = ['max']
        elif scaling == 'log':
            objectives = ['peak_height_log']
            objective_modes = ['max']
        elif scaling == 'decile':
            objectives = ['peak_height_decile']
            # TODO: What is the objective modes for classification?
            # objective_modes = 'min'
        objective_thresholds = [None]
        print(f"Columns for regression: {columns_regression}")

        # Set smiles_cols.
        smiles_cols = ['A_smiles', 'B_smiles']

        # Get reaction components.
        with open(rxn_components_path, 'r') as f:
            rxn_components = json.load(f)

        if init_exp:
            if train_choose:
                label_benchmark = f"benchmark_init_chosen_{init_size}exp_{embedding}_scaling_{scaling}_batch_{batch}_acq_{acq}.csv"
            else:
                label_benchmark = f"benchmark_init_{init_size}exp_{embedding}_scaling_{scaling}_batch_{batch}_acq_{acq}.csv"
        else:
            label_benchmark = f"benchmark_init_random_embedding_{embedding}_scaling_{scaling}_batch_{batch}_acq_{acq}.csv"

        # Remove previous files.
        if os.path.exists(label_benchmark):
            os.remove(label_benchmark)

        if os.path.exists(f'pred_{label_benchmark}'):
            os.remove(f'pred_{label_benchmark}')

        if os.path.exists(f'results_{label_benchmark}'):
            os.remove(f'results_{label_benchmark}')

        bench = Benchmark(
            df_ground=df_exp,
            features_regression=columns_regression,
            objective_names=objectives,
            objective_modes=objective_modes,
            reaction_components=rxn_components,
            objective_thresholds=objective_thresholds,
            columns_train=columns_train,
            init_size=init_size,
            filename=label_benchmark,
            filename_results=f'results_{label_benchmark}',
            index_column=sort_column,acquisition_function=acq,
            smiles_cols=smiles_cols,
            use_ecfp=ecfp,
            max_peak_height=max_path
        )

        bench.run(
            steps=int(budget/batch), batch=batch, seed=seed,
            init_method=sampling_method,
            plot_train=True, plot_predictions=True
        )

        if not os.path.exists('results'):
            os.mkdir('results')

        shutil.move(label_benchmark, f'results/{label_benchmark}')
        shutil.move(f'pred_{label_benchmark}', f'results/pred_{label_benchmark}')
        shutil.move(f'results_{label_benchmark}', f'results/results_{label_benchmark}')

#######################
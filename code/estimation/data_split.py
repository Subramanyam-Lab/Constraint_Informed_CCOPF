import os
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

def generate_gaussian_dataset(N=10000, D=10, mu=-0.024, sigma=0.036, seed=0):
    """Return a df of synthetic Gaussian ."""
    rng = np.random.RandomState(seed)
    data = rng.normal(loc=mu, scale=sigma, size=(N, D))
    return pd.DataFrame(data, columns=[f"x{i}" for i in range(1, D+1)])

def generate_cauchy_dataset(N=10000, D=10, scale=0.02, seed=0):
    """Return a df of a synthetic Cauchy."""
    rng = np.random.RandomState(seed)
    data = rng.standard_cauchy(size=(N, D)) * scale
    return pd.DataFrame(data, columns=[f"x{i}" for i in range(1, D+1)])

def load_real_dataset(path):
    """Pre-process the NordProol dataset and drop the -inf/NaN values"""
    df = pd.read_csv(path)
    df = df.replace({',': '', ' ': ''}, regex=True)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    regions = ['50Hz','AMP','AT','PL','TBW','TTG','TTGf','50Hzf','FR','FRf']
    errs = []
    for r in regions:
        act, fc = f"{r}-Actual", f"{r}-Forecast"
        if act in df and fc in df:
            err_col = f"{r}-Error"
            df[err_col] = (df[act] - df[fc]) / df[act]
            errs.append(err_col)
    df[errs] = df[errs].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=errs).reset_index(drop=True)
    return df[errs]

def split_synthetic(df_fn, name, out_dir, n_splits=10, train_frac=0.8):
    """Generate different synetic datasets and do the 80-20 splits."""
    os.makedirs(out_dir, exist_ok=True)
    N = len(df_fn(seed=0))
    cutoff = int(train_frac * N)
    for i in range(1, n_splits+1):
        seed = i  
        df = df_fn(seed=seed)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        df.iloc[:cutoff].to_csv(f"{out_dir}/{name}_split{i}_train.csv", index=False)
        df.iloc[cutoff:].to_csv(f"{out_dir}/{name}_split{i}_test.csv",  index=False)

def split_real(df, name, out_dir, n_splits=10, test_size=0.2, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    for i, (train_idx, test_idx) in enumerate(ss.split(df), 1):
        df.iloc[train_idx].to_csv(f"{out_dir}/{name}_split{i}_train.csv", index=False)
        df.iloc[test_idx].to_csv(f"{out_dir}/{name}_split{i}_test.csv",  index=False)

def main():
    # Path for output files and number of different datasets
    real_csv    = "/Users/tianyangyi/Desktop/Constraint_Informed_CCOPF/data/15daysNordPoolFinal.csv"
    out_dir     = "../data/splits"
    n_splits    = 10

    # Synthetic Gaussian and Cauchy
    split_synthetic(generate_gaussian_dataset, "gaussian", out_dir, n_splits)
    split_synthetic(generate_cauchy_dataset,  "cauchy",   out_dir, n_splits)

    # NordPool dataset with different splits
    df_real = load_real_dataset(real_csv)
    split_real(df_real, "real", out_dir, n_splits)

    total = 2*n_splits + n_splits
    print(f"Done: {total}×2 files written to {out_dir}")

if __name__ == "__main__":
    main()

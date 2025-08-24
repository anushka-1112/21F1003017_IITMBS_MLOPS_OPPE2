import pandas as pd
import numpy as np

def poison_labels(input_csv: str, output_csv: str, label_col: str = "Class", flip_percent: float = 0.05, random_state: int = 42):
    
    df = pd.read_csv(input_csv)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV.")

    # Number of samples to flip
    n_samples = len(df)
    n_flip = int(flip_percent * n_samples)

    np.random.seed(random_state)
    flip_indices = np.random.choice(df.index, size=n_flip, replace=False)

    # Flip labels: 1 -> 0, 0 -> 1
    df.loc[flip_indices, label_col] = 1 - df.loc[flip_indices, label_col]

    # Save poisoned data
    df.to_csv(output_csv, index=False)

    print(f"✅ Flipped {n_flip} labels ({flip_percent*100:.1f}%) and saved to {output_csv}")
    return df

import numpy as np
import pandas as pd

def preprocess_train_data(data_path, drop_std=True):
    """
    Reads in the training data, drops unnecessary columns, and creates the binary class2 column.
    If drop_std is True, also drops all columns ending with '.std'.
    Returns the processed DataFrame.
    """
    df = pd.read_csv(data_path)
    df = df.drop(columns=['id', 'partlybad','date'])
    df['class2'] = np.where(df['class4'].eq('nonevent'), 'nonevent', 'event')

    if drop_std:
        std_cols = [col for col in df.columns if col.endswith(".std")]
        df = df.drop(columns=std_cols)

    return df  

def split_xy(df):
    """
    Splits the DataFrame into features (X) and target variables (y2 for class2 and y4 for class4).
    Returns X, y2, and y4.
    """
    X = df.drop(columns=['class4', 'class2'])
    y2 = df['class2']
    y4 = df['class4']
    return X, y2, y4

def generate_synthetic_data(train_df, num_datasets=10):
    """
    Takes in dataset with mean and std columns for each numerical feature.
    Generates synthetic datasets by adding normally distributed noise based on the std values.
    The classes of the synthetic datasets are the same as the original dataset.
    Returns a single DataFrame with all synthetic datasets concatenated.
    """
    mean_cols = [col for col in train_df.columns if col.endswith(".mean")]
    std_cols = [col for col in train_df.columns if col.endswith(".std")]
    df_means = train_df.drop(columns=std_cols)
    df_stds = train_df.drop(columns=mean_cols)
    synthetic_dfs = []
    for i in range(num_datasets):
        synthetic_df = df_means.copy()
        for col in mean_cols:
            std_col = col.replace(".mean", ".std")
            synthetic_df[col] += np.random.normal(0, df_stds[std_col])
        synthetic_dfs.append(synthetic_df)
    return pd.concat(synthetic_dfs, ignore_index=True)

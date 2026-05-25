import numpy as np
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def standardize_data(
    df: pd.DataFrame,
    cols_to_include: list[str] | None=None,
    scaler: StandardScaler | None=None,
):
    """
    Standardize data. Use scaler if provided, in which case only the scalers transform
    function will be called. If scaler is None, fit sklearn's StandardScaler and transform
    data to zero mean and unit variance.

    Args:
        - df: DataFrame:
            Data to standardize.
        - cols_to_include: list[str] or None
            Specify columns to include in the standardization. If None, all
            columns will be included. Default is None. Columns that were excluded
            will be joined back to the dataframe preserving original column order.
        - scaler: StandardScaler or None
            Default is None. If None, use sklearn's StandardScaler.

    Returns: Tuple of
        - DataFrame: The scaled data.
        - StandardScaler: The fitted scaler, if no scaler was provided in the arguments.
            Input scalar, if one was provided.
    """
    if cols_to_include is None:
        cols_to_include = df.columns

    # Separate the data that will not be standardized
    df_X = df[cols_to_include]
    cols_not_included = [col for col in df.columns if col not in cols_to_include]

    if scaler is not None:
        scaled_data = scaler.transform(df_X)
    else:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_X)
    df_scaled = pd.DataFrame(scaled_data, columns=cols_to_include)

    if len(cols_not_included) > 0:
        # Join the separated data
        df_scaled[cols_not_included] = df[cols_not_included]

    # Rearrange back to original order
    df_scaled = df_scaled[df.columns]

    return df_scaled, scaler


def apply_scaler(
    scaler: StandardScaler,
    df: pd.DataFrame,
    cols_to_include: list[str] | None=None
):
    df_scaled, _ = standardize_data(
        df,
        cols_to_include=cols_to_include,
        scaler=scaler
    )

    return df_scaled


def remove_unwanted_cols(
    df: pd.DataFrame,
    cols_to_keep: list[str]
):
    cols_to_include = [col for col in df.columns if col in cols_to_keep]
    return df[cols_to_include]


def combine_label_col_to_dataframe(
    data_df: pd.DataFrame,
    label_df: pd.DataFrame,
    label_col: str,
    binary_labels: tuple[str, str],
    binary_col_name: str = "class2"
):
    data_cols = data_df.columns.to_list()
    df = data_df.copy()

    df = pd.concat([df, label_df.reset_index()], axis=1)

    target_label, inverse_label = binary_labels

    df[binary_col_name] = boolean_to_label(
        boolean_series=(df[label_col] == target_label),
        target_label=target_label,
        inverse_label=inverse_label
    )

    col_order = [label_col, binary_col_name] + data_cols
    df = df[col_order]

    return df


def boolean_to_label(
    boolean_series: pd.Series,
    target_label,
    inverse_label
):
    labels = np.array([inverse_label] * len(boolean_series), dtype=object)
    labels[boolean_series] = target_label
    return pd.Series(labels, name=boolean_series.name)


def encode_label(
    df,
    label_col,
):
    """
    Numeric encoding for string labels
    """
    unique_labels = np.unique_values(df[label_col].values)
    label_to_number_map = dict()
    number_to_label_map = dict()

    for i in range(len(unique_labels)):
        label_to_number_map[unique_labels[i]] = i
        number_to_label_map[i] = unique_labels[i]

    numeric_col = df[label_col].map(label_to_number_map)

    return numeric_col, number_to_label_map


def prepare_standardized_datasets(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    data_vars: list[str],
    label_var: str,
    label_values: tuple[str, str],
    include_validation_split: bool=True,
    validation_split_size: float=0.2,
):
    cols_to_keep = data_vars + [label_var]

    # Preliminary clean-up
    df_train = remove_unwanted_cols(df_train, cols_to_keep)
    df_test = remove_unwanted_cols(df_test, cols_to_keep)

    X = df_train.drop(columns=label_var)
    y = df_train[label_var]
    X_test = df_test

    # Split data
    if include_validation_split:
        X_train, X_validation, y_train, y_validation = train_test_split(
            X, y, test_size=validation_split_size, shuffle=True
        )
    else:
        X_train, y_train = X, y
        X_validation, y_validation = None, None

    # Standardize to mean 0, variance 1
    X_train, scaler = standardize_data(X_train, cols_to_include=data_vars)

    if X_validation is not None:
        X_validation = apply_scaler(scaler, X_validation, cols_to_include=data_vars)

    X_test = apply_scaler(scaler, X_test, cols_to_include=data_vars)

    # Combine back to dataframes
    df_train = combine_label_col_to_dataframe(
        data_df=X_train,
        label_df=y_train,
        label_col=label_var,
        binary_labels=label_values
    )

    df_test = X_test

    if (X_validation is not None) and (y_validation is not None):
        df_validation = combine_label_col_to_dataframe(
            data_df=X_validation,
            label_df=y_validation,
            label_col=label_var,
            binary_labels=label_values
        )
    else:
        df_validation = None

    return df_train, df_validation, df_test

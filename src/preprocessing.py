import pandas as pd

from sklearn.preprocessing import StandardScaler


def standardize_data(
    df: pd.DataFrame,
    cols_to_include: list[str] | None=None
):
    """
    Standardize data with sklearn's StandardScaler.

    Args:
        - df: DataFrame:
            Data to standardize.
        - cols_to_include: list[str] or None
            Specify columns to include in the standardization. If None, all
            columns will be included. Default is None. Columns that were excluded
            will be joined back to the dataframe preserving original column order.

    Returns:
        - DataFrame: The scaled data.
        - StandardScaler: The fitted scaler.
    """
    if cols_to_include is None:
        cols_to_include = df.columns

    # Separate the data that will not be standardized
    df_X = df[cols_to_include]
    cols_not_included = [col for col in df.columns if col not in cols_to_include]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_X)
    df_scaled = pd.DataFrame(scaled_data, columns=cols_to_include)

    if len(cols_not_included) > 0:
        # Join the separated data
        df_scaled[cols_not_included] = df[cols_not_included]

    # Rearrange back to original order
    df_scaled = df_scaled[df.columns]

    return df_scaled, scaler

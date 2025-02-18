import pandas as pd
import pytest

# Function to combine DataFrames
def combine_dataframes(dfs, prefixes, merge_column='ID'):
    """
    Combines a list of DataFrames on a specific column and adds prefixes to columns
    only if the same column exists in multiple DataFrames.

    Args:
        dfs (list of pd.DataFrame): List of DataFrames to combine.
        prefixes (list of str): List of prefixes for each DataFrame.
        merge_column (str): The column to merge on (default is 'ID').

    Returns:
        pd.DataFrame: The combined DataFrame.
    """
    # Check for duplicate columns across DataFrames
    all_columns = []
    for df in dfs:
        all_columns.extend(df.columns)
    duplicate_columns = set([col for col in all_columns if all_columns.count(col) > 1 and col != merge_column])

    # Add prefixes to duplicate columns (except the merge column)
    for i, df in enumerate(dfs):
        df = df.rename(columns={col: f'{prefixes[i]}{col}' for col in df.columns if col in duplicate_columns and col != merge_column})
        dfs[i] = df  # Update the DataFrame in the list

    # Merge all DataFrames on the merge column
    merged_df = dfs[0]  # Start with the first DataFrame
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=merge_column, how='outer')  # Merge on the merge column

    return merged_df

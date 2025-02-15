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

# Pytest Test Cases
def test_combine_dataframes_unique_columns():
    # Test case: DataFrames with unique columns
    df1 = pd.DataFrame({'ID': [1, 2], 'value1': [10, 20]})
    df2 = pd.DataFrame({'ID': [1, 3], 'value2': [100, 300]})
    result = combine_dataframes([df1, df2], ['df1_', 'df2_'])
    expected = pd.DataFrame({
        'ID': [1, 2, 3],
        'value1': [10, 20, None],
        'value2': [100, None, 300]
    })
    pd.testing.assert_frame_equal(result, expected)

def test_combine_dataframes_duplicate_columns():
    # Test case: DataFrames with duplicate columns
    df1 = pd.DataFrame({'ID': [1, 2], 'common_col': [10, 20]})
    df2 = pd.DataFrame({'ID': [1, 3], 'common_col': [100, 300]})
    result = combine_dataframes([df1, df2], ['df1_', 'df2_'])
    expected = pd.DataFrame({
        'ID': [1, 2, 3],
        'df1_common_col': [10, 20, None],
        'df2_common_col': [100, None, 300]
    })
    pd.testing.assert_frame_equal(result, expected)

def test_combine_dataframes_empty_dataframe():
    # Test case: One of the DataFrames is empty
    df1 = pd.DataFrame({'ID': [1, 2], 'value1': [10, 20]})
    df2 = pd.DataFrame(columns=['ID', 'value2'])
    result = combine_dataframes([df1, df2], ['df1_', 'df2_'])
    expected = pd.DataFrame({
        'ID': [1, 2],
        'value1': [10, 20],
        'value2': [None, None]
    })
    pd.testing.assert_frame_equal(result, expected)

def test_combine_dataframes_missing_merge_column():
    # Test case: One DataFrame is missing the merge column
    df1 = pd.DataFrame({'ID': [1, 2], 'value1': [10, 20]})
    df2 = pd.DataFrame({'value2': [100, 200]})
    with pytest.raises(KeyError):
        combine_dataframes([df1, df2], ['df1_', 'df2_'])

def test_combine_dataframes_custom_merge_column():
    # Test case: Custom merge column
    df1 = pd.DataFrame({'custom_id': [1, 2], 'value1': [10, 20]})
    df2 = pd.DataFrame({'custom_id': [1, 3], 'value2': [100, 300]})
    result = combine_dataframes([df1, df2], ['df1_', 'df2_'], merge_column='custom_id')
    expected = pd.DataFrame({
        'custom_id': [1, 2, 3],
        'value1': [10, 20, None],
        'value2': [100, None, 300]
    })
    pd.testing.assert_frame_equal(result, expected)

def test_combine_dataframes_no_duplicate_columns():
    # Test case: No duplicate columns (no prefixes should be added)
    df1 = pd.DataFrame({'ID': [1, 2], 'value1': [10, 20]})
    df2 = pd.DataFrame({'ID': [1, 3], 'value2': [100, 300]})
    df3 = pd.DataFrame({'ID': [1, 4], 'value3': [1000, 4000]})
    result = combine_dataframes([df1, df2, df3], ['df1_', 'df2_', 'df3_'])
    expected = pd.DataFrame({
        'ID': [1, 2, 3, 4],
        'value1': [10, 20, None, None],
        'value2': [100, None, 300, None],
        'value3': [1000, None, None, 4000]
    })
    pd.testing.assert_frame_equal(result, expected)

import pandas as pd

# Function to combine DataFrames
def combine_dataframes(dfs, prefixes, merge_column='ID'):
    """
    Combines a list of DataFrames on a specific column and adds prefixes to columns.

    Args:
        dfs (list of pd.DataFrame): List of DataFrames to combine.
        prefixes (list of str): List of prefixes for each DataFrame.
        merge_column (str): The column to merge on (default is 'ID').

    Returns:
        pd.DataFrame: The combined DataFrame.
    """
    # Add prefixes to columns (except the merge column)
    for i, df in enumerate(dfs):
        df = df.add_prefix(prefixes[i])  # Add prefix to all columns
        df = df.rename(columns={f'{prefixes[i]}{merge_column}': merge_column})  # Rename the prefixed merge column back
        dfs[i] = df  # Update the DataFrame in the list

    # Merge all DataFrames on the merge column
    merged_df = dfs[0]  # Start with the first DataFrame
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=merge_column, how='outer')  # Merge on the merge column

    return merged_df

# Sample DataFrames
df1 = pd.DataFrame({
    'ID': [1, 2, 3],
    'value1': [10, 20, 30]
})

df2 = pd.DataFrame({
    'ID': [1, 2, 4],
    'value2': [100, 200, 400]
})

df3 = pd.DataFrame({
    'ID': [1, 3, 4],
    'value3': [1000, 3000, 4000]
})

# List of DataFrames
dfs = [df1, df2, df3]

# Prefixes for each DataFrame
prefixes = ['df1_', 'df2_', 'df3_']

# Call the function
result = combine_dataframes(dfs, prefixes)

# Display the result
print(result)



import pandas as pd

# Sample DataFrame
data = {
    'variable': ['var1', 'var1', 'var2', 'var2', 'var3', 'var3'],
    'test': ['testA', 'testB', 'testA', 'testB', 'testA', 'testB'],
    'test_stat': [1.2, 2.3, 3.4, 4.5, 5.6, 6.7],
    'p_value': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
}

df = pd.DataFrame(data)

# Pivot the DataFrame
pivot_df = df.pivot(index='variable', columns='test', values=['test_stat', 'p_value'])

# Flatten the MultiIndex columns
pivot_df.columns = [f'{test}_{value}' for value, test in pivot_df.columns]

# Reset the index to make 'variable' a column again
pivot_df = pivot_df.reset_index()

# Display the result
print(pivot_df)



import pandas as pd

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

# Sample DataFrames
df1 = pd.DataFrame({
    'ID': [1, 2, 3],
    'value1': [10, 20, 30],
    'common_col': [100, 200, 300]  # Common column
})

df2 = pd.DataFrame({
    'ID': [1, 2, 4],
    'value2': [100, 200, 400],
    'common_col': [150, 250, 350]  # Common column
})

df3 = pd.DataFrame({
    'ID': [1, 3, 4],
    'value3': [1000, 3000, 4000]
})

# List of DataFrames
dfs = [df1, df2, df3]

# Prefixes for each DataFrame
prefixes = ['df1_', 'df2_', 'df3_']

# Call the function
result = combine_dataframes(dfs, prefixes)

# Display the result
print(result)


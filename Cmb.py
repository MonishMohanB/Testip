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

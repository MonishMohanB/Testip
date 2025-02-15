import pandas as pd

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

# Add prefixes to columns (except the 'ID' column)
for i, df in enumerate(dfs):
    df = df.add_prefix(prefixes[i])  # Add prefix to all columns
    df = df.rename(columns={f'{prefixes[i]}ID': 'ID'})  # Rename the prefixed ID column back to 'ID'
    dfs[i] = df  # Update the DataFrame in the list

# Merge all DataFrames on the 'ID' column
merged_df = dfs[0]  # Start with the first DataFrame
for df in dfs[1:]:
    merged_df = pd.merge(merged_df, df, on='ID', how='outer')  # Merge on 'ID'

# Display the final merged DataFrame
print(merged_df)

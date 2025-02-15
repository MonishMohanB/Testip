import pandas as pd

# Sample DataFrames
df1 = pd.DataFrame({
    'id': [1, 2, 3],
    'value1': [10, 20, 30]
})

df2 = pd.DataFrame({
    'id': [1, 2, 4],
    'value2': [100, 200, 400]
})

df3 = pd.DataFrame({
    'id': [1, 3, 4],
    'value3': [1000, 3000, 4000]
})

# List of DataFrames
dfs = [df1, df2, df3]

# Column to merge on
merge_column = 'id'

# Prefixes for each DataFrame
prefixes = ['df1_', 'df2_', 'df3_']

# Initialize the merged DataFrame with the first DataFrame
merged_df = dfs[0].add_prefix(prefixes[0])

# Iterate over the remaining DataFrames and merge them
for i in range(1, len(dfs)):
    df = dfs[i].add_prefix(prefixes[i])
    merged_df = pd.merge(merged_df, df, left_on=f'{prefixes[0]}{merge_column}', right_on=f'{prefixes[i]}{merge_column}', how='outer')

# Drop duplicate merge columns
for prefix in prefixes[1:]:
    merged_df.drop(columns=[f'{prefix}{merge_column}'], inplace=True)

# Rename the first merge column to just 'id'
merged_df.rename(columns={f'{prefixes[0]}{merge_column}': merge_column}, inplace=True)

print(merged_df)

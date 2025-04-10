import pandas as pd
from scipy.stats import chi2_contingency

# Example dataset
data = {
    'Age_Group': ['Young', 'Old', 'Young', 'Old', 'Young', 'Old'],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'Education': ['High', 'College', 'High', 'College', 'High', 'College'],
    'Purchased': [1, 0, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

# 1. Create observation frequency
observation_freq = df.groupby(['Age_Group', 'Gender', 'Education']).size().reset_index(name='Observation_Freq')

# 2. Create event frequency
event_freq = df.groupby(['Age_Group', 'Gender', 'Education'])['Purchased'].sum().reset_index(name='Event_Freq')

# 3. Merge frequencies
merged_df = pd.merge(observation_freq, event_freq, on=['Age_Group', 'Gender', 'Education'])

# 4. Calculate probability
merged_df['Probability'] = merged_df['Event_Freq'] / merged_df['Observation_Freq']

# 5. Create contingency table (example: Age_Group vs. Purchased)
contingency_table = pd.crosstab(df['Age_Group'], df['Purchased'])

# 6. Run chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square Statistic: {chi2}\np-value: {p_value}")

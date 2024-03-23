import pandas as pd

# Read XPT file
df = pd.read_sas('LLCP2022.XPT ', format='xport')
df.head()

# Create a list of columns to keep
core_columns = ['_RFHLTH', '_TOTINDA', '_SMOKER3', '_DRNKWK2', '_MICHD', '_LTASTH1',
                '_AGEG5YR', '_SEX', '_RACEPR1', '_EDUCAG', '_INCOMG1', '_BMI5CAT']

# Subset the DataFrame
df_subset = df[core_columns]
df_subset.head()

# Change the column names
shortened_columns = {
    '_RFHLTH': 'GenHealth',
    '_TOTINDA': 'PhysAct',
    '_SMOKER3': 'SmokeStatus',
    '_DRNKWK2': 'AlcoholIntake',
    '_MICHD': 'HeartDisease',
    '_LTASTH1': 'AsthmaStatus',
    '_AGEG5YR': 'AgeGroup',
    '_SEX': 'Gender',
    '_RACEPR1': 'RaceEthnicity',
    '_EDUCAG': 'EduLevel',
    '_INCOMG1': 'IncomeLevel',
    '_BMI5CAT': 'BMI'
}

# Apply renaming
data = df_subset.rename(columns=shortened_columns)
print(data.columns)
print(data.head())
print(data.shape)

# Counting NaN values in each column
nan_counts = data.isna().sum()
print(nan_counts)

# Dropping rows where any of the three specific columns have NaN values
data_cleaned = data.dropna(subset=['SmokeStatus', 'HeartDisease', 'BMI'])
print(data_cleaned.shape)

# Counting values out of the specified range in the AlcoholIntake column
out_of_range_count = ((data['AlcoholIntake'] > 98999) | (data['AlcoholIntake'] == 99900)).sum()

print(f"Number of AlcoholIntake values out of the specified range: {out_of_range_count}")

# Deleting rows where AlcoholIntake values are out of the specified range
data_filtered = data_cleaned[~((data_cleaned['AlcoholIntake'] > 98999) | (data_cleaned['AlcoholIntake'] == 99900))]

print(f"Shape of the original data: {data_cleaned.shape}")
print(f"Shape of the data after removing out-of-range AlcoholIntake values: {data_filtered.shape}")

# Converting AlcoholIntake values from float to integer
data_filtered['AlcoholIntake'] = data_filtered['AlcoholIntake'].astype(int)

# Display the head of the DataFrame to verify the conversion
data_filtered.describe()

# Calculate z-scores for all numeric columns
z_scores = data_filtered.apply(zscore)

# Identify rows with any column having z-score > 3 or < -3 (outliers)
outliers_mask = (z_scores.abs() > 3).any(axis=1)

# Remove rows that have outliers in any column
data_no_outliers = data_filtered[~outliers_mask]

print(f"Original data size: {data_filtered.shape}")
print(f"Data size after removing outliers in any column: {data_no_outliers.shape}")

# Rename the DataFrame
cleaned_data = data_no_outliers

# Save the cleaned data to a new CSV file
cleaned_data.to_csv('cleaned_data.csv', index=False)

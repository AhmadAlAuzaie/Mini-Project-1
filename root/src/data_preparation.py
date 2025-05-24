import pandas as pd

file_path = '/mnt/data/mini_project_1_data.csv'

# Reset index after cleaning
df.reset_index(drop=True, inplace=True)

# Display basic information about the dataset
df.info(), df.head()

# Check for missing values
df.isnull().sum()

# Summary statistics of the dataset
df.describe().T

# Check for duplicate entries
df.duplicated().sum()
duplicates = df[df.duplicated(keep=False)].sort_values(by=list(df.columns))

# Drop duplicate entries
df.drop_duplicates(inplace=True)

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df['high_share'] = (df['shares'] > df['shares'].median()).astype(int)
    df['weekday'] = df['weekday'].str.strip().str.lower().str.capitalize()
    df['data_channel'] = df['data_channel'].fillna('Unknown').str.strip().str.lower().str.replace('_', ' ').str.capitalize()
    
    # Prepare features
    x = df.drop(columns=['shares', 'ID', 'URL'])
    y = df['high_share']
    
    # Drop rows where any x value is missing
    x = x.copy()
    x = x.dropna()
    y = y.loc[x.index]  # Align y accordingly
    
    return x, y
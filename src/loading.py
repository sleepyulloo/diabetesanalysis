import pandas as pd 


# during the EDA process I relazied that 
# the data had a lot of values that don't make sense mediccally 
# lik a person cannot have glucose,blood pressure or insulin of 0 
# will be cleaning the data here 

def load (filepath="data/diabetes.csv"):
    return pd.read_csv(filepath)

def replace_zeros_with_median(df, columns):
    df_clean = df.copy()
    for col in columns:
        median = df_clean[df_clean[col] != 0][col].median()
        df_clean[col] = df_clean[col].replace(0, median)
    return df_clean

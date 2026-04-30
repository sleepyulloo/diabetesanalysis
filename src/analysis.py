import pandas as pd 

def summary(df):
    return df.describe()

def missing_data(df):
    return df.isnull().sum()

def number_of_zeros(df):
    return (df == 0).sum()

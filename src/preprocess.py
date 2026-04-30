import pandas as pd
from sklearn.preprocessing import StandardScaler

def replace_zeros_with_median(df, columns):
    df_clean = df.copy()
    for col in columns:
        median = df_clean[df_clean[col] != 0][col].median()
        df_clean[col] = df_clean[col].replace(0, median)
    return df_clean

def prepare_features_and_target(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    return X, y

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

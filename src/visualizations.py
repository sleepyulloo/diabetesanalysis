import seaborn as sns
import matplotlib.pyplot as plt

# a correlation heatmap of all the columns 
def plot_correlation_heatmap(df, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation")
    plt.tight_layout()
    return plt

def plot_outcome_distribution(df):
    return df['Outcome'].value_counts()

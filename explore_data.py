import pandas as pd
df = pd.read_csv('dataset/deceptive-opinion.csv')

print(df.shape)
print(df.columns.tolist())
print(df.head())
print(df['deceptive'].value_counts())
print(df['polarity'].value_counts())
print(pd.crosstab(df['polarity'], df['deceptive']))
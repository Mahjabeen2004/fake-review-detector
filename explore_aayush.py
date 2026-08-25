import pandas as pd

df = pd.read_csv('dataset_aayush/amazon_reviews.txt', delimiter='\t')

print("Shape:", df.shape)

print("\nLabel balance:")
print(df['LABEL'].value_counts())

print("\nVerified purchase vs Label (cross-check our fake/genuine mapping):")
print(pd.crosstab(df['LABEL'], df['VERIFIED_PURCHASE']))

print("\nProduct category count:", df['PRODUCT_CATEGORY'].nunique())
print(df['PRODUCT_CATEGORY'].value_counts().head(10))

print("\nFirst few rows:")
print(df[['LABEL', 'RATING', 'VERIFIED_PURCHASE', 'PRODUCT_CATEGORY', 'REVIEW_TEXT']].head())
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)


if __name__ == "__main__":
    df = pd.read_csv('dataset_aayush/amazon_reviews.txt', delimiter='\t')

    print("Cleaning all reviews... this may take a moment (21,000 rows)")
    df['clean_text'] = df['REVIEW_TEXT'].apply(preprocess_text)

    print(df[['REVIEW_TEXT', 'clean_text']].head())

    df.to_csv('dataset_aayush/amazon_reviews_clean.csv', index=False)
    print("\nSaved cleaned dataset to dataset_aayush/amazon_reviews_clean.csv")
    print("Shape:", df.shape)
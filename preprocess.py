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
    sample = "This hotel was AMAZING!! Best stay ever, 10/10."
    print("BEFORE:", sample)
    print("AFTER: ", preprocess_text(sample))
    print()

    df = pd.read_csv('dataset/deceptive-opinion.csv')
    print("Cleaning all reviews... this may take a few seconds")
    df['clean_text'] = df['text'].apply(preprocess_text)
    print(df[['text', 'clean_text']].head())
    df.to_csv('dataset/deceptive-opinion-clean.csv', index=False)
    print("\nSaved cleaned dataset to dataset/deceptive-opinion-clean.csv")
    print("Shape:", df.shape)
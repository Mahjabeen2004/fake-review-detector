import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the stopword list (the "the, is, at, a" words) and the stemmer (chops words to root form)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    # Step 1: lowercase everything
    text = text.lower()

    # Step 2: remove punctuation and numbers, keep only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)

    # Step 3: tokenize (split sentence into a list of words)
    words = text.split()

    # Step 4: remove stopwords AND apply stemming in one pass
    words = [stemmer.stem(word) for word in words if word not in stop_words]

    # Step 5: join back into a single cleaned string
    return ' '.join(words)


# ---- Sanity check on one example first ----
sample = "This hotel was AMAZING!! Best stay ever, 10/10."
print("BEFORE:", sample)
print("AFTER: ", preprocess_text(sample))
print()

# ---- Now apply it to the whole dataset ----
df = pd.read_csv('dataset/deceptive-opinion.csv')

print("Cleaning all reviews... this may take a few seconds")
df['clean_text'] = df['text'].apply(preprocess_text)

# Sanity check: compare a few before/after side by side
print(df[['text', 'clean_text']].head())

# Save the cleaned dataset so we don't have to redo this every time
df.to_csv('dataset/deceptive-opinion-clean.csv', index=False)
print("\nSaved cleaned dataset to dataset/deceptive-opinion-clean.csv")
print("Shape:", df.shape)
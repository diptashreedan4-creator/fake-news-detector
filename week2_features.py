import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Load the cleaned data from Week 1
data = pd.read_csv("fake_news_cleaned.csv")
data['clean_text'] = data['clean_text'].fillna('')

print("Loaded cleaned data. Shape:", data.shape)

# ---------------------------
# PART 1: Exploratory Data Analysis (EDA)
# ---------------------------

# Word count per article
data['word_count'] = data['clean_text'].apply(lambda x: len(x.split()))

print("\nWord count statistics:")
print(data['word_count'].describe())

# Plot: word count distribution
plt.figure(figsize=(8, 5))
plt.hist(data['word_count'], bins=50, color='steelblue')
plt.title("Distribution of Article Word Counts")
plt.xlabel("Word Count")
plt.ylabel("Number of Articles")
plt.savefig("word_count_distribution.png")
print("Saved chart: word_count_distribution.png")

# Most common words in FAKE news (label = 1) vs REAL news (label = 0)
fake_words = ' '.join(data[data['label'] == 1]['clean_text']).split()
real_words = ' '.join(data[data['label'] == 0]['clean_text']).split()

fake_common = Counter(fake_words).most_common(15)
real_common = Counter(real_words).most_common(15)

print("\nTop 15 words in FAKE news:")
for word, count in fake_common:
    print(f"  {word}: {count}")

print("\nTop 15 words in REAL news:")
for word, count in real_common:
    print(f"  {word}: {count}")

# ---------------------------
# PART 2: TF-IDF Feature Engineering
# ---------------------------

print("\nBuilding TF-IDF features... this may take a minute.")
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(data['clean_text'])

print("TF-IDF matrix shape (articles, features):", X_tfidf.shape)
print("Sample feature words:", vectorizer.get_feature_names_out()[:20])

# Save the TF-IDF vectorizer and data for Week 3
import pickle
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

import scipy.sparse
scipy.sparse.save_npz("X_tfidf.npz", X_tfidf)
data['label'].to_csv("y_labels.csv", index=False)

print("\nSaved: tfidf_vectorizer.pkl, X_tfidf.npz, y_labels.csv")
print("Week 2 complete! Ready for Week 3 model building.")
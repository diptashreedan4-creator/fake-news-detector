import pandas as pd
import re

# Load the dataset
data = pd.read_csv("fake_news.csv")

# Step 1: Handle missing values
data = data.dropna(subset=['text'])          # drop rows with no article text (only ~39 rows)
data['title'] = data['title'].fillna('')     # fill missing titles with empty string
data['author'] = data['author'].fillna('')   # fill missing authors with empty string

print("Shape after dropping missing text rows:", data.shape)

# Step 2: Manual stopword list (common English words that don't help classification)
stopwords = set("""
a an the and or but if while is are was were be been being
in on at to for of with as by from about into over after
this that these those it its i you he she they we
not no nor so than too very can will just should now
""".split())

# Step 3: Manual text cleaning function
def clean_text(text):
    text = str(text).lower()                      # lowercase everything
    text = re.sub(r'[^a-z\s]', ' ', text)          # remove punctuation/numbers, keep letters only
    tokens = text.split()                          # manual tokenization (split by whitespace)
    tokens = [word for word in tokens if word not in stopwords and len(word) > 1]  # remove stopwords
    return ' '.join(tokens)

# Step 4: Apply cleaning to the text column
print("Cleaning text... this may take a minute for 20,000+ rows.")
data['clean_text'] = data['text'].apply(clean_text)

# Step 5: Preview before vs after
print("\nBEFORE cleaning (original text, first row):")
print(data['text'].iloc[0][:200])

print("\nAFTER cleaning (first row):")
print(data['clean_text'].iloc[0][:200])

# Step 6: Save the cleaned data so we don't have to redo this every time
data.to_csv("fake_news_cleaned.csv", index=False)
print("\nSaved cleaned data to fake_news_cleaned.csv")
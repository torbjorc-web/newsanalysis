import codecademylib3_seaborn
import pandas as pd
import numpy as np
from articles import articles
from preprocessing import preprocess_text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer


# Task 2: View one article
print("=== Sample Article ===")
print(articles[7])
print()

# Task 3: Preprocess all articles
processed_articles = [preprocess_text(document) for document in articles]
print("=== Preprocessed Article ===")
print(processed_articles[7])
print()

# Task 4-5: Initialize and fit CountVectorizer
count_vectorizer = CountVectorizer()
counts = count_vectorizer.fit_transform(processed_articles)

# Task 6-7: Convert counts to tf-idf using TfidfTransformer
tfidf_transformer = TfidfTransformer(norm=None)
tfidf_scores_transformed = tfidf_transformer.fit_transform(counts)

# Task 8-9: Initialize and fit TfidfVectorizer (one-step approach)
tfidf_vectorizer = TfidfVectorizer(norm=None)
tfidf_scores = tfidf_vectorizer.fit_transform(processed_articles)

# Task 10: Check if tf-idf scores are equal
print("=== TF-IDF Comparison ===")
if np.allclose(tfidf_scores_transformed.todense(), tfidf_scores.todense()):
    print(pd.DataFrame({'Are the tf-idf scores the same?': ['YES']}))
else:
    print(pd.DataFrame({'Are the tf-idf scores the same?': ['No, something is wrong :(']}))
print()

# Get vocabulary of terms
feature_names = tfidf_vectorizer.get_feature_names_out()

# Get article column names
article_index = [f"Article {i+1}" for i in range(len(articles))]

# Create DataFrame with word counts
df_word_counts = pd.DataFrame(counts.T.todense(), index=feature_names, columns=article_index)
print("=== Word Counts DataFrame ===")
print(df_word_counts)
print()

# Create DataFrame with tf-idf scores (using TfidfVectorizer results)
df_tf_idf = pd.DataFrame(tfidf_scores.T.todense(), index=feature_names, columns=article_index)
print("=== TF-IDF Scores DataFrame ===")
print(df_tf_idf)
print()

# Task 11-12: Get highest scoring tf-idf term for each article
print("=== Highest TF-IDF Term per Article ===")
for i in range(1, 11):
    top_term = df_tf_idf[[f'Article {i}']].idxmax()[0]
    print(f"Article {i}: {top_term}")

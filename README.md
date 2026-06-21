# Read the News Analysis

This project uses TF-IDF to analyze a small collection of news articles from *The News International*. The goal is to identify the most important terms in each article and use those terms as a quick signal of topic.

## What the project does

- Loads a list of news articles.
- Preprocesses the text with tokenization and lemmatization.
- Counts word occurrences using `CountVectorizer`.
- Converts counts to TF-IDF scores using `TfidfTransformer`.
- Computes TF-IDF scores directly using `TfidfVectorizer`.
- Checks that both TF-IDF methods return the same result.
- Finds the highest-scoring TF-IDF term for each article.

## Files

- `script.py` — main script for the analysis.
- `articles.py` — contains the sample news articles.
- `preprocessing.py` — contains the text preprocessing function.

## How it works

1. Each article is preprocessed to normalize the text.
2. Word counts are created for every article.
3. Counts are transformed into TF-IDF scores.
4. The highest TF-IDF term in each article is used as a simple topic indicator.

## Output

The script prints:

- One raw article.
- One preprocessed article.
- Word count and TF-IDF DataFrames.
- A check confirming both TF-IDF methods match.
- The top TF-IDF term for each article.

## Requirements

- Python
- pandas
- numpy
- scikit-learn
- codecademylib3_seaborn

## Notes

This project uses a small dataset, so it is meant for learning and exploration rather than production use.

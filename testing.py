import os
import pandas as pd
import numpy as np
import ast  # To safely evaluate Python literals
from nltk import ngrams
from collections import defaultdict

# --- Finish Method and Next Word Functions ---
def next_word(n_gram_series, n_prior_words):
    key = tuple(n_prior_words)
    # Filter series for n-grams that start with the key
    candidates = n_gram_series[n_gram_series.index.map(lambda x: x[:len(key)] == key)]
    if not candidates.empty:
        next_ngram = candidates.idxmax()
        return next_ngram[-1]  # Return the predicted next word (last element)
    else:
        return None

def finish_method(data: pd.Series, n_prior_words, max_iterations=100):
    method = list(n_prior_words)
    iterations = 0
    while iterations < max_iterations:
        next_word_pred = next_word(data, method[-len(n_prior_words):])
        if next_word_pred is None:
            break
        if next_word_pred == "<end>":
            method.append(next_word_pred)
            break
        method.append(next_word_pred)
        iterations += 1
    return method

# --- N-gram Model Building Function ---
def build_ngram(n, train_dir):
    ngram_counts = defaultdict(int)  # Store counts of n-grams
    total_ngrams = 0  # Total n-grams processed
    files_processed = 0  # Limit to 2 files

    for file in os.listdir(train_dir):
        if files_processed >= 2:
            break
        if file.endswith(".csv"):
            file_path = os.path.join(train_dir, file)
            file_df = pd.read_csv(file_path)
            files_processed += 1

            for _, row in file_df.iterrows():
                # Ensure 'Tokens' column exists and is not empty
                if 'Tokens' not in row or pd.isnull(row['Tokens']):
                    continue
                try:
                    tokens = ast.literal_eval(row['Tokens'])
                except (ValueError, SyntaxError) as e:
                    print(f"Skipping row due to parse error: {row['Tokens']} - {e}")
                    continue

                if len(tokens) < n:
                    continue

                # Generate n-grams from tokens and count them
                for ngram_tuple in ngrams(tokens, n):
                    ngram_counts[ngram_tuple] += 1
                    total_ngrams += 1

    if total_ngrams == 0:
        print("Warning: No n-grams found in the training data.")
        return pd.Series()  # Return empty series if nothing found

    # Convert counts to probabilities (flat model: key = n-gram tuple, value = probability)
    ngram_model = {ngram: count / total_ngrams for ngram, count in ngram_counts.items()}
    print(ngram_model)
    return pd.Series(ngram_model)

# --- Perplexity Computation Function ---
def perplexity(n_gram_model, test_dir, n):
    total_log_prob = 0
    total_tokens = 0
    files_processed = 0  # Limit to 2 files

    for file in os.listdir(test_dir):
        if files_processed >= 2:
            break
        if file.endswith(".csv"):
            file_path = os.path.join(test_dir, file)
            file_df = pd.read_csv(file_path)
            files_processed += 1

            for _, row in file_df.iterrows():
                try:
                    tokens = ast.literal_eval(row['Tokens'])
                except (ValueError, SyntaxError):
                    continue

                if len(tokens) < n:
                    continue

                log_prob_sum = 0
                # For a flat n-gram model, iterate from index (n-1) to end
                for i in range(n - 1, len(tokens)):
                    # Form the n-gram: for bigrams (n=2), this gives (tokens[i-1], tokens[i])
                    ngram_tuple = tuple(tokens[i - (n - 1): i + 1])
                    word_prob = n_gram_model.get(ngram_tuple, 1e-5)
                    log_prob_sum += np.log(word_prob)

                total_log_prob += log_prob_sum
                total_tokens += (len(tokens) - (n - 1))

    if total_tokens == 0:
        return float('inf')
    avg_log_prob = total_log_prob / total_tokens
    return np.exp(-avg_log_prob)

# --- Main Code Execution ---
if __name__ == "__main__":
    train_dir = "training data/"  # Directory with training CSVs
    test_dir = "testing data/"    # Directory with testing CSVs

    n = 2  # Set n=2 for bigrams; change to 3 for trigrams, etc.
    ngram_model = build_ngram(n, train_dir)

    # Print the bigram model (pandas Series)
    print("Bigram Model:")
    print(ngram_model)

    # Compute perplexity on the test set
    perp_value = perplexity(ngram_model, test_dir, n)
    print(f"Perplexity of the Bigram Model: {perp_value:.6f}")

    # Test the finish method to generate a predicted sequence
    n_prior_words = ['the']  # Starting context for prediction (for bigrams, one word context)
    predicted_sequence = finish_method(ngram_model, n_prior_words)
    print(f"Predicted sequence (Bigram): {predicted_sequence}")

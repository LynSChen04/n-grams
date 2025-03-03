import os
import pandas as pd
import numpy as np
import ast  # To safely evaluate Python literals
from nltk import ngrams
from collections import defaultdict

# --- Finish Method and Next Word Functions ---

# --- N-gram Model Building Function ---
def build_ngram(n, train_dir):
    method_series = extract_method(train_dir)  # Use extracted methods
    ngram_counts = defaultdict(int)
    total_ngrams = 0

    for tokens in method_series:
        if len(tokens) < n:
            continue
        for ngram_tuple in ngrams(tokens, n):
            ngram_counts[ngram_tuple] += 1
            total_ngrams += 1

    if total_ngrams == 0:
        print("Warning: No n-grams found in the training data.")
        return pd.Series()

    # Convert counts to probabilities
    ngram_model = {ngram: count / total_ngrams for ngram, count in ngram_counts.items()}
    return pd.Series(ngram_model)

def extract_method(test_dir):
    method_series = pd.Series([])
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
                print("test")
                method_series = pd.concat([method_series, pd.Series([tokens])], ignore_index=True)
    print(method_series)

    # Convert counts to probabilities (flat model: key = n-gram tuple, value = probability)
    print(tokens)
    return pd.Series(method_series)

# --- Perplexity Computation Function ---
def perplexity(n_gram_model, test_dir, n):
    method_series = extract_method(test_dir)  # Use extracted methods
    total_log_prob = 0
    total_tokens = 0

    for tokens in method_series:
        if len(tokens) < n:
            continue

        log_prob_sum = 0
        for i in range(n - 1, len(tokens)):
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

    n = 10  # Set n=2 for bigrams; change to 3 for trigrams, etc.
    ngram_model = build_ngram(n, train_dir)

    # Print the bigram model (pandas Series)
    print("Bigram Model:")
    print(ngram_model)

    # Compute perplexity on the test set
    perp_value = perplexity(ngram_model, test_dir, n)
    print(f"Perplexity of the Bigram Model: {perp_value:.6f}")


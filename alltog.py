import os
import re
import ast
import numpy as np
import pandas as pd
from collections import defaultdict
from pygments.lexers.jvm import JavaLexer
from pygments.lexers import get_lexer_by_name
from pygments.token import Token
from nltk.util import ngrams

# Initialize Java lexer
lexer = JavaLexer()

# --- Method Preprocessing Functions ---
def remove_duplicates(data):
    print("Removing duplicate methods...")
    before = len(data)
    data = data.drop_duplicates(subset="Method Code", keep='first')
    print(f"Removed {before - len(data)} duplicates.")
    return data

def filter_ascii_methods(data):
    print("Filtering non-ASCII methods...")
    before = len(data)
    data = data[data['Method Code'].apply(lambda x: all(ord(char) < 128 for char in x))]
    print(f"Removed {before - len(data)} non-ASCII methods.")
    return data

def remove_outliers(data, lower_percentile=5, upper_percentile=95):
    print("Removing outlier methods...")
    method_lengths = data['Method Code'].apply(len)
    lower_bound = method_lengths.quantile(lower_percentile / 100)
    upper_bound = method_lengths.quantile(upper_percentile / 100)
    before = len(data)
    data = data[(method_lengths >= lower_bound) & (method_lengths <= upper_bound)]
    print(f"Removed {before - len(data)} outliers.")
    return data

def remove_boilerplate_methods(data):
    print("Removing boilerplate methods...")
    boilerplate_patterns = [
        r"\bset[A-Z][a-zA-Z0-9_]*\(.*\)\s*{",
        r"\bget[A-Z][a-zA-Z0-9_]*\(.*\)\s*{"
    ]
    boilerplate_regex = re.compile("|".join(boilerplate_patterns))
    before = len(data)
    data = data[~data['Method Code'].apply(lambda x: bool(boilerplate_regex.search(x)))]
    print(f"Removed {before - len(data)} boilerplate methods.")
    return data

def remove_comments_from_dataframe(df, method_column, language="java"):
    print("Removing comments from methods...")
    def remove_comments(code):
        lexer = get_lexer_by_name(language)
        tokens = lexer.get_tokens(code)
        return ''.join(token[1] for token in tokens if not (lambda t: t[0] in Token.Comment)(token))
    
    df["Method Code No Comments"] = df[method_column].apply(remove_comments)
    print("Finished removing comments.")
    return df

# --- Tokenization ---
def tokenize_code(code):
    tokens = [t[1] for t in lexer.get_tokens(code)]
    tokens.append("<END>")
    return tokens

def process_files_in_folder(folder_path):
    """Preprocess Java methods in CSV files."""
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path)

            if "Method Code" not in data.columns:
                print(f"Skipping {filename} - Missing 'Method Code' column")
                continue  # Skip this file

            # Apply preprocessing steps
            data = remove_duplicates(data)
            data = filter_ascii_methods(data)
            data = remove_outliers(data)
            data = remove_boilerplate_methods(data)
            data = remove_comments_from_dataframe(data, 'Method Code', 'java')

            # Tokenize methods
            if "Method Code No Comments" in data.columns:
                data["Tokens"] = data["Method Code No Comments"].apply(tokenize_code)

            # Save back to CSV
            data.to_csv(file_path, index=False)

def extract_methods(folder_path):
    print("Extracting methods from files...")
    method_series = pd.Series([])
    files_processed = 0
    for filename in os.listdir(folder_path):
        if files_processed >= 2:
            break
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            print(f"Extracting from: {filename}")
            if "Tokens" in df.columns:
                for tokens in df["Tokens"]:
                    try:
                        token_list = ast.literal_eval(tokens)
                        method_series = pd.concat([method_series, pd.Series([token_list])], ignore_index=True)
                    except (ValueError, SyntaxError):
                        continue
            files_processed += 1
    print("Finished extracting methods.")
    return method_series

def build_ngram(n, train_dir):
    print(f"Building {n}-gram model...")
    method_series = extract_methods(train_dir)
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
    print(f"Total {n}-grams: {total_ngrams}")
    return pd.Series({ngram: count / total_ngrams for ngram, count in ngram_counts.items()})

def perplexity(n_gram_model, test_dir, n):
    print("Calculating perplexity...")
    method_series = extract_methods(test_dir)
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
    result = np.exp(-avg_log_prob)
    print(f"Perplexity: {result}")
    return result

# --- Run Processing ---
folder_path = "data/"
process_files_in_folder(folder_path)
n = 2  # Bigram model
train_dir = "training data"  # Folder containing training CSV files
test_dir = "testing data"  # Folder containing test CSV files

# Step 1: Process the training and test data
print("Processing training data...")
process_files_in_folder(train_dir)

print("Processing test data...")
process_files_in_folder(test_dir)

# Step 2: Build the n-gram model
print("Building bigram model...")
bigram_model = build_ngram(n, train_dir)

# Step 3: Compute perplexity
print("Calculating perplexity of the bigram model...")
bigram_perplexity = perplexity(bigram_model, test_dir, n)

# Step 4: Print results
print(f"Bigram Model Perplexity: {bigram_perplexity}")
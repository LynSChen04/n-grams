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
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Vocabulary
import numpy as np
import json

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
    tokens = [t[1] for t in lexer.get_tokens(code) if t[0] != Token.Text and t[1] != ' ']
    tokens.append("<END>")
    return tokens

def process_files_in_folder(folder_path):
    """Preprocess Java methods in CSV files."""
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {filename}")  # Print the filename being processed
            try:
                data = pd.read_csv(file_path)
                if "Method Code" not in data.columns:
                    print(f"Skipping {filename} - Missing 'Method Code' column")
                    continue  # Skip this file if the column is missing
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
            except Exception as e:
                print(f"Error processing {filename}: {e}")  # Print the error if it occurs

def extract_methods(folder_path):
    print("Extracting methods from files...")
    method_series = pd.Series([])
    files_processed = 0
    for filename in os.listdir(folder_path):
        if files_processed >= 25 and folder_path == "testing data":
            break
        if files_processed >= 50 and folder_path == "training data":
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
    n = n + 1
    ngram_counts = defaultdict(int)
    total_ngrams = 0
    for tokens in train_dir:
        if len(tokens) < n:
            continue
        for ngram_tuple in ngrams(tokens, n):
            ngram_counts[ngram_tuple] += 1
            total_ngrams += 1
    if total_ngrams == 0:
        print("Warning: No n-grams found in the training data.")
        return {}
    print(f"Total {n}-grams: {total_ngrams}")
    return {ngram: count for ngram, count in ngram_counts.items()}

def perplexity(ngram_counts, test_data, n, alpha=0.01):  # Start with lower alpha
    print("Calculating perplexity...")
    n = n + 1
    total_ngrams = sum(ngram_counts.values())
    vocab_size = len(ngram_counts) + 1  # +1 for unseen n-grams
    total_log_prob = 0
    total_tokens = 0
    for tokens in test_data:
        log_prob_sum = 0
        for ngram in ngrams(tokens, n, pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>"):
            count = ngram_counts.get(ngram, 0)
            smoothed_prob = (count + alpha) / (total_ngrams + alpha * vocab_size)
            smoothed_prob = max(smoothed_prob, 1e-10)  # Prevent too small values
            log_prob_sum += np.log(smoothed_prob)
        total_log_prob += log_prob_sum
        total_tokens += len(tokens)
    if total_tokens == 0:
        return float('inf')
    avg_log_prob = total_log_prob / total_tokens
    result = np.exp(-avg_log_prob)
    print(f"Perplexity: {result}")
    return result

def predict_tokens(file_name, n, ngram_model):
    print(f"Predicting tokens for file: {file_name}")
    predictions = {}

    try:
        data = pd.read_csv(file_name)
        if "Tokens" not in data.columns:
            print(f"Skipping {file_name} - Missing 'Tokens' column")
            return

        for index, row in data.iterrows():
            tokens = ast.literal_eval(row["Tokens"])
            if len(tokens) < n:
                continue

            current_ngram = tuple(tokens[:n])
            predicted_tokens = []

            while len(predicted_tokens) < 100:
                next_token_counts = {k[-1]: v for k, v in ngram_model.items() if k[:-1] == current_ngram}
                total_count = sum(next_token_counts.values())
                if not next_token_counts:
                    print(f"No next tokens found for n-gram: {current_ngram}")
                    break
                next_token_probs = {token: count / total_count for token, count in next_token_counts.items()}
                next_token = max(next_token_probs, key=next_token_probs.get)

                element = (next_token, next_token_probs[next_token])
                predicted_tokens.append(element)

                current_ngram = current_ngram[1:] + (next_token,)
                if next_token == "<END>":
                    break

            predictions[index] = predicted_tokens

        with open(f"results.json", "w") as json_file:
            json.dump(predictions, json_file, indent=4)

        print(f"Predictions saved to results.json")

    except Exception as e:
        print(f"Error predicting tokens for {file_name}: {e}")

"""# --- Run Processing ---


folder_path = "data/"


process_files_in_folder(folder_path)"""


n = 2 # Bigram model


train_data = extract_methods("training data")


test_data = extract_methods("testing data")





"""# Step 1: Process the training and test data


print("Processing training data...")


process_files_in_folder("training data")





print("Processing test data...")


process_files_in_folder("testing data")"""





# Step 2: Build the n-gram model


print("Building bigram model...")


ngram_model = build_ngram(n, train_data)


print(ngram_model)


# Step 3: Compute perplexity


print("Calculating perplexity of the bigram model...")


bigram_perplexity = perplexity(ngram_model, test_data, n)





# Step 4: Print results


print(f"Bigram Model Perplexity: {bigram_perplexity}")


# Step 5: Predict tokens (Use N-gram model)

predict_tokens("sample_tests/test_selection.csv", n, ngram_model)
import os
import pandas as pd
from collections import defaultdict
import numpy as np
import ast  # To safely evaluate the string representation of lists

# Function to build the unigram model
def build_unigram_model(train_dir):
    unigram_counts = defaultdict(int)  # Stores the counts of unigrams
    files_processed = 0  # To keep track of how many files have been processed

    for file in os.listdir(train_dir):
        file_name = os.fsdecode(file)
        if file_name.endswith(".csv") and files_processed < 2:  # Limit to first 2 files
            file_path = os.path.join(train_dir, file_name)
            file_df = pd.read_csv(file_path)

            print(f"Reading file: {file_path}")

            for _, row in file_df.iterrows():
                try:
                    # Convert the string representation of a list to an actual list
                    tokens = ast.literal_eval(row['Tokens'])  
                    print(f"Tokens: {tokens}")  # Debug print
                except (ValueError, SyntaxError):
                    print(f"Skipping row due to invalid token data: {row['Tokens']}")
                    continue  # Skip rows with invalid token data

                # Update the unigram counts for each token
                for token in tokens:
                    unigram_counts[token] += 1

            files_processed += 1
            if files_processed >= 2:  # Stop after 2 files
                break

    # Convert counts to probabilities
    total_tokens = sum(unigram_counts.values())
    unigram_model = {token: count / total_tokens for token, count in unigram_counts.items()}

    return unigram_model

# Function to compute perplexity of the unigram model
def perplexity(unigram_model, test_dir):
    total_log_prob = 0
    total_tokens = 0
    files_processed = 0  # To keep track of how many files have been processed

    for file in os.listdir(test_dir):
        file_name = os.fsdecode(file)
        if file_name.endswith(".csv") and files_processed < 2:  # Limit to first 2 files
            file_path = os.path.join(test_dir, file_name)
            file_df = pd.read_csv(file_path)

            print(f"Testing file: {file_path}")

            for index, row in file_df.iterrows():
                try:
                    # Convert the string representation of a list to an actual list
                    tokens = ast.literal_eval(row['Tokens'])
                    print(f"Tokens: {tokens}")  # Debug print
                except (ValueError, SyntaxError):
                    print(f"Skipping row due to invalid token data: {row['Tokens']}")
                    continue  # Skip rows with invalid token data

                log_prob_sum = 0
                for token in tokens:
                    # Get the probability for the token from the unigram model
                    token_prob = unigram_model.get(token, 1e-5)  # If token is missing, give it a small probability
                    log_prob_sum += np.log(token_prob)

                total_log_prob += log_prob_sum
                total_tokens += len(tokens)

            files_processed += 1
            if files_processed >= 2:  # Stop after 2 files
                break

    if total_tokens == 0:
        return float('inf')  # Return infinite perplexity if no tokens found

    avg_log_prob = total_log_prob / total_tokens
    return np.exp(-avg_log_prob)

# Function to print out the unigram model
def print_unigram_model(unigram_model):
    if not unigram_model:
        print("Unigram model is empty!")
    else:
        for token, prob in unigram_model.items():
            print(f"Token: {token}, Probability: {prob:.6f}")

# Main code execution
if __name__ == "__main__":
    train_dir = "training data/"
    test_dir = "testing data/"  # Assuming you have a test directory

    # Build unigram model
    unigram_model = build_unigram_model(train_dir)

    # Print unigram model
    print("Unigram Model:")
    print_unigram_model(unigram_model)

    # Compute perplexity of the model on the test set
    perplexity_value = perplexity(unigram_model, test_dir)
    print(f"Perplexity of the Unigram Model: {perplexity_value:.6f}")

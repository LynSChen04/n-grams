import os
import math
from ngram import next_word, finish_method
import numpy as np

import pandas as pd

def perplexity(n_gram_model, test_dir, n):
    total_log_prob = 0
    total_tokens = 0

    for file in os.listdir(test_dir):
        file_name = os.fsdecode(file)
        if file_name.endswith(".csv"):
            file_path = os.path.join(test_dir, file_name)
            file_df = pd.read_csv(file_path)

            for index, row in file_df.iterrows():
                try:
                    tokens = json.loads(row['Tokens'])  # Use json instead of eval for safety
                except json.JSONDecodeError:
                    continue  # Skip rows with invalid token data
                
                if len(tokens) < n:
                    continue  # Skip if not enough tokens for n-gram context
                
                log_prob_sum = 0
                for i in range(n, len(tokens)):  
                    context = tuple(tokens[i - n:i])  
                    next_word = tokens[i]

                    # Use dictionary get() to avoid redundant checks
                    word_prob = n_gram_model.get(context, {}).get(next_word, 1e-5)
                    log_prob_sum += np.log(word_prob)

                total_log_prob += log_prob_sum
                total_tokens += max(1, len(tokens) - n)  

    if total_tokens == 0:
        return float('inf')  

    avg_log_prob = total_log_prob / total_tokens
    return np.exp(-avg_log_prob)  


# main code 
# intake a corpus from a txt file on the command line
# preprocess the method and make it ready for training
# for n = 1-10, train each possible n-gram model
# keep only the most accurate model, 

# use general code to produce accuracy of 1-k n-grams, and pick the one with the best accuracy.
import os
import math

#N-grams pseudocode (creation of probability model)

#tokenize the corpus

#creating the n-gram model (pseudocode for just n)

#generate_ngram(corpus, N):
#	tokens = array of tokens in corpus
#	result = []
#	for every consecutive n tokens:
#		add the n consecutive tokens to result
#	return result

#ngram = generate_ngram(corpus, N)

#build a n-gram model using the ngram - may use lambda method?

#build a dictionary from ngram
#in the dictionary count the number of repeated occurrences of #token 1 - token n-1, and for each repetition add one to #dictionary entry dictionary[token 1 - token n-1][token n]
#transform the dictionary of raw counts / frequencies to #probabilities by:
#	for token 1 to token n-1 in dictionary:
#		occurrences = sum of dictionary[token 1 to token n-1]
#		values
#		for token n in dictionary[token 1 to token n-1]:
#			dictionary[token 1 to n-1][token n] /=
#occurrences

#function for predicting the next token
#predict_token(token 1 - token n - 1):
#	next_token = dictionary[token 1 - token n-1]
#	if next_token is maximum value from the different entries
#above:
#	return next_token
#else find maximum value and return
#else end prediction or return “no prediction available”



import pandas as pd

def next_word(n_gram_series, n_prior_words):
    """
    Predicts the next word given n preceding words from a Pandas Series.

    Args:
        n_gram_series (pd.Series): A Series where:
            - Index = n-gram tuple (words).
            - Values = Probability or frequency of the next word.
        n_prior_words (list): The preceding words (n-1 words).

    Returns:
        str or None: The predicted next word based on the probabilities.
    """
    key = tuple(n_prior_words)  # Convert list to tuple to match the Series index
    
    # Look up the probability of the next word in the Series
    candidates = n_gram_series[n_gram_series.index.str.startswith(str(key))]

    if not candidates.empty:
        # Select the next word with the highest probability/frequency
        next_word = candidates.idxmax()  # Get the index with the max value
        return next_word[-1]  # Return the actual next word
    else:
        return None




def perplexity(n_gram_series):
    """
    Calculate the perplexity of the n-gram model using a Pandas Series of probabilities .

    Args:
        n_gram_series (pd.Series): A Series where:
            - Index = n-gram tuple (words).
            - Values = Probability of the n-gram occurring.
        test_sentences (list): List of test sentences (strings).

    Returns:
        float: The perplexity of the model.
    """
    log_prob_sum = 0
    total_predictions = 0

    if total_predictions == 0:
        return float('inf')  # Avoid division by zero

    avg_log_prob = log_prob_sum / total_predictions
    return np.exp(-avg_log_prob)


# main code 
# intake a corpus from a txt file on the command line
# preprocess the method and make it ready for training
# for n = 1-10, train each possible n-gram model
# keep only the most accurate model, 

# use general code to produce accuracy of 1-k n-grams, and pick the one with the best accuracy.
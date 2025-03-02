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





#Evaluation method: 

def next_word(data, n_prior_words):
    """
    Predicts the next word given n preceding words.

    Args:
        data (dict): The n-gram model data.
        n_prior_words (list): The preceding words.

    Returns:
        str: The predicted next word.
    """
    key = tuple(n_prior_words)
    if key in data:
        return max(data[key], key=data[key].get)
    else:
        return None

def accuracy(test_data, probabilities):
    """
    Calculate the accuracy of the n-gram model.

    Args:
        test_data (list): The test data.
        probabilities (dict): The n-gram model probabilities.

    Returns:
        float: The accuracy of the model.
    """
    correct_predictions = 0
    total_predictions = 0

    for method in test_data:
        words = method.split()
        for i in range(len(words) - 1):
            n_prior_words = words[:i+1]
            predicted_word = next_word(probabilities, n_prior_words)
            if predicted_word == words[i+1]:
                correct_predictions += 1
            total_predictions += 1

    return (correct_predictions / total_predictions) * 100
    
def perplexity(test_data, probabilities):
    """
    Calculate the perplexity of the n-gram model.
    
    Args:
        test_data (list): The test data.
        probabilities (dict): The n-gram model probabilities.
    
    Returns:
        float: The perplexity of the model.
    """
    log_prob_sum = 0
    total_predictions = 0
        
    for method in test_data:
        words = method.split()
        for i in range(len(words) - 1):
            n_prior_words = words[:i+1]
            key = tuple(n_prior_words)
            next_word = words[i+1]
            if key in probabilities and next_word in probabilities[key]:
                prob = probabilities[key][next_word]
            else:
                prob = 1e-6  # Smoothing for unseen words
            log_prob_sum += math.log(prob)
            total_predictions += 1
    
    avg_log_prob = log_prob_sum / total_predictions
    perplexity = math.exp(-avg_log_prob)
    return perplexity


# main code 
# intake a corpus from a txt file on the command line
# preprocess the method and make it ready for training
# for n = 1-10, train each possible n-gram model
# keep only the most accurate model, 

# use general code to produce accuracy of 1-k n-grams, and pick the one with the best accuracy.
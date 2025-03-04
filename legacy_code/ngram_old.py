import os
import pandas as pd
from nltk import ngrams

train_dir = os.fsencode("training data/")
ngram = pd.Series

def build_ngram(n):
    for file in os.listdir(train_dir):
        file_name = os.fsdecode(file)
        if file_name.endswith(".csv"):
            file_df = pd.read_csv("./training data/"+file_name)
            
            ngram.add(pd.Series(ngrams(file_df, n)).valueCounts())

    print(ngram)
    return ngram

def next_word(n_gram_series, n_prior_words):
    key = tuple(n_prior_words)
    candidates = n_gram_series[n_gram_series.index.map(lambda x: x[:len(key)] == key)]
    if not candidates.empty:
        next_ngram = candidates.idxmax()
        print(next_ngram)
        return next_ngram[-1]
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

#using assigned training data (dictionary), test the accuracy of the n-gram #model, returns accuracy as value out of 100


#main code 

#intake a corpus from a txt file on the command line
#preprocess the method and make it ready for training
#for n = 1-10, train each possible n-gram model
#keep only the most accurate model, 


#use general code to produce accuracy of 1-k n-grams, and pick #the one with the best accuracy. 
ngram_data = {
    ('the', 'quick'): 0.5,
    ('quick', 'brown'): 0.4,
    ('brown', 'fox'): 0.7,
    ('fox', 'jumps'): 0.6,
    ('jumps', 'over'): 0.8,
    ('over', 'the'): 0.3,
    ('the', 'lazy'): 0.5,
    ('lazy', 'dog'): 0.9,
    ('dog', '<end>'): 1.0
}

# Convert it to a Pandas Series
n_gram_series = pd.Series(ngram_data)

ngram_data = {
    ('the', 'quick'): 0.5,
    ('quick', 'brown'): 0.4,
    ('brown', 'fox'): 0.7,
    ('fox', 'jumps'): 0.6,
    ('jumps', 'over'): 0.8,
    ('over', 'a'): 0.3,
    ('a', 'lazy'): 0.2,
    ('lazy', 'dog'): 0.9,
    ('dog', '<end>'): 1.0
}

n_gram_series = pd.Series(ngram_data)
n_prior_words = ['the']
predicted_sequence = finish_method(n_gram_series, n_prior_words)
print(predicted_sequence)


trigram_data = {
    ('I', 'love', 'python'): 0.4,
    ('love', 'python', 'programming'): 0.6,
    ('python', 'programming', 'in'): 0.8,
    ('programming', 'in', 'a'): 0.5,
    ('in', 'a', 'fun'): 0.7,
    ('a', 'fun', 'way'): 0.9,
    ('fun', 'way', '<end>'): 1.0
}

trigram_series = pd.Series(trigram_data)
n_prior_words_trigram = ['I', 'love']
predicted_sequence_trigram = finish_method(trigram_series, n_prior_words_trigram)
print("Trigram test:", predicted_sequence_trigram)
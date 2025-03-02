import os
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
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


#ngram = pd.Series(ngrams(corpus, N)).valueCounts()

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

def finish_method(data, n_prior_words):
    method = list(n_prior_words)
    while method[-1] != "":
        next_word = predict_token(data, method)
        method.append(next_word)
    return method #returns as a list

#predicts the next word given n preceding words. Outputs #expected next word

def accuracy(data, n_val):
    accuracy = []
    for i in data["evaluation methods"]: #assuming i is a list of tokens for that method
        predicted_method = finish_method(data, i[:n_val])
        actual_method = i
        blue_score = sentence_bleu([actual_method], predicted_method)
        accuracy.append(blue_score)
        return sum(accuracy)/len

#using assigned training data (dictionary), test the accuracy of the n-gram #model, returns accuracy as value out of 100


#main code 

#intake a corpus from a txt file on the command line
#preprocess the method and make it ready for training
#for n = 1-10, train each possible n-gram model
#keep only the most accurate model, 


#use general code to produce accuracy of 1-k n-grams, and pick #the one with the best accuracy. 

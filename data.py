import pandas as pd
from pydriller import Repository
import os
import csv
import javalang
from javalang.parse import parse
from javalang.tree import MethodDeclaration 

df_res = pd.read_csv('results.csv')

repoList = []
for idx, row in df_res.iterrows():
    repoList.append("https://www.github.com/{}".format(row['name']))
print(repoList[0:5])

def extract_methods_from_java(code):
    """
    Extract methods from Java source code using javalang parser.

    Args:
        code (str): The Java source code.

    Returns:
        list: A list of tuples containing method names and their full source code.
    """
    methods = []
    try:
        tree = javalang.parse.parse(code)
        lines = code.splitlines()
        for _, node in tree.filter(javalang.tree.MethodDeclaration):
            method_name = node.name

            start_line = node.position.line - 1
            end_line = None

            if node.body:
                last_statement = node.body[-1]
                if hasattr(last_statement, 'position') and last_statement.position:
                    end_line = last_statement.position.line
            
            if end_line:
                method_code = "\n".join(lines[start_line:end_line+1])
            else:
                method_code = "\n".join(lines[start_line:])
            
            methods.append((method_name, method_code))
    
    except Exception as e:
        print(f"Error parsing Java code: {e}")

    return methods

def extract_methods_to_csv_from_master(repo_path, output_csv):
    """
    Extract methods from Java files in the master branch and save them in a CSV file.

    Args:
        repo_path (str): Path to the Git repository.
        output_csv (str): Path to the output CSV file.
    """
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Commit Hash', 'File Name', 'Method Name', 'Method Code', 'Commit Link'])

        branch_name = row['defaultBranch']
        for commit in Repository(repo_path, only_in_branch=branch_name).traverse_commits():
            print(f"Processing commit: {commit.hash}")

            for modified_file in commit.modified_files:
                if modified_file.filename.endswith(".java") and modified_file.source_code:
                    methods = extract_methods_from_java(modified_file.source_code)

                    for method_name, method_code in methods:
                        commit_link = f"{repo_path}/commit/{commit.hash}"
                        csv_writer.writerow([commit.hash, modified_file.filename, method_name, method_code, commit_link])
                    
                    print(f"Extracted methods from {modified_file.filename} in commit {commit.hash}")


def extract_methods_to_csv(repo_path, output_csv):
    """
    Extract methods from Java files in a repository and save them in a CSV file.

    Args:
        repo_path (str): Path to the Git repository.
        output_csv (str): Path to the output CSV file.
    """
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Branch Name", "Commit Hash", "File Name", "Method Name", "Method Code", "Commit Link"])

        branch_name = row['defaultBranch']
        for commit in Repository(repo_path, only_in_branch=branch_name).traverse_commits():
            print(f"Processing commit: {commit.hash}")

            for modified_file in commit.modified_files:
                if modified_file.filename.endswith(".java") and modified_file.source_code:
                    methods = extract_methods_from_java(modified_file.source_code)

                    for method_name, method_code in methods:
                        commit_link = f"{repo_path}/commit/{commit.hash}"
                        csv_writer.writerow([branch_name, commit.hash, modified_file.filename, method_name, method_code, commit_link])
                    
                    print(f"Extracted methods from {modified_file.filename} in commit {commit.hash}")

#problems in row 9 spring-cloud/spring-cloud-config
for repo in repoList[0:40]:
    fileNameToSave = ''.join(repo.split('github.com')[1:])
    fileNameToSave = fileNameToSave.replace('/', '_')

    output_csv_file = os.path.join("data", "extracted_methods_{}.csv".format(fileNameToSave))

    extract_methods_to_csv(repo, output_csv_file)

    print(repo)

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

#Def next_word(Dictionary)

#take n last tokens from given data set
#search dictionary for most predicted next word. 
#if next predicted token is nothing, stop function
#else rerun next_word
#returns fully completed line of code

#predicts the next word given n preceding words. Outputs #expected next word

#we need a token for the end of a line...


#Def accuracy(training data (dictionary/list?), probabilities (dictionary))
#(assuming training data is ready to use) for each line in the training data
#   call next_word to finish the sentence
#   put finished sentence in var
#   compare finished sentence to training data's sentence
#   store accuracy of sentence in list
#Add up all the accuracies in list, divide by len of list
# return accuracy

#using assigned training data (dictionary), test the accuracy of the n-gram #model, returns accuracy as value out of 100



#use general code to produce accuracy of 1-k n-grams, and pick #the one with the best accuracy. 

import pandas as pd
from pydriller import Repository
import os
import csv
import javalang
from javalang.parse import parse
from javalang.tree import MethodDeclaration

import re

from pygments.lexers.jvm import JavaLexer
from pygments.lexers import get_lexer_by_name
from pygments.token import Token

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

for repo in repoList[41:]:
    fileNameToSave = ''.join(repo.split('github.com')[1:])
    fileNameToSave = fileNameToSave.replace('/', '_')

    output_csv_file = os.path.join("data", "extracted_methods_{}.csv".format(fileNameToSave))

    extract_methods_to_csv(repo, output_csv_file)

    print(repo)

# Removal of Type 1 Clones (Exact clones with identical code apart from differences in whitespace, comments, and formatting)
def remove_duplicates(data):
    """Remove duplicate methods based on method content
        Almost Type-1 with the exception of comments
    """
    return data.drop_duplicates(subset="Method Code",keep = 'first')

def filter_ascii_methods(data):
    """Filter out methods that contain non-ASCII characters"""
    try:
        return data[data['Method Code'].apply(lambda x: all(ord(char) < 128 for char in x))]
    except KeyError:
        print("KeyError: 'Method Code' column not found.")
        return data

def remove_outliers(data, lower_percentile=5, upper_percentile=95):
    """Remove outliers based on method length"""
    try:
        method_lengths = data['Method Code'].apply(len)
        lower_bound = method_lengths.quantile(lower_percentile/100)
        upper_bound = method_lengths.quantile(upper_percentile/100)
        return data[(method_lengths >= lower_bound) & (method_lengths <= upper_bound)]
    except KeyError:
        print("KeyError: 'Method Code' column not found.")
        return data

#Tokenization with Pygments

def remove_boilerplate_methods(data):
    """Remove boilerplate methods like getters and setters"""
    try:
        boilerplate_patterns = [
            r"\bset[A-Z][a-zA-Z0-9_]*\(.*\)\s*{",
            r"\bget[A-Z][a-zA-Z0-9_]*\(.*\)\s*{"
        ]
        boilerplate_regex = re.compile("|".join(boilerplate_patterns))
        data = data[~data['Method Code'].apply(lambda x: bool(boilerplate_regex.search(x)))]
        return data
    except KeyError:
        print("KeyError: 'Method Code' column not found.")
        return data

def remove_comments_from_dataframe(df: pd.DataFrame, method_column: str, language: str) -> pd.DataFrame:
    """
    Remove comments from the methods in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the methods.
        method_column (str): Column name containing the raw Java methods.
        language (str): Programming language for the lexer (e.g., 'java').

    Returns:
        pd.DataFrame: Updated DataFrame with a new column 'Java Method No Comments'.
    """
    def remove_comments(code):
        lexer = get_lexer_by_name(language)
        tokens = lexer.get_tokens(code)
        clean_code = ''.join(token[1] for token in tokens if not (lambda t: t[0] in Token.Comment)(token))

        return clean_code
    try:
        df["Method Code No Comments"] = df[method_column].apply(remove_comments)
    except KeyError:
        print(f"KeyError: '{method_column}' column not found.")
    return df

def process_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path)
            
            print("Initial dataset size:", len(data))

            # Remove duplicates
            data = remove_duplicates(data)
            print("After removing duplicates:", len(data))
            
            # Filter out non-ASCII methods
            data = filter_ascii_methods(data)
            print("After filtering non-ASCII methods:", len(data))
            
            # Remove outliers
            data = remove_outliers(data)
            print("After removing outliers:", len(data))
            
            # Remove boilerplate methods
            data = remove_boilerplate_methods(data)
            print("After removing boilerplate methods:", len(data))
            
            # Remove comments from methods
            data = remove_comments_from_dataframe(data, 'Method Code', 'java')
            print("After removing comments:", len(data))

            print(data.head())
            data.to_csv(file_path, index=False)


# Example usage
folder_path = 'data/'
process_files_in_folder(folder_path)


lexer = JavaLexer()

def tokenize_code(code):
    tokens = [t[1] for t in lexer.get_tokens(code)]
    return tokens

def add_tokens_to_csv_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            try:
                df = pd.read_csv(filepath)
                if "Method Code No Comments" in df.columns:
                    df["Tokens"] = df["Method Code No Comments"].apply(tokenize_code)                    
                df.to_csv(filepath, index=False)
            except Exception as e:
                print(f"Error processing file {filepath}: {e}")


# Example usage
directory_path = 'data/'
add_tokens_to_csv_files(directory_path)
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

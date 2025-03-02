import pandas as pd
import os
import csv
import javalang
from javalang.parse import parse
from javalang.tree import MethodDeclaration
import re
from pygments.lexers.jvm import JavaLexer
from pygments.lexers import get_lexer_by_name
from pygments.token import Token

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

def process_methods_from_txt(input_txt, output_csv):
    """
    Process methods from a text file and save them in a CSV file.

    Args:
        input_txt (str): Path to the input text file.
        output_csv (str): Path to the output CSV file.
    """
    with open(input_txt, 'r', encoding='utf-8') as txtfile, open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Method Code'])

        for line in txtfile:
            method_code = line.strip()
            if method_code:
                csv_writer.writerow([method_code])

def remove_duplicates(data):
    """Remove duplicate methods based on method content
        Almost Type-1 with the exception of comments
    """
    return data.drop_duplicates(subset="Method Code", keep='first')

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
input_txt_file = 'methods.txt'
output_csv_file = 'methods.csv'
process_methods_from_txt(input_txt_file, output_csv_file)

folder_path = 'data/'
process_files_in_folder(folder_path)

directory_path = 'data/'
add_tokens_to_csv_files(directory_path)

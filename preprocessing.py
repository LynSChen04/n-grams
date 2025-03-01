import re
import pandas as pd

from pygments.lexers.jvm import JavaLexer
from pygments.lexers import get_lexer_by_name
from pygments.token import Token
import os

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
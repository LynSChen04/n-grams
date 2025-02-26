import re
import pandas as pd

from pygments.lexers.jvm import JavaLexer
from pygments.lexers import get_lexer_by_name
from pygments.token import Token

# Removal of Type 1 Clones (Exact clones with identical code apart from differences in whitespace, comments, and formatting)
def remove_duplicates(data):
    """Remove duplicate methods based on method content
        Almost Type-1 with the exception of comments
    """
    return data.drop_duplicates(subset="Method Java",keep = 'first')

def filter_ascii_methods(data):
    """Filter out methods that contain non-ASCII characters"""
    return data[data['Method Java'].apply(lambda x: all(ord(char) < 128 for char in x))]

def remove_outliers(data, lower_percentile=5, upper_percentile=95):
    """Remove outliers based on method length"""
    method_lengths = data['Method Java'].apply(len)
    lower_bound = method_lengths.quantile(lower_percentile/100)
    upper_bound = method_lengths.quantile(upper_percentile/100)
    return data[(method_lengths >= lower_bound) & (method_lengths <= upper_bound)]

#Tokenization with Pygments

def remove_boilerplate_methods(data):
    """Remove boilerplate methods like getters and setters"""
    boilerplate_patterns = [
        r"\bset[A-Z][a-zA-Z0-9_]*\(.*\)\s*{",
        r"\bset[A-Z][a-zA-Z0-9_]*\(.*\)\s*{"
    ]
    boilerplate_regex = re.compile("|".join(boilerplate_patterns))
    data = data[~data['Method Java'].apply(lambda x: bool(boilerplate_regex.search(x)))]
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
    
    df["Method Java No Comments"] = df[method_column].apply(remove_comments)
    return df


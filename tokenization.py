from pygments.lexers.jvm import JavaLexer
#from pygments.lexer import get_lexer_by_name
from pygments.token import Token
import os
import pandas as pd

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
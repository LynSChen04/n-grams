import pandas as pd
from pydriller import Repository
import os
import csv
import javalang
from javalang.parse import parse
from javalang.tree import MethodDeclaration 

df_res = pd.read_csv('ghs-repos.csv')

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
                last_statement = node.boyd[-1]
                if hasattr(last_statement, 'position') and last_statement.position:
                    end_line = last_statement.position.line
            
            if end_line:
                method_code = "\n".join(lines[start_line:end_line+1])
            else:
                method_code = "\n".join(lines[start_line:])
            
            methods.append((method_name, method_code))
    
    except Exception as e:
        print(f"Error parsing Java code: {e}")


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

        for commit in Repository(repo_path, only_in_branch='master').traverse_commits():
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

        branch_name = "master"
        for commit in Repository(repo_path, only_in_branch=branch_name).traverse_commits():
            print(f"Processing commit: {commit.hash}")

            for modified_file in commit.modified_files:
                if modified_file.filename.endswith(".java") and modified_file.source_code:
                    method = extract_methods_from_java(modified_file.source_code)

                    for method_name, method_code in methods:
                        commit_link = f"{repo_path}/commit/{commit.hash}"
                        csv_writer.writerow([branch_name, commit.hash, modified_file.filename, method_name, method_code, commit_link])
                    
                    print(f"Extracted methods from {modified_file.filename} in commit {commit.hash}")


for repo in repoList[0:1]:
    fileNameToSave = ''.join(repo.split('github.com')[1:])
    fileNameToSave = fileNameToSave.replace('/', '_')

    output_csv_file = "extracted_methods_{}.csv".format(fileNameToSave)

    extract_methods_to_csv_from_master(repo, output_csv_file)

    print(repo)
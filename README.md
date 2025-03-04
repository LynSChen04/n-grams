# GenAI for Software Development Assignment 1
Lynelle Chen, Ben Tremblay, Rowan Miller

* [1 Introduction](#1-introduction)  
* [2 Setup](#2-setup)
* [3 Run Model](#3-run-model)
* [4 Report](#4-report)

## **1. Introduction**  
We have developed a model to automatically complete a Java method given a starting set of code tokens. Our model uses the n-gram method to predict the next token in a sequence given (n - 1) preceding tokens. It is trained on a corpus of public GitHub repositories and stores the frequency of all sequences of n consecutive tokens. When predicting the next token in a sequence, it chooses the one with the highest probability in the training corpus. Our model was evaluated by being asked to automatically complete a method given only its first (n - 1) tokens.

## **2. Setup**  
This project is implemented in **Python 3** and is compatible with **macOS, Linux, and Windows**.  

Clone the repository to your workspace:  
```shell
~ $ git clone https://github.com/rvmiller0/n-grams.git
```
(2) Navigate into the repository:
```shell
~ $ cd n-grams
~/n-grams $
```
(3) Set up a virtual environment and activate it (optional):

For macOS/Linux:
```shell
~/n-grams $ python -m venv ./venv/
~/n-grams $ source venv/bin/activate
(venv) ~/n-grams $ 
```

Install the required dependencies:
`pip install -r requirements.txt`

When you're finished, use the following command to deactivate the virtual environment:
`(venv) $ deactivate`

## **3. Run Model**
To set the value of `n`, open `ngram.py` with your preferred text editor. On line **LINE**, edit the value of the variable `n`. Save and quit. The default value for `n` is 6.

To train, test, and evaluate the n-gram model, simply run the following:

`python ngram.py`

The files `results_student_model.json` and `results_teacher_model.json` will be generated. These will display the results of the model's evaluation on two reserved datasets.

## 4. Report
Our overall report is available in the file Assignment_Report.pdf.

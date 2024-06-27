# Bigram Language Model
This project implements a simple character-level language model using PyTorch. The model is trained on a dataset of names and can generate new names based on the learned patterns.

# Table of Contents
- [Overview]([#overview]) 
- [Requirements]([#requirements])
- [Installation and Usage]([#installation-and-usage])
- [Code Explanation]([#code-explanation])

# Overview
The character-level language model reads a list of names, processes them into sequences of character pairs(bigrams), and trains a simple neural network to predict the next character in a sequence. The model can then generate new names by sampling from the learned probability distributions.

# Requirements
- Python 3.x
- PyTorch

# Installation and Usage
- Create a virtual environment:
  ```
  python -m venv env
  ```
- Activate the virtual environment:
  ```
  .\env\Scripts\activate
  ```
- Install the Libraries:
  ```
  pip install -r requirements.txt
  ```
- Place your dataset of names in a text file named names.txt inside a data directory.
- Run:
  ```
  python bigram.py
  ```
- Deactivate the Environment:
  ```
  deactivate
  ```

# Code Explanation

**'prepare_data'**

This function reads the dataset, processes the characters, and creates the necessary mappings (stoi and itos). It also generates the input (xs) and output (ys) sequences for training.

**'train_model'**

This function trains a simple neural network with one hidden layer. It uses one-hot encoding for the input and the softmax function to calculate the probabilities for the output characters. The model is trained using gradient descent.

**'generate_text'**

This function generates new names by sampling from the learned probability distributions. It starts with the start-of-sequence token and continues sampling characters until the end-of-sequence token is generated.

**'main'**

The main function orchestrates the data preparation, model training, and text generation.



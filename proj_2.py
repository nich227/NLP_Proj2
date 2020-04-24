'''
Name: Kevin Chen
NetID: nkc160130
CS 6320
Due: 4/29/2020
Dr. Moldovan
Version: Python 3.8.0
'''

import os
import array
import re
import time
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import naive_bayes
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download nltk modules
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

# Token class


class Token:
    def __init__(self):
        self.token = ""
        self.position = 0
        self.sentence = ""
        self.lemma = ""
        self.post = ""
        self.representation = []
        self.tag = 0

    def __init__(self, token, tag):
        self.token = token
        self.position = 0
        self.sentence = ""
        self.lemma = ""
        self.post = ""
        self.representation = []
        self.tag = tag

    def __str__(self):
        return "Token: " + str(self.token) + "\nPosition: " + str(self.position) + "\nSentence: " + str(self.sentence) + "\nLemma: " + self.lemma + "\nPOS Tag: " + self.post + "\nRepresentation: " + str(self.representation) + "\nTag: " + str(self.tag)


# Reading in an input file to extract sentences, tokens and NER tags
def read_file(fileName, known_tokens=[], is_test=False):

    # Opening the file
    with open(fileName, 'r') as file:
        token_re = re.compile("^.*\t.*$")

        token_list = []
        cur_tag_key = 1
        tag_dict = {}
        num_sentences = 0
        prev_line = ""

        # Read in tokens here until the sentence ends, then put in token_list
        token_tmp = []

        # Put the sentence containing the token here
        token_sentence = []

        # Reading line by line
        for line in [str.strip() for str in file.readlines()]:

            # This is a token and tag tuple (token, tag identifier)
            if token_re.match(line):

                # Handle out of vocabulary words for test data
                if is_test == True and line.casefold().split('\t')[0] not in known_tokens:
                    line = "UNK\tUNK"

                # Add new tag to known tags
                if line.split('\t')[1] not in [tuple[1] for tuple in list(tag_dict.items())]:
                    tag_dict.update({cur_tag_key: line.split('\t')[1]})
                    cur_tag_key += 1

                # Add token to list of tokens
                token_tmp.append(Token(line.casefold().split('\t')[0], list(tag_dict.keys())[
                                 list(tag_dict.values()).index(line.split('\t')[1])]))

                # Add token to the sentence
                token_sentence.append(line.casefold().split('\t')[0])

            # End of sentence (and previous line had a token)
            if(line == "") and token_re.match(prev_line):
                num_sentences += 1

                # Add sentence data to all tokens in current sentence
                for token, pos in zip(token_tmp, range(len(token_sentence))):
                    token.sentence = token_sentence
                    token.position = pos

                token_list.extend(token_tmp)
                token_tmp = []
                token_sentence = []

            # Note what the previous line was
            prev_line = line

    # Return back tokens, tag encodings and sentence counts
    return (token_list, tag_dict, num_sentences)

# Get POS tags, adds corresponding POS tags and one-hot vector for POS tags


def pos_process(token_list):

    # Iterate through all sentences within the tokens
    cur = 0
    while cur < len(token_list):
        pos_sentence = nltk.pos_tag(token_list[cur].sentence)

        # Put POS tags into all tokens within this sentence
        for pos in [pos[1] for pos in pos_sentence]:
            token_list[cur].post = pos
            cur += 1

    # Create one-hot vector for each token
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False, dtype=np.uint8)
    integer_encoded = label_encoder.fit_transform(
        np.array([token.post for token in token_list]))
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    # Put all one hot encoded vectors with their respective token
    for onehot, token in zip(onehot_encoded, token_list):
        token.representation = onehot

# Get lemmas, adds corresponding lemma and append one-hot vector for lemma


def lemma_process(token_list, is_test=False):

    lemmatizer = WordNetLemmatizer()

    # Convert all POS tags to WordNet tag
    wntag_list = []
    for token in token_list:
        # Unknown token and is test
        if is_test == True and token.token == "UNK":
            token.lemma = "UNK"
        # Adjective
        elif token.post.startswith('J'):
            wntag_list.append(wordnet.ADJ)
        # Verb
        elif token.post.startswith('V'):
            wntag_list.append(wordnet.VERB)
        # Noun
        elif token.post.startswith('V'):
            wntag_list.append(wordnet.NOUN)
        # Adverb
        elif token.post.startswith('R'):
            wntag_list.append(wordnet.ADV)
        else:
            wntag_list.append(None)

    # Get lemma for each WordNet tag
    for wn_tag, token in zip(wntag_list, token_list):

        # Unknown token and is test
        if is_test == True and token.token == "UNK":
            continue

        if wn_tag is None:
            token.lemma = token.token
        else:
            token.lemma = lemmatizer.lemmatize(token.token, wn_tag)

    # Create one-hot vector for each token
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False, dtype=np.uint8)
    integer_encoded = label_encoder.fit_transform(
        np.array([token.lemma for token in token_list]))
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    # Put all one hot encoded vectors with their respective token (append to existing POS one-hot)
    for onehot, token in zip(onehot_encoded, token_list):
        token.representation = np.concatenate(
            (token.representation, onehot), axis=None)


# Driver of the program
if __name__ == "__main__":
    start_time = time.time()

    print("WARNING: This application requires the use of a large amount of RAM, at least approx. 80 GB, to run properly. You have been warned.")
    print()

    # Get sentences, tokens and tags for train data and get pos and lemma vectors
    train_token_list, train_tag_dict, train_num_sentences = read_file(
        "modified_train.txt")
    pos_process(train_token_list)
    lemma_process(train_token_list)

    # Train SVM Model
    model = svm.LinearSVC()
    model.fit([token.representation for token in train_token_list],
              [token.tag for token in train_token_list])

    # Get sentences, tokens and tags for train data and get pos and lemma vectors
    test_token_list, test_tag_dict, test_num_sentences = read_file(
        "modified_test.txt")
    pos_process(test_token_list)
    lemma_process(test_token_list, True)

    # Test SVM Model
    predictions = model.predict(
        [token.representation for token in test_token_list])

    # Report performance
    y_test = [token.tag for token in test_token_list]
    print('---------------')
    print('| Performance |')
    print('---------------')
    print('Accuracy score: ', accuracy_score(y_test, predictions))
    print('Precision score: ', precision_score(
        y_test, predictions, average='weighted'))
    print('Recall score: ', recall_score(
        y_test, predictions, average='weighted'))
    print('F1 score: ', f1_score(y_test, predictions, average='weighted'))

    # End of program
    print('-----\n', 'Project 2 took', round(time.time() -
                                             start_time, 4), 'seconds to complete.')

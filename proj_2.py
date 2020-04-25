'''
Name: Kevin Chen
NetID: nkc160130
CS 6320
Due: 4/29/2020
Dr. Moldovan
Version: Python 3.8.0
'''

import os
import re
import time
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download nltk modules
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

# Token class


class Token:
    def __init__(self):
        self.token = ""
        self.sentence = ""
        self.lemma = ""
        self.post = ""
        self.representation = []
        self.tag = 0

    def __init__(self, token, tag):
        self.token = token
        self.sentence = ""
        self.lemma = ""
        self.post = ""
        self.representation = []
        self.tag = tag

# Reading in an input file to extract sentences, tokens and NER tags


def read_file(fileName):

    # Opening the file
    with open(fileName, 'r') as file:
        token_re = re.compile("^.*\t.*$")

        token_list = []
        unique_token_list = []
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

                # Add new tag to known tags
                if line.split('\t')[1] not in [tuple[1] for tuple in list(tag_dict.items())]:
                    tag_dict.update({cur_tag_key: line.split('\t')[1]})
                    cur_tag_key += 1

                # Add token to list of tokens
                new_token = Token(line.casefold().split('\t')[0], list(tag_dict.keys())[
                    list(tag_dict.values()).index(line.split('\t')[1])])
                token_tmp.append(new_token)

                # Add token to the sentence
                token_sentence.append(new_token.token)
                
                # New token has been found 
                if new_token.token not in unique_token_list:
                    unique_token_list.append(new_token.token)

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

    # Return back tokens, tag encodings, sentence counts and unique token list
    return (token_list, tag_dict, num_sentences, unique_token_list)

# Get POS tags, adds corresponding POS tags and one-hot vector for POS tags


def pos_process(token_list, post_list=[], is_test=False):

    # POS tag list (for train)
    train_post = post_list

    # Iterate through all sentences within the tokens
    cur = 0
    while cur < len(token_list):
        pos_sentence = nltk.pos_tag(token_list[cur].sentence)

        # Put POS tags into all tokens within this sentence
        for pos in [pos[1] for pos in pos_sentence]:
            token_list[cur].post = pos

            # Found a train POS tag that's not in the list of tags
            if is_test == False and pos not in train_post:
                train_post.append(pos)

            # Found a test POS tag that's not in the train
            if is_test == True and pos not in post_list:
                token_list[cur].post = "UNK"

            cur += 1

    # Create one-hot vector for each token
    onehot_encoded = []
    for pos in [token.post for token in token_list]:
        onehot = np.zeros(len(train_post)+1, dtype=np.uint8)

        # If the POS tag is within the train POS tag list
        if is_test == False or pos in train_post:
            onehot[train_post.index(pos)] = 1
        else:
            onehot[-1] = 1

        # Add this one hot encoded vector to the list of one-hot encoded vectors
        onehot_encoded.append(onehot)

    # Put all one hot encoded vectors with their respective token
    for onehot, token in zip(onehot_encoded, token_list):
        token.representation = onehot

    return train_post if is_test == False else None

# Get lemmas, adds corresponding lemma and append one-hot vector for lemma and returns the train vocabulary (if train)


def lemma_process(token_list, lemma_list=[], is_test=False):

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Train vocab with a list of lemmas (for train data only)
    train_vocab = lemma_list

    # Convert all POS tags to WordNet tag
    wntag_list = []
    for token in token_list:
        # Adjective
        if token.post.startswith('J'):
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
        # No WordNet tag (just put original token as lemma)
        if wn_tag is None:
            token.lemma = token.token
        else:
            token.lemma = lemmatizer.lemmatize(token.token, wn_tag)

        # Add new lemma to vocabulary (if train)
        if is_test == False and token.lemma not in train_vocab:
            train_vocab.append(token.lemma)

        # Lemma is not in vocabulary (if test)
        if is_test == True and token.lemma not in train_vocab:
            token.lemma = "UNK"

    # Create one-hot vector for each token
    onehot_encoded = []
    for lemma in [token.lemma for token in token_list]:
        onehot = np.zeros(len(train_vocab)+1, dtype=np.uint8)

        # If the POS tag is within the train POS tag list
        if is_test == False or lemma in train_vocab:
            onehot[train_vocab.index(lemma)] = 1
        else:
            onehot[-1] = 1

        # Add this one hot encoded vector to the list of one-hot encoded vectors
        onehot_encoded.append(onehot)

    # Put all one hot encoded vectors with their respective token (append to existing POS one-hot)
    for onehot, token in zip(onehot_encoded, token_list):
        token.representation = np.concatenate(
            (token.representation, onehot), axis=None)

    return train_vocab if is_test == False else None


# Driver of the program
if __name__ == "__main__":
    start_time = time.time()

    print("WARNING: This application requires the use of a large amount of RAM, at least approx. 80 GB, to run properly. You have been warned.")
    print()

    # Get sentences, tokens and tags for train data and get pos and lemma vectors
    train_token_list, train_tag_dict, train_num_sentences, train_unique_tokens = read_file(
        "modified_train.txt")
    train_post = pos_process(train_token_list)
    train_vocab = lemma_process(train_token_list)

    # Train SVM Model
    model = svm.LinearSVC()
    model.fit([token.representation for token in train_token_list],
              [token.tag for token in train_token_list])

    # Get sentences, tokens and tags for test data and get pos and lemma vectors
    test_token_list, test_tag_dict, test_num_sentences, test_unique_tokens = read_file(
        "modified_test.txt")
    pos_process(test_token_list, train_post, True)
    lemma_process(test_token_list, train_vocab, True)
    
    # Reverse the train tag dict
    reverse_train_tag_dict = {}
    for tag in train_tag_dict:
        reverse_train_tag_dict.update({train_tag_dict[tag]: tag})

    
    # Test SVM Model
    predict_time = time.time()
    predictions = model.predict(
        [token.representation for token in test_token_list])
    
    # Correct predictions (remove BIO tag violations)
    entity_type = ""
    i = 0
    for prediction in predictions:
        tag = train_tag_dict[prediction]

        # If this is an O tag (resets entity_type)
        if tag == "O":
            entity_type = ""

        # Look for B (beginning tag)
        elif "B-" in tag:
            entity_type = tag.split("-")[1]

        # Entity type for an I tag and there is no matching B tag (it should be a standalone B tag)
        elif "I-" in tag and entity_type == "":
            predictions[i] = reverse_train_tag_dict["B-"+tag.split("-")[1]]

        # Entity type for an I tag does not match B tag (change to the B-tag type)
        elif "I-" in tag and tag.split("-")[1] != entity_type:
            predictions[i] = reverse_train_tag_dict["I-"+entity_type]

        i += 1
        
    predict_time = time.time() - predict_time

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
    
    print('Throughput (time): ', round(predict_time, 3), "seconds")
    print('Throughput (data): ', round(os.path.getsize("modified_test.txt") / (predict_time * 1000.0), 3), "kbps")

    # End of program
    print('-----\n', 'Project 2 took', round(time.time() -
                                             start_time, 3), 'seconds to complete.')

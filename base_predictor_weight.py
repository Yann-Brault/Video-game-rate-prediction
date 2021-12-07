"""
Base Predictor on data. Used the positive and negative weight of the meaning of words. 
"""

import pandas as pd
import unidecode
import re
import time

NEG_WORD: list[str]
POS_WORD: list[str]


def load_lexicon():
    with open('lexicon/neg_words_normalized.txt', 'r', encoding='utf-8') as f:
        data = f.read()
        neg_word = data.split('\n')
        f.close()

    with open('lexicon/pos_words_normalized.txt', 'r', encoding='utf-8') as f:
        data = f.read()
        pos_word = data.split('\n')
        f.close()
    
    return neg_word, pos_word

def remove_accents(text):
    return unidecode.unidecode(text)

NEG_WORD, POS_WORD = load_lexicon()

data = pd.read_csv('dataset/dataset_original.csv')
start = time.time_ns()

row = data['avis'][33]
row = remove_accents(row).lower()
row_filtered = row.replace('.', ' ').replace(',', '').replace('\r', '').replace(';', '').replace('\n', '')

row_decode = row_filtered.split(' ')

print(row_decode)


neg = 0
pos = 0

for word in row_decode:
    if word in NEG_WORD:
        neg+=1
    if word in POS_WORD:
        pos+=1

end = time.time_ns()
print((end - start)/10**9 )
print("neg count", neg)
print("pos count", pos)
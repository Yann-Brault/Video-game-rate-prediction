"""
Base Predictor on data. Used the positive and negative weight of the meaning of words. 
"""

import pandas as pd
import unidecode
import re
from tqdm import trange

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

def clean_str(avis):
    #print('\n\nAVIS : ', avis)
    if len(avis) > 0 or avis != None:
    

        avis = re.sub(',|;|\&|\#|\@|\%|\:|\>|\<|\(|\)|\{|\}|\=|\+|\-|\_|\[|\}|\^|\*|\!|\?|\/|\¨|\~|\\\|\§|\||[0-9]|\[|\]', '', avis)
        avis = avis.replace('.', ' ').replace('\t', '').replace('\r', '').replace('\n', '')
        avis = remove_accents(avis).lower()

    return avis

def count_neg_pos(avis_list):
    neg = 0
    pos = 0
    for word in avis_list:
        if word in NEG_WORD:
            neg+=1
        if word in POS_WORD:
            pos+=1
    return neg, pos





if __name__ == "__main__":

    NEG_WORD, POS_WORD = load_lexicon()
    data = pd.read_csv('dataset/dataset_original.csv')
    new_data = pd.DataFrame(data={'jeux': [], 'note': [], 'classe_bon_mauvais': [], 'negative_words': [], 'positive_words': [], 'avis': []})

    index = 0
    for i in trange(data.shape[0]):
        avis = data['avis'][i]

        if not pd.isna(avis) and type(avis) == str:
            avis = clean_str(avis)
            jeux = data['jeux'][i]
            note = int(data['note'][i])
            classe_bon_mauvais = 0
            if note > 12:
                classe_bon_mauvais = 1
            else:
                classe_bon_mauvais = 0

            avis_list = avis.split(' ')
            neg, pos = count_neg_pos(avis_list)

            new_data.loc[index] = [jeux, note, classe_bon_mauvais, neg, pos, avis]
            index += 1

    print(new_data.head(8))  
    new_data.to_csv('dataser_clean.csv')    
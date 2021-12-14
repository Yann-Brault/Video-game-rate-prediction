from spellchecker import SpellChecker
import pandas as pd
import re
from tqdm import trange

df = pd.read_csv("./dataset/refactored_dataset_clean.csv")


def replace_nan(df):
    for i in range(df.shape[0]):
        if pd.isna(df["avis"][i]):
            if df["note"][i] > 11: 
                df.at[i, "avis"] = "bon"
            else:
                df.at[i, "avis"] = "mauvais"
    return df


df = replace_nan(df)

spell = SpellChecker(language='fr')

for i in trange(df.shape[0]):
    
    avis = df.at[i, 'avis']
    avis_list = avis.split(" ")
    
    for word in avis_list:
        if word != " ":
            avis_list = spell.unknown(avis_list)
    
    new_list = []
    for word in avis_list:
        new_list.append(spell.correction(word))
    
    
    df.at[i, 'avis'] = ' '.join(new_list)

df.to_csv("dataset/refactored_dataset_clean.csv", index=False)
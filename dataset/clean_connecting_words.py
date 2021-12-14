import pandas as pd
import re
from tqdm import trange

df = pd.read_csv("./dataset/refactored_dataset.csv")

connecting_words = [
    "c'est", "ces", "ses", "s'est", "a", "de", "du", 
    "et", "le", "les", "un", "une", "pour", "sur", "etc", "est", "c",
    'la', "jeu", "que", "des", "en", "ce", "qu", "ca", "y", "j", "sa", "son",
    "au", "ai", "mon", "ma", "mes", "qui", "je", "tu", "il", "ils", "elles", "elle", "vous", "nous",
    "qu'il", "qu'elle", "qu'ils", "qu'elles", "qu'on",
    "on", "se", "par"]

def replace_nan(df):
    for i in range(df.shape[0]):
        if pd.isna(df["avis"][i]):
            if df["note"][i] > 11: 
                df.at[i, "avis"] = "bon"
            else:
                df.at[i, "avis"] = "mauvais"
    return df


df = replace_nan(df)

for i in trange(df.shape[0]):
    avis = df.at[i, 'avis']
    avis_list = avis.split(" ")
    
    new_list = []
    for word in avis_list:
        if word != " " and not word in connecting_words and len(word) > 1:
            new_list.append(word)
            
    df.at[i, 'avis'] = ' '.join(new_list)

df.to_csv("dataset/refactored_dataset_clean.csv", index=False)

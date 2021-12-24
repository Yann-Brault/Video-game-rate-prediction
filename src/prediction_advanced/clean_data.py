import re
from spellchecker import SpellChecker

import pandas as pd
from pandas.core.frame import DataFrame
from tqdm import trange
import unidecode


class CleanData:

    def __init__(self, max_words, df: DataFrame = None) -> None:
        self.max_words = max_words
        self.df = df
        self.unused_chars = ',|;|\&|\#|\@|\%|\:|\>|\<|\(|\)|\{|\}|\=|\+|\-|\_|\[|\}|\^|\*|\!|\?|\/|\¨|\~|\\\|\§|\||[0-9]|\[|\]|\"'
        self.connecting_words = [
            "c'est", "ces", "ses", "s'est", "a", "de", "du", 
            "et", "le", "les", "un", "une", "pour", "sur", "etc", "est", "c",
            'la', "jeu", "que", "des", "en", "ce", "qu", "ca", "y", "je", "sa", "son",
            "au", "ai", "mon", "ma", "mes", "qui", "je", "tu", "il", "ils", "elles", "elle", "vous", "nous",
            "qu'il", "qu'elle", "qu'ils", "qu'elles", "qu'on",
            "on", "se", "par"]
        self.spell = SpellChecker(language='fr')

    def correction_spelling(self, avis):
        """ 
        try to elimiminates the unknown word of a sentence, and 
        replacing it by a correct word. 
        """
        
        avis_list = avis.split(" ")
        
        bad = []
        for word in avis_list:
            if word != " ":
                bad = self.spell.unknown(avis_list)

        new_list = []
        for word in avis_list:
            if word in bad:
                new_list.append(self.spell.correction(word))
            else:
                new_list.append(word)
        
        return ' '.join(new_list)

    def replace_nan(self):
        """
        Replace nan by 'bon' or 'mauvais' in the dataframe.
        """

        for i in range(self.df.shape[0]):
            if pd.isna(self.df["avis"][i]):
                if self.df["note"][i] > 11: 
                    self.df.at[i, "avis"] = "bon"
                else:
                    self.df.at[i, "avis"] = "mauvais"
        return self.df

    def clean_str(self, avis):
        """
        Remove special characters from the string.
        """

        if len(avis) > 0 or avis != None:
            avis = re.sub(self.unused_chars, ' ', avis)
            avis = avis.replace('.', ' ').replace('\t', '').replace('\r', '').replace('\n', '')
            avis = unidecode.unidecode(avis).lower()
       
        return avis
    
    def clean_connecting_words(self, avis):
        avis_list = avis.split(" ")

        new_list = []
        for word in avis_list:
            if word != " " and not word in self.connecting_words and len(word) > 1:
                new_list.append(word)

        return ' '.join(new_list)

    def clean_avis(self, avis):
        avis = self.clean_str(avis)
        avis = self.clean_connecting_words(avis)
        avis = self.correction_spelling(avis)
        
        return avis

    def clean_dataset(self):
        """
        Main method call to prepare a text to be vectorized.
        """

        self.df = self.replace_nan()
        for i in trange(self.df.shape[0]):
            avis = self.df.at[i, 'avis']
            avis = self.clean_str(avis)
            avis = self.clean_connecting_words(avis)
            avis = self.correction_spelling(avis)
                
            self.df.at[i, 'avis'] = avis

    def filter_long_avis(self):
        """
        Filter the string with too many words.
        """

        to_drop = []
        for i in trange(self.df.shape[0]):
            avis = self.df['avis'][i]
            avis_list = avis.split()
            if len(avis_list) > self.max_words:
                to_drop.append(i)
        self.df.drop(to_drop, inplace=True)
        print(f"dropped {len(to_drop)} lines")

    def fix_repartition(self):
        """
        Fix the bad repartitions of the dataset, by removing randomly good advice.
        """

        d = self.df.groupby(['classe_bon_mauvais'], as_index=False).count()
        nb_bad = d['avis'][0]
        nb_good = d['avis'][1]
        print(nb_bad, nb_good)

        to_remove = nb_good - nb_bad
        
        while(to_remove > 0):
            row: DataFrame = self.df.sample()
            index = row.first_valid_index()
            print(f"{to_remove}")

            if row['classe_bon_mauvais'][index] == 1:
                self.df.drop(index, inplace=True)
                to_remove -= 1
        
        d = self.df.groupby(['classe_bon_mauvais'], as_index=False).count()
        nb_bad = d['avis'][0]
        nb_good = d['avis'][1]
        print("2", nb_bad, nb_good)

    def save_data(self, path):
        self.df.to_csv(path, index=False)

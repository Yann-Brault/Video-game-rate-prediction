import re
from spellchecker import SpellChecker

import pandas as pd
from pandas.core.frame import DataFrame
from tqdm import trange



class CleanData:

    def __init__(self, max_words, df: DataFrame = None) -> None:
        self.max_words = max_words
        self.df = df
        self.unused_chars = ',|;|\&|\#|\@|\%|\:|\>|\<|\(|\)|\{|\}|\=|\+|\_|\[|\}|\^|\*|\!|\?|\/|\¨|\~|\\\|\§|\||[0-9]|\[|\]|\"'
        self.connecting_words = [
            "c'est", "ces", "ses", "s'est", "a", "de", "du", 
            "et", "le", "les", "un", "une", "pour", "sur", "etc", "est", "c",
            'la', "jeu", "que", "des", "en", "ce", "qu", "ca", "y", "je", "sa", "son",
            "au", "ai", "mon", "ma", "mes", "qui", "je", "tu", "il", "ils", "elles", "elle", "vous", "nous",
            "qu'il", "qu'elle", "qu'ils", "qu'elles", "qu'on",
            "on", "se", "par"]
        self.urls = r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%|\-)*\b'
            
        self.spell = SpellChecker(language='fr')

    def correction_spelling(self, review):
        """ 
        try to elimiminates the unknown word of a sentence, and 
        replacing it by a correct word. 
        """
        
        review_list = review.split(" ")
        
        bad = []
        for word in review_list:
            if word != " ":
                bad = self.spell.unknown(review_list)

        new_list = []
        for word in review_list:
            if word in bad:
                new_list.append(self.spell.correction(word))
            else:
                new_list.append(word)
        
        return ' '.join(new_list)

    def replace_nan(self):
        """
        Replace nan by 'bon' or 'mauvais' in the dataframe.
        """
        to_drop = []
        for i in self.df.index:
            r = self.df['avis'][i]
            if pd.isna(r) or r in ['nan', 'Nan'] or type(r) == float:
                to_drop.append(i)
        print("droped nan : ", len(to_drop))
        self.df = self.df.drop(to_drop)

    def remove_urls(self, review):
        review = re.sub(self.urls, '', review, flags=re.MULTILINE)
        return(review)

    def clean_str(self, review):
        """
        Remove special characters from the string.
        """

        if len(review) > 0 or review != None:
            review = re.sub(self.unused_chars, ' ', review)
            review = review.replace('.', ' ').replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')
            review = review.lower()
            review = re.sub(' +', ' ', review)
            review = re.sub(r' (?! ) ', '', review) #removing single characters
       
        return review
    
    def clean_stop_words(self, review):
        review_list = review.split(" ")

        new_list = []
        for word in review_list:
            if word != " " and not word in self.connecting_words and len(word) > 1:
                new_list.append(word)

        return ' '.join(new_list)

    def clean_review(self, review):
        review = self.clean_str(review)
        review = self.clean_stop_words(review)
        review = self.correction_spelling(review)
        
        return review

    def clean_dataset(self):
        """
        Main method call to prepare a text to be vectorized.
        """

        self.df = self.replace_nan()
        for i in trange(self.df.shape[0]):
            review = self.df.at[i, 'avis']
            review = self.clean_str(review)
            review = self.clean_stop_words(review)
            review = self.correction_spelling(review)
                
            self.df.at[i, 'avis'] = review

    def filter_long_review(self):
        """
        Filter the string with too many words.
        """

        to_drop = []
        for i in self.df.index:
            review = self.df['avis'][i]
            review_list = review.split()

            if len(review_list) > self.max_words:
                to_drop.append(i)

        self.df = self.df.drop(to_drop)
        print(f"dropped {len(to_drop)} lines")

    def fix_repartition_for_4_classes(self):
        """
        Fix the bad repartitions of the dataset, by removing randomly good reviews.
        """

        d = self.df.groupby(['classe_bon_mauvais'], as_index=False).count()
        nb_bad = d['avis'][0]

        nb_good = d['avis'][2]
        to_remove = nb_good - nb_bad

        while(to_remove > 0):
            row: DataFrame = self.df.sample()
            index = row.first_valid_index()
            print(f"{to_remove}")

            if row['classe_bon_mauvais'][index] == 2:
                self.df.drop(index, inplace=True)
                to_remove -= 1


        nb_good = d['avis'][3]
        to_remove = nb_good - nb_bad

        while(to_remove > 0):
            row: DataFrame = self.df.sample()
            index = row.first_valid_index()
            print(f"{to_remove}")

            if row['classe_bon_mauvais'][index] == 3:
                self.df.drop(index, inplace=True)
                to_remove -= 1
        
        d = self.df.groupby(['classe_bon_mauvais'], as_index=False).count()
        nb_bad = d['avis'][0]
        nb_good = d['avis'][1]
        print("2", nb_bad, nb_good)

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

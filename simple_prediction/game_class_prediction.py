import collections
import numpy as np
import pandas as p
import seaborn as sns
sns.set(style="darkgrid")
sns.set(font_scale=1.3)
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer


def format_name(df):
    for i in df.columns: 
        if i != "jeux":
            continue
        df[i] = p.factorize(df[i])[0]  
    df.to_csv("./refactored_dataset.csv")

def replace_nan(df):
    for i in range(df.shape[0]):
        if p.isna(df["avis"][i]):
            if df["note"][i] > 11: 
                df.at[i, "avis"] = "good"
            else:
                df.at[i, "avis"] = "bad"
    return df


# Permet d'avoir un graphique des mots les plus fréquents du corpus
def show_frequencies(df):
    corpus = df["avis"]
    cv = CountVectorizer()
    bow  = cv.fit_transform(corpus)
    word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
    word_counter = collections.Counter(word_freq)
    word_counter_df = p.DataFrame(word_counter.most_common(20), columns=['word', 'freq'])

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
    plt.show()



# def tokenize_reviews():

# def vectorize_review():

if __name__ == '__main__':
    # original_df = p.read_csv("../dataset/dataset_clean.csv")
    # format_name(original_df)

    working_df = p.read_csv("./refactored_dataset.csv")
    working_df = replace_nan(working_df)
    show_frequencies(working_df)

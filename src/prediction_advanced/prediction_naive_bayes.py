from nltk.classify.util import names_demo_features
from clean_data import CleanData
import pandas as pd
from joblib import dump, load
import json

import nltk

PREDICT = False
SAVE = True
MODEL_NAME = 'models/test_Naive1.model'
FEATURES_NAME = 'models/features.json'
TEST_SIZE = 1/3

MAX_WORD = 300

DATASET = 'dataset/csv/dataset_clean_2.0.csv_repartion_fixed.csv'


def review_features(review, word_features):
    review_words = set(review)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in review_words)
    return features

def compute_word_features(df):
    all_words = nltk.FreqDist()
            
    print("Computes words frequencies :")
    acc = 0
    size = df.shape[0]
    for avis in df['avis']:
        acc+=1
        print(acc, "/", size, sep='', end='\r')
        for word in avis.split():
            all_words[word] += 1

    word_features = list(all_words)[:2000]
    print("\rdone.")

    return word_features

def get_tuples(df, word_features):
    print("get all tuples of reviews :")
    reviews = []
    acc = 0
    size = df.shape[0]
    for i in df.index:
        acc+=1
        print(acc, "/", size, sep='', end='\r')
        reviews.append((df['avis'][i].split(), df['classe_bon_mauvais'][i]))
    print("\rdone.")

    return reviews

def compute_features(reviews, size, word_features):

    print("get Features out of review :")
    # featuresets = [(review_features(r, word_features), c) for (r,c) in reviews]
    featuresets = []
    acc = 0
    size = len(reviews)
    for (r, c) in reviews:
        acc+=1
        print(acc, "/", size, sep='', end='\r')
        featuresets.append((review_features(r, word_features), c))

    test_limits = int(size* TEST_SIZE)
    train_set, test_set = featuresets[test_limits:], featuresets[:test_limits]
    print("\rdone.")

    return train_set, test_set


if __name__ == "__main__":

    if PREDICT:
        nbc: nltk.NaiveBayesClassifier = load(MODEL_NAME)

        with open(FEATURES_NAME, 'r', encoding='utf8') as fp:
            data = json.load(fp)
            word_features = data['features']

        avis = input("write an opinions to predict : \n")
        c = CleanData(MAX_WORD)
        avis = c.clean_review(avis)
        print(word_features)
        test = review_features(avis, word_features)
        print(nbc.classify(test))

    else:
        
        df = pd.read_csv(DATASET)[['classe_bon_mauvais', 'avis']]
        print(df.groupby(['classe_bon_mauvais'], as_index=False).count())


        word_features = compute_word_features(df)
        reviews = get_tuples(df, word_features)
        train_set, test_set = compute_features(reviews, df.shape[0], word_features)
        
        print("training : ")
        nbc = nltk.NaiveBayesClassifier.train(train_set)
        
        if SAVE:
            data = {}
            data['features'] = word_features
            dump(nbc, MODEL_NAME)
            with open(FEATURES_NAME, 'w', encoding='utf8') as fp:
                json.dump(data, fp, indent=4, ensure_ascii=False)


        print("predict...")
        
        labels = []
        tests = []

        acc = 0
        size = len(test_set)
        for feat, label in test_set:
            acc+=1
            print(acc, "/", size, sep='', end='\r')
            observed = nbc.classify(feat)
            tests.append(observed)
            labels.append(label)
            
        print("Confusion Matrix:\n", nltk.ConfusionMatrix(labels, tests))
        print("accuracy:", nltk.accuracy(labels, tests))
        #print("recall:", nltk.recall(labels, tests))
        print("precision:", nltk.precision(labels, tests))
        print("f1_score:", nltk.f_measure(labels, tests))
        

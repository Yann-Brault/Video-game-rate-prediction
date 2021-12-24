from gensim.models.keyedvectors import KeyedVectors

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from joblib import dump, load
from clean_data import CleanData
from utils import compute_metrics

import nltk

PREDICT = False
SAVE = False
LOAD = True
MODEL_NAME = 'models/test1.model'


MAX_WORD = 300
MAX_ITER = 500
TEST_SIZE = 1/3
LAYERS = (13, 13, 13)
VEC_DIM = 200
DATASET = 'dataset/csv/dataset_clean_2.0.csv_repartion_fixed.csv'
WORD2VEC = 'dataset/vectors/vec_200_no_phrase.vec'


class VectorReader:

    def __init__(self, path: str, vector_dim: int, max_len) -> None:
        self.path = path
        self.vector_dim = vector_dim
        self.max_len = max_len
        self.words_dictionnary = KeyedVectors

    def init_embed(self):
        self.words_dictionnary: KeyedVectors = KeyedVectors.load_word2vec_format('dataset/vectors/vec-200.bin', binary=True)
    
    def transform_to_vector_sum(self, X_train):
        vectors_list = []
        lenavis = len(X_train)
        acc = 1
        for avis in X_train:
            print(f"{acc}/{lenavis}", end='\r')
            acc+=1
            avis_list = avis.split()
            avis_vec = []
            
            for word in avis_list:
                try:
                    word_vec = self.words_dictionnary[word]
                    avis_vec.append(word_vec)
                except:
                    pass

            for i in range(len(avis_vec), self.vector_dim):
                avis_vec.append(np.zeros(self.vector_dim, dtype=np.float32))

            tot_vec = []
            for i in range(self.vector_dim):
                sum = 0.0
                for vec in avis_vec[i]:
                    sum += vec
                tot_vec.append(sum)

            vectors_list.append(tot_vec)      

        return vectors_list      


if __name__ == "__main__":

    if PREDICT:
        mlpc: MLPClassifier = load(MODEL_NAME)
        avis = input("write an opinions to predict : \n")
        c = CleanData(MAX_WORD)
        avis = c.clean_avis(avis)
        v = VectorReader(WORD2VEC, VEC_DIM, MAX_WORD)
        v.init_embed()
        avis_vec = v.transform_to_vector_sum([avis])
        prediction = mlpc.predict(avis_vec)
        print(prediction)


    else:
        df = pd.read_csv(DATASET)[['classe_bon_mauvais', 'avis']]
        mlpc: MLPClassifier

        # c = CleanData(df, MAX_WORD)
        # c.filter_long_avis()
        # c.fix_repartition()
        # df.to_csv(f'{DATASET}_repartion_fixed.csv')

        print(df.groupby(['classe_bon_mauvais'], as_index=False).count())

        X = df['avis'].copy()
        y = df['classe_bon_mauvais'].copy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

        
        v = VectorReader(WORD2VEC, VEC_DIM, MAX_WORD)
        v.init_embed()

        if LOAD:
            mlpc = load(MODEL_NAME)
            X_test_vec = v.transform_to_vector_sum(X_test)
        else:
            X_test_vec = v.transform_to_vector_sum(X_test)
            X_train_vec = v.transform_to_vector_sum(X_train)
        
            mlpc = MLPClassifier(hidden_layer_sizes=LAYERS, max_iter=MAX_ITER)
        
            print("fit")
            mlpc.fit(X_train_vec, y_train)
        
        if SAVE:
            dump(mlpc, MODEL_NAME)

        print("predict")
        predictions = mlpc.predict(X_test_vec)

        M = confusion_matrix(y_test, predictions)

        print(M)
        print(compute_metrics(2, M))

        print(classification_report(y_test, predictions))

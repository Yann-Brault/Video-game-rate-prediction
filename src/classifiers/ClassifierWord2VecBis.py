import os
import tempfile
from typing import DefaultDict
import pandas as pd

import numpy as np
from src.classifiers.Classifier import Classifier

from src.classifiers.ClassifierWord2Vec import ClassifierWord2Vec
from gensim.models import FastText  


class ClassifierWord2VecMix(ClassifierWord2Vec):

    def __init__(self, data, word2vec_bin=None, max_word=0, max_iter=0, test_size=0, layers=0, vec_dim=0) -> None:
        super().__init__(
            data,
            word2vec_bin=word2vec_bin,
            max_word=max_word,
            max_iter=max_iter,
            test_size=test_size,
            layers=layers,
            vec_dim=vec_dim
            )

        self.words_dictionary_self = None
        
        self.create_dictionnary()

    def create_dictionnary(self):
        sentences = [s.split() for s in self.data['avis']]

        print("Training model")
        model = FastText(sentences=sentences, vector_size=self.vec_dim)
        print('Training done')
        self.words_dictionary_self = model.wv

    def _transform_to_vec(self, set_to_transform):
        vectors_list = []
        len_set = len(set_to_transform)
        acc = 1
        for review in set_to_transform:
            print(f"{acc}/{len_set}", end='\r')
            acc+=1
            reviews_list = review.split()
            reviews_vec = []
            
            for word in reviews_list:
                try:
                    word_vec = self.words_dictionnary[word]
                    reviews_vec.append(word_vec)
                except:
                    print("new")
                    word_vec = self.words_dictionary_self[word]
                    reviews_vec.append(word_vec)

            for i in range(len(reviews_vec), self.vec_dim):
                reviews_vec.append(np.zeros(self.vec_dim, dtype=np.float32))

            tot_vec = []
            for i in range(self.vec_dim):
                sum = 0.0
                for vec in reviews_vec[i]:
                    sum += vec
                tot_vec.append(sum)

            vectors_list.append(tot_vec)
        
        return vectors_list
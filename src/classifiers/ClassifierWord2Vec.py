import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from gensim.models.keyedvectors import KeyedVectors

from src.prediction_advanced.utils import compute_metrics
from src.classifiers.Classifier import Classifier




class ClassifierWord2Vec(Classifier):

    def __init__(self, data, word2vec_bin = None, max_word=0, max_iter=0, test_size=0, layers=0, vec_dim=0, create = True) -> None:
        super().__init__(data)

        self.max_word = max_word
        self.max_iter = max_iter
        self.test_size = test_size
        self.layers = layers
        self.vec_dim = vec_dim
        self.word2vec_bin = word2vec_bin
    
        self.words_dictionary = None

        if create:
            self.load_dictionnary_from_bin()
        else:
            self.create_dictionnary()

    def load_dictionnary_from_bin(self):
        print(f"Loading dictionary from binary {self.word2vec_bin}...")
        self.words_dictionnary: KeyedVectors = KeyedVectors.load_word2vec_format(self.word2vec_bin, binary=True)

    def create_dictionnary(self):
        print("Creating vector dictionary from scratch...")
        pass

    def init_sets(self):
        print("Initialization of train and test sets...")
        X = self.data['avis'].copy()
        y = self.data['classe_bon_mauvais'].copy()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size)
    
    def init_classifier(self):
        print(f"Initialization of the MLP Classifier with {self.layers} layers and {self.max_iter} max iteration.")
        self.classifier = MLPClassifier(hidden_layer_sizes=self.layers, max_iter=self.max_iter)

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
                    pass

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

    def fit_transform_data(self):
        if self.X_test is not None:
            print(f"Transforming reviews of the test set into vectors...")
            self.X_test = self._transform_to_vec(self.X_test)
        
        if self.X_train is not None:
            print(f"Transforming reviews of the train set into vectors...")
            self.X_train = self._transform_to_vec(self.X_train)
        
    def train(self):
        print("Training on data...")
        self.classifier.fit(self.X_train, self.y_train)
    
    def predict(self):
        print("Prediciton on tests...")
        self.predictions = self.classifier.predict(self.X_test)

    def show_results(self):
        M = confusion_matrix(self.y_test, self.predictions)

        print(M)
        print(compute_metrics(2, M))

        print(classification_report(self.y_test, self.predictions))
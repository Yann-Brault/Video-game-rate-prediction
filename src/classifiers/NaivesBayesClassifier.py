import collections

from sklearn.model_selection import train_test_split

from src.classifiers.Classifier import Classifier
from src.utils.clean_data import CleanData
from joblib import dump, load
import json

import nltk


class NaivesBayes(Classifier):
    """
    This classifiers uses a dictionnary of features, to represent the sentence.
    First of all a dictionnary is created out of the whole corpus of the given data.
    Then for each review, a set of their own features are given.

    The machin e learning model uses the basic Naives Bayes model, provided by the librairy nltk.
    """

    def __init__(self, data, nb_word, test_size=0) -> None:
        super().__init__(data)

        self.test_size = test_size

        self.train_set = None
        self.test_set = None

        self.nb_word = nb_word

        self.refsets = None
        self.testsets = None
        self.predictions_labels = None
        self.predictions = None

        self.word_features = None
    
    def _review_features(self, review):
        """
        Builds the dictionnary for a review.
        """
        
        review_words = set(review)
        features = {}
        for word in self.word_features:
            features['contains({})'.format(word)] = (word in review_words)
        return features

    def _compute_word_features(self):
        """
        Computes and get all features from the corpus.
        """
        
        all_words = nltk.FreqDist()
    
        print("Computes words frequencies ...")
        acc = 0
        size = self.data.shape[0]
        for avis in self.data['avis']:
            acc+=1
            print(acc, "/", size, sep='', end='\r')
            for word in avis.split():
                all_words[word] += 1

        self.word_features = list(all_words)[:self.nb_word]

    def _compute_features(self, tuple_to_compute):
        """
        Computes the features for a single review.
        """
        
        print("get Features out of review ...")
        
        featuresets = []
        acc = 0
        size = len(tuple_to_compute)
        for (r, c) in tuple_to_compute:
            acc+=1
            print(acc, "/", size, sep='', end='\r')
            featuresets.append((self._review_features(r), c))
        
        return featuresets

    def _get_tuples(self, input, output):
        """
        Transforms the input and ouptut in tuple to fit the nltk model.
        """
        
        print("Get all tuples of reviews ...")
        reviews = []
        size = len(input)
        for i in input.index:
            print(i, "/", size, sep='', end='\r')
            reviews.append((input[i].split(), output[i]))

        return reviews

    def load(self, model_path: str, features_path: str = None):
        """
        Loads a previous model trained and a previous features dictionnary.
        """
        
        self.classifier = load(model_path)

        with open(features_path, 'r', encoding='utf8') as fp:
            data = json.load(fp)
            self.word_features = data['features']

    def save(self, path, features = None):
        """
        Save a trained model and a the created features dictionnary.
        """

        data = {}
        data['features'] = self.word_features
        dump(self.classifier, path)
        with open(features, 'w', encoding='utf8') as fp:
            json.dump(data, fp, indent=4, ensure_ascii=False)

    def init_classifier(self):
        """
        Inits the classifier.
        """
        
        print("Initialization of the word features dictionnary")
        self._compute_word_features()

    def init_sets(self):
        """
        Inits and split the train/test sets.
        """

        print("Initialization of train and test sets...")
        X = self.data['avis'].copy()
        y = self.data['classe_bon_mauvais'].copy()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size)
        
    def fit_transform_data(self):
        """
        Tranforms the data of each train and test sets.
        """
        
        if self.X_test is not None:
            print(f"Transforming reviews of the test set into tuples of features...")
            self.test_set = self._get_tuples(self.X_test, self.y_test)
            self.test_set = self._compute_features(self.test_set)
        
        if self.X_train is not None:
            print(f"Transforming reviews of the train set into tuples of features...")
            self.train_set = self._get_tuples(self.X_train, self.y_train)
            self.train_set = self._compute_features(self.train_set)

    def train(self):
        """
        Trains the models on the train set.
        """
        
        print("Training on data...")
        self.classifier = nltk.NaiveBayesClassifier.train(self.train_set)

    def predict(self):
        """
        Predicts on the test set.
        """
        
        print("Prediciton on tests...")

        self.predictions = []
        self.predictions_labels = []
        self.refsets = collections.defaultdict(set)
        self.testsets = collections.defaultdict(set)
        
        for i, (feats, label) in enumerate(self.test_set):
            print(f"{i}/{len(self.test_set)}", end='\r')
            self.refsets[label].add(i)
            observed = self.classifier.classify(feats)
            self.testsets[observed].add(i)
            self.predictions_labels.append(label)
            self.predictions.append(observed)
    
    def show_results(self):
        """
        Shows results of the prediction, and computes the confusion matrix and the classification report.
        """
        
        print("Confusion Matrix:\n", nltk.ConfusionMatrix(self.predictions_labels, self.predictions))
        print("accuracy:", nltk.accuracy(self.predictions_labels, self.predictions))
        
        for i in range(2):
            print (f"class {i}:")
            print("f1_score:", nltk.f_measure(self.refsets[i], self.testsets[i]))
            print("recall:", nltk.recall(self.refsets[i], self.testsets[i]))
            print("precision:", nltk.precision(self.refsets[i], self.testsets[i]))

    def predict_input(self, review: str):
        """
        Predict a review provided in the console.
        """
        
        test = self._review_features(review.split())
        print(self.classifier.classify(test))
from abc import abstractmethod
from joblib import dump, load
from enum import Enum
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from src.prediction_advanced.utils import compute_metrics
from sklearn.metrics import accuracy_score, classification_report

class ClassifierType(Enum):
    WORD2VEC = 1
    NAIVES_BAYES = 2
    WORD2VEC_MIX = 3
    MNB_CLASSIC = 4
    LOG_REG_CLASSIC = 5
    
    
class Classifier:
    """
    Model Class of all Classifiers.
    """
    
    def __init__(self, data) -> None:
        self.data = data
        self.verify_data()
    
        self.classifier = None
        self.predictions = None
        
        self.X_train = None
        self.y_train = None
    
        self.X_test = None
        self.y_test = None
    
    def verify_data(self):
        print("Verifying data ... ")
        to_drop = []
        for i in self.data.index:
            r = self.data['avis'][i]
            if pd.isna(r) or r in ['nan', 'Nan'] or type(r) == float:
                to_drop.append(i)
        print("droped nan : ", len(to_drop))
        self.data = self.data.drop(to_drop)
    
    def show_repartition(self) -> None:
        print(self.data.groupby(['classe_bon_mauvais'], as_index=False).count())
    
    def save(self, path, features = None):
        dump(self.classifier, path)
    
    def load(self, model_path: str, features_path: str = None):
        self.classifier = load(model_path)
    
    def train(self):
        print("Training on data...")
        self.classifier.fit(self.X_train, self.y_train)
    
    def predict(self):
        print("Prediciton on tests...")
        self.predictions = self.classifier.predict(self.X_test)

    def show_results(self):
        print('==========================Classifier Results============================')
        M = confusion_matrix(self.y_test, self.predictions)
        print(M)

        print('\n Accuracy: ', accuracy_score(self.y_test, self.predictions))
        print('\n Score: ', self.classifier.score(self.X_test, self.y_test))

        print(compute_metrics(2, M))
        print(classification_report(self.y_test, self.predictions))

    # Abstract methods

    def plot_matrix_classification_report(self, path, classes):
        pass

    def plot_accuracy_precisions(self, path, classes: list[int], accuracies: list[list[int]], precisions: list[list[int]]):
        pass


    @abstractmethod
    def fit_transform_data(self):
        pass
    
    @abstractmethod
    def init_sets(self):
        pass
    
    @abstractmethod
    def init_classifier(self):
        pass
    
    @abstractmethod
    def predict_input(self, review: str):
        pass
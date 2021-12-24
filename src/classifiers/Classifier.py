from joblib import dump, load
from enum import Enum

class ClassifierType(Enum):
    WORD2VEC = 1
    NAIVES_BAYES = 2
    

class Classifier:

    def __init__(self, data) -> None:
        self.data = data
        self.classifier = None
        self.predictions = None
        
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        pass
    
    def show_repartition(self) -> None:
        print(self.data.groupby(['classe_bon_mauvais'], as_index=False).count())

    def fit_transform_data(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass
    
    def save(self, path, features = None):
        dump(self.classifier, path)

    def load(self, model_path: str, features_path: str = None):
        self.classifier = load(model_path)

    def init_sets(self):
        pass

    def init_classifier(self):
        pass

    def predict_input(self, review: str):
        pass

    def show_results(self):
        pass
from src.classifiers.Classifier import Classifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier


class TFIDF_MLP(Classifier):

    def __init__(self, data, test_size, max_iter, layers, max_features) -> None:
        super().__init__(data)

        self.test_size = test_size
        self.max_iter = max_iter
        self.layers = layers
        self.max_features = max_features

    def init_sets(self):
        print("Initialization of train and test sets...")
        td = TfidfVectorizer(max_features=self.max_features) 
        X = self.data['avis'].copy()
        X = td.fit_transform(X).toarray()
        y = self.data['classe_bon_mauvais'].copy()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, random_state=0)

    def init_classifier(self):
        #print(f"Initialization of the LogReg Classifier with a reg of {self.regularization} and {self.max_iter} max iteration.")
        self.classifier = MLPClassifier(hidden_layer_sizes=self.layers, max_iter=self.max_iter)
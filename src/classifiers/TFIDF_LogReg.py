from src.classifiers.Classifier import Classifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression



class TFIDF_LogReg(Classifier):

    def __init__(self, data, test_size, max_iter, regularization, max_features) -> None:
        super().__init__(data)

        self.test_size = test_size
        self.max_iter = max_iter
        self.regularization = regularization
        self.max_features = max_features

    def init_sets(self):
        print("Initialization of train and test sets...")
        self.td = TfidfVectorizer(max_features=self.max_features) 
        X = self.data['avis'].copy()
        X = self.td.fit_transform(X).toarray()
        y = self.data['classe_bon_mauvais'].copy()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, random_state=0)

    def init_classifier(self):
        print(f"Initialization of the LogReg Classifier with a reg of {self.regularization} and {self.max_iter} max iteration.")
        self.classifier = LogisticRegression(C=self.regularization, max_iter=self.max_iter)

    def predict_input(self, review: str):
        array = self.td.transform([review])
        prediction = self.classifier.predict(array)
        print(prediction)

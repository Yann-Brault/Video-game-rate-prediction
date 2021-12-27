from abc import abstractmethod
from joblib import dump, load
from enum import Enum
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import src.utils.utils as u
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


class ClassifierType(Enum):
    WORD2VEC = 1
    NAIVES_BAYES = 2
    WORD2VEC_MIX = 3
    TFIDF_MNB = 4
    TFIDF_LogReg = 5
    
    
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

        print(u.compute_metrics(2, M))
        print(classification_report(self.y_test, self.predictions))

    # Abstract methods

    def get_accuracy(self):
        return accuracy_score(self.y_test, self.predictions)

    def get_precisions(self, c):
        d = classification_report(self.y_test, self.predictions, output_dict=True)
        c_str = str(c)

        return d[c_str]['precision']


    def plot_matrix_classification_report(self, cp_path, matric_path, classes):
        pass

    def plot_accuracy_precisions(self, acc_path, prec_path, label, params, classes: list[int], accuracies: list[int], precisions: list[list[int]]):
        
        plt.xlabel(label)
        plt.ylabel('accuracy')
        plt.plot(params, accuracies)
        plt.savefig(acc_path)
        plt.show()

        plt.xlabel(label)
        plt.ylabel('precisions')
        
        color = ['r', 'b', 'g', 'y']
        for i in range(len(precisions)):
            plt.plot(params, precisions[i], color=color[i], label=classes[i])
        plt.legend(loc="upper right", title='classes')

        plt.savefig(prec_path)
        plt.show()



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
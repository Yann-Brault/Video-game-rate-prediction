import pandas as pd
from src.classifiers.MLPClassifierW2V import MLP_Word2Vec

from src.classifiers.TFIDF_LogReg import TFIDF_LogReg
from src.classifiers.ClassifierWord2Vec import ClassifierWord2Vec
from src.classifiers.Classifier import ClassifierType
from src.classifiers.ClassifierWord2VecMix import ClassifierWord2VecMix
from src.classifiers.NaivesBayesClassifier import NaivesBayes
from src.classifiers.TFIDF_Multinomial import TFIDF_MNB
from sklearn.neural_network import MLPClassifier

from src.utils.clean_data import CleanData


class PipelineClassifier:

    def __init__(self, classifier_type, data, vec_bin=None, max_features=None, nb_word_n=None, max_word=None, max_iter=None, test_size=None, layers=None, vec_dim=None, reg=None, alpha=None) -> None:
        self.classifier = None
        self.classifier_type = classifier_type
        self.data = data
        self.max_features = max_features
        self.vec_bin = vec_bin
        self.nb_word_n = nb_word_n
        self.max_word = max_word
        self.max_iter = max_iter
        self.test_size = test_size
        self.layers = layers
        self.vec_dim = vec_dim
        self.reg = reg
        self.alpha = alpha

        self.__init_classifier()

    def __init_classifier(self):
        if self.classifier_type == ClassifierType.WORD2VEC:
            self.classifier = ClassifierWord2Vec(
                data=self.data,
                word2vec_bin=self.vec_bin,
                max_iter=self.max_iter,
                layers=self.layers,
                vec_dim=self.vec_dim,
                test_size=self.test_size
            )

        elif self.classifier_type == ClassifierType.NAIVES_BAYES:
            self.classifier = NaivesBayes(
                data=self.data,
                nb_word=self.nb_word_n,
                test_size=self.test_size
            )

        elif self.classifier_type == ClassifierType.WORD2VEC_MIX:
            self.classifier = ClassifierWord2VecMix(
                data=self.data,
                word2vec_bin=self.vec_bin,
                max_iter=self.max_iter,
                layers=self.layers,
                vec_dim=self.vec_dim,
                test_size=self.test_size
            )
        
        elif self.classifier_type == ClassifierType.TFIDF_LogReg:
            self.classifier = TFIDF_LogReg(
                data=self.data,
                test_size=self.test_size,
                max_iter=self.max_iter,
                regularization=self.reg,
                max_features=self.max_features
            )

        elif self.classifier_type == ClassifierType.TFIDF_MNB:
            self.classifier = TFIDF_MNB(
                data=self.data,
                test_size=self.test_size,
                alpha=self.alpha,
                max_features=self.max_features
            )
        elif self.classifier_type == ClassifierType.MLP_Word2Vec:
            self.classifier = MLP_Word2Vec(
                data=self.data,
                test_size=self.test_size,
                max_iter=self.max_iter,
                regularization=self.reg,
                max_features=self.max_features
            )

    def load(self, model_path, features_path=None):
        self.classifier.load(model_path, features_path)
        self.classifier.init_sets()
        self.classifier.X_train = None
        self.classifier.y_train = None
        self.classifier.fit_transform_data()

    def save(self, model_path, features_path=None):
        self.classifier.save(model_path, features_path)

    def train(self):
        self.classifier.init_sets()
        self.classifier.init_classifier()
        self.classifier.fit_transform_data()
        self.classifier.train()

    def transform_data(self):
        self.classifier.init_sets()
        self.classifier.fit_transform_data()
    
    def train_without_transform(self):
        self.classifier.train()
    
    def predict(self):
        self.classifier.show_repartition()
        self.classifier.predict()
        self.classifier.show_results()

    def predict_input(self):
        review = input("Write a review to predict: \n")
        c = CleanData(self.max_word)
        review = c.clean_review(review)
        review = self.classifier.predict_input(review)


MODEL_NAME = 'models/Log_Reg_Classic.model'
JSON_FEATURES = 'models/features_naives_4000.json'

NB_WORD_NB = 4000
MAX_WORD = 300
MAX_ITER = 500
TEST_SIZE = 0.2
LAYERS = (13, 13, 13)
VEC_DIM = 200
REG = 1.0
ALPHA = 1.0


MAX_FEATURES = 10000
VEC_BIN = 'dataset/vectors/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin'


CLASSIFIER = ClassifierType.WORD2VEC
DATA_ANALYSIS = True

DATASET = 'dataset/csv/dataset_0-3.csv'

PLOT_MATRIX_PATH = 'assets/w2v/mlp/word2vec_LogReg_cm_0-3.plot.png'
CP_PATH = 'assets/w2v/mlp/word2vec_LogReg_cr_0-3.txt'

PLOT_ACC_PATH = 'assets/features/nb/naives_bayes_4500_cm_0-3.plot.png'
PLOT_PREC_PATH = 'assets/features/nb/naives_bayes__4500_cm_0-3.plot.png'

TILE_CM = 'Naives Bayes 4500 words'
TITLE_PREC_ACC = 'number of words'

MODEL_PATH = 'assets/features/nb/naives_bayes'
CLASSES = [0,1,2,3]

if __name__ == "__main__":

    
    if DATA_ANALYSIS:
        df = pd.read_csv(DATASET)[['classe_bon_mauvais', 'avis']]

        p = PipelineClassifier(CLASSIFIER, data=df, vec_bin=VEC_BIN, max_iter=500, layers=LAYERS, vec_dim=VEC_DIM, test_size=TEST_SIZE)

        p.train()
        p.predict()

        p.classifier.plot_matrix_classification_report(TILE_CM, CP_PATH, PLOT_MATRIX_PATH, CLASSES)

    else:
        df = pd.read_csv(DATASET)[['classe_bon_mauvais', 'avis']]

        params = [1500, 3000, 4500, 6000]

        accuracies = []
        precisions = [[], []]

        for i in range(len(params)):

            p = PipelineClassifier(CLASSIFIER, data=df, nb_word_n=params[i], test_size=TEST_SIZE)
            p.train()
            p.predict()
            
            accuracies.append(p.classifier.get_accuracy())
            for c in CLASSES:
                precisions[c].append(p.classifier.get_precisions(c))

        p.classifier.plot_accuracy_precisions(TITLE_PREC_ACC, PLOT_ACC_PATH, PLOT_PREC_PATH, TITLE_PREC_ACC, params, CLASSES, accuracies, precisions)
        
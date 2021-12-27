import pandas as pd
from scipy.sparse import data
from sklearn.naive_bayes import MultinomialNB
from src.classifiers.TFIDF_LogReg import TFIFF_LogReg_classic

from src.classifiers.ClassifierWord2Vec import ClassifierWord2Vec
from src.classifiers.Classifier import ClassifierType
from src.classifiers.ClassifierWord2VecMix import ClassifierWord2VecMix
from src.classifiers.NaivesBayesClassifier import NaivesBayes
from src.classifiers.TFIDF_Multinomial import TFIFF_Multinomial_classic
from src.prediction_advanced.clean_data import CleanData


PREDICT = False
SAVE = True
LOAD = False
MODEL_NAME = 'models/Log_Reg_Classic.model'
JSON_FEATURES = 'models/features_naives_4000.json'

NB_WORD_NB = 4000
MAX_WORD = 300
MAX_ITER = 500
TEST_SIZE = 1/3
LAYERS = (13, 13, 13)
VEC_DIM = 200
REG = 1.0
ALPHA = 1.0


MAX_FEATURES = 10000
DATASET = 'dataset/csv/dataset_new_repartition_4.csv'
VEC_BIN = 'dataset/vectors/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin'


CLASSIFIER = ClassifierType.LOG_REG_CLASSIC

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
        if CLASSIFIER == ClassifierType.WORD2VEC:
            self.classifier = ClassifierWord2Vec(
                data=self.data,
                word2vec_bin=self.vec_bin,
                max_word=self.max_word,
                max_iter=self.max_iter,
                layers=self.layers,
                vec_dim=self.vec_dim,
                test_size=self.test_size,
            )

        elif CLASSIFIER == ClassifierType.NAIVES_BAYES:
            self.classifier = NaivesBayes(
                data=self.data,
                nb_word=self.nb_word_n,
                test_size=self.test_size
            )

        elif CLASSIFIER == ClassifierType.WORD2VEC_MIX:
            self.classifier = ClassifierWord2VecMix(
                data=self.data,
                word2vec_bin=self.vec_bin,
                max_word=self.max_word,
                max_iter=self.max_iter,
                layers=self.layers,
                vec_dim=self.vec_dim,
                test_size=self.test_size,
            )
        
        elif CLASSIFIER == ClassifierType.LOG_REG_CLASSIC:
            self.classifier = TFIFF_LogReg_classic(
                data=self.data,
                test_size=self.test_size,
                max_iter=self.max_iter,
                regularization=self.reg,
                max_features=self.max_features
            )

        elif CLASSIFIER == ClassifierType.MNB_CLASSIC:
            self.classifier = TFIFF_Multinomial_classic(
                data=self.data,
                test_size=self.test_size,
                alpha=self.alpha,
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
    
    def predict(self):
        self.classifier.show_repartition()
        self.classifier.predict()
        self.classifier.show_results()

    def predict_input(self):
        review = input("Write a review to predict: \n")
        c = CleanData(self.max_word)
        review = c.clean_review(review)
        review = self.classifier.predict_input(review)

if __name__ == "__main__":
    df = pd.read_csv(DATASET)[['classe_bon_mauvais', 'avis']]
    
    p = PipelineClassifier(ClassifierType.LOG_REG_CLASSIC, df, test_size=TEST_SIZE, max_iter=MAX_ITER, reg=REG, max_features=MAX_FEATURES)


    p.train()
    p.predict()


    # if CLASSIFIER == ClassifierType.WORD2VEC:
    #     classifier = ClassifierWord2Vec(
    #         data=df,
    #         word2vec_bin=VEC_BIN,
    #         max_word=MAX_WORD,
    #         max_iter=MAX_ITER,
    #         layers=LAYERS,
    #         vec_dim=VEC_DIM,
    #         test_size=TEST_SIZE,
    #     )

    # elif CLASSIFIER == ClassifierType.NAIVES_BAYES:
    #     classifier = NaivesBayes(
    #         data=df,
    #         nb_word=NB_WORD_NB,
    #         test_size=TEST_SIZE
    #     )

    # elif CLASSIFIER == ClassifierType.WORD2VEC_MIX:
    #     classifier = ClassifierWord2VecMix(
    #         data=df,
    #         word2vec_bin=VEC_BIN,
    #         max_word=MAX_WORD,
    #         max_iter=MAX_ITER,
    #         layers=LAYERS,
    #         vec_dim=VEC_DIM,
    #         test_size=TEST_SIZE,
    #     )
    
    # elif CLASSIFIER == ClassifierType.LOG_REG_CLASSIC:
    #     classifier = TFIFF_LogReg_classic(
    #         data=df,
    #         test_size=TEST_SIZE,
    #         max_iter=MAX_ITER,
    #         regularization=REG,
    #         max_features=MAX_FEATURES
    #     )
    # elif CLASSIFIER == ClassifierType.MNB_CLASSIC:
    #     classifier = TFIFF_Multinomial_classic(
    #         data=df,
    #         test_size=TEST_SIZE,
    #         alpha=ALPHA,
    #         max_features=MAX_FEATURES
    #     )
    

    # # classifier.show_repartition()


    # if LOAD:
    #     classifier.load(MODEL_NAME, JSON_FEATURES)
    #     if not PREDICT:
    #         classifier.init_sets()
    #         classifier.X_train = None
    #         classifier.y_train = None
    #         classifier.fit_transform_data()
    # else:
    #     classifier.init_sets()
    #     classifier.init_classifier()
    #     classifier.fit_transform_data()
    #     classifier.train()

    # if SAVE:
    #     classifier.save(MODEL_NAME, JSON_FEATURES)

    # if PREDICT:
    #     review = input("Write a review to predict: \n")
    #     c = CleanData(MAX_WORD)
    #     review = c.clean_review(review)
    #     review = classifier.predict_input(review)

    # else:
    #     classifier.predict()
    #     classifier.show_results()

    
    

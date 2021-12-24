import pandas as pd

from src.classifiers.ClassifierWord2Vec import ClassifierWord2Vec
from src.classifiers.Classifier import ClassifierType
from src.classifiers.NaivesBayesClassifier import NaivesBayes
from src.prediction_advanced.clean_data import CleanData


PREDICT = True
SAVE = False
LOAD = True
MODEL_NAME = 'models/testNaives2_4000.model'
JSON_FEATURES = 'models/features_naives_4000.json'

MAX_WORD = 300
MAX_ITER = 500
TEST_SIZE = 1/3
LAYERS = (13, 13, 13)
VEC_DIM = 200
DATASET = 'dataset/csv/dataset_clean_2.0.csv_repartion_fixed.csv'
VEC_BIN = 'dataset/vectors/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin'


CLASSIFIER = ClassifierType.NAIVES_BAYES


if __name__ == "__main__":
    df = pd.read_csv(DATASET)[['classe_bon_mauvais', 'avis']]
    
    if CLASSIFIER == ClassifierType.WORD2VEC:

        classifier = ClassifierWord2Vec(
            data=df,
            word2vec_bin=VEC_BIN,
            max_word=MAX_WORD,
            max_iter=MAX_ITER,
            layers=LAYERS,
            vec_dim=VEC_DIM,
            test_size=TEST_SIZE,
            create = True
        )

    elif CLASSIFIER == ClassifierType.NAIVES_BAYES:
        classifier = NaivesBayes(
            data=df,
            test_size=TEST_SIZE
        )

    else:
        pass

    classifier.show_repartition()

    if LOAD:
        classifier.load(MODEL_NAME, JSON_FEATURES)
        if not PREDICT:
            classifier.init_sets()
            classifier.X_train = None
            classifier.y_train = None
            classifier.fit_transform_data()
    else:
        classifier.init_sets()
        classifier.init_classifier()
        classifier.fit_transform_data()
        classifier.train()

    if SAVE:
        classifier.save(MODEL_NAME, JSON_FEATURES)

    if PREDICT:
        review = input("Write a review to predict: \n")
        c = CleanData(MAX_WORD)
        review = c.clean_avis(review)
        review = classifier.predict_input(review)

    else:
        classifier.predict()
        classifier.show_results()

    
    

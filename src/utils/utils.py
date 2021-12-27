##############################

# Méthode d'extraction TF-IDF
# Fonctionne sur le principe de Grid Search
# Applique plusieurs fois le classfier sur les données avec différents paramètres afin de trouver le meilleur
# fonctionne avec des classifier Multunomial NB et logisitic regression

# base on : https://towardsdatascience.com/sentiment-analysis-with-text-mining-13dd2b33de27
# and on : https://medium.com/analytics-vidhya/applying-text-classification-using-logistic-regression-a-comparison-between-bow-and-tf-idf-1f1ed1b83640


from pprint import pprint
from time import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB

def grid(classifier, params, X_train, X_test, y_train, y_test):

    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    grid_search = GridSearchCV(classifier, cv=5, param_grid=params)

    print("Performing grid search...")
    print(f"classifier : {classifier}")
    pprint(f"parameters : {params}")
    
    initial_t = time()
    # on fit les datas à notre grid search
    grid_search.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - initial_t))
    print()
    print("---------------SCORE-------------")
    print(grid_search.cv_results_)

    # On récupère le meilleur score de prédiction ( à priori équivalent à la précision)

    print("Best CV score : %0.3f" % grid_search.best_score_)
    print("Best parameters set: ")
    # On donne également les meilleurs paramètres
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(params.keys()) :
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    print("Test score with best_estimator_ : %0.3f" % grid_search.best_estimator_.score(X_test, y_test))
    print("\n")
    print("Classification Report Test Data")
    print(classification_report(y_test, grid_search.best_estimator_.predict(X_test)))

def replace_nan(df):
    for i in range(df.shape[0]):
        if pd.isna(df["avis"][i]):
            if df["classe_bon_mauvais"][i] == 1: 
                df.at[i, "avis"] = "good"
            else:
                df.at[i, "avis"] = "bad"
    return df

def compute_metrics(taille, M):
    TP = [0] * taille
    TN = [0] * taille
    FP = [0] * taille
    FN = [0] * taille

    Total = 0
    for i in range(taille):
        for j in range(taille):
            Total += M[i][j]

    for i in range(taille):
        TP[i] = M[i][i]
        for j in range(taille):
            FN[i] += M[i][j]
            FP[i] += M[j][i]
        FN[i] -= M[i][i]
        FP[i] -= M[i][i]
        TN[i] = Total - FP[i] - FN[i] + TP[i]
    return TN, TP, FN, FP


if __name__ == '__main__':
    df = pd.read_csv("./dataset/good_repartition.csv")

    df = replace_nan(df)

    X = df.iloc[:, 2].values  # X correspond aux reviews
    y = df.iloc[:, 1].values  # y correspond aux classes comme c'est ce que l'on cherche à prévoir 

    # Extractions des features
    td = TfidfVectorizer(max_features=10000)
    X = td.fit_transform(X).toarray()

    # On split les datas en différents ensemble d'entrainement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

    logreg_classifier = LogisticRegression(C=1, max_iter=1000).fit(X_train, y_train) # max_iter à 100 de base, faut monter ici car trop de data
    logreg_score = logreg_classifier.score(X_test, y_test)

    mnb_classifier = MultinomialNB().fit(X_train, y_train)
    mnb_score = mnb_classifier.score(X_test, y_test)

    print('======================================================')
    print(f"\n LogReg score {logreg_score}")
    print(f"\n MNB score {mnb_score}")
    print('======================================================')

    param_logreg_grid_ = {'C': [1e-5, 1e-3, 1e-1, 1e0, 1e1, 1e2]}
    # as for the alpha param mnb, c is the hyperparameter of log reg
    param_mnb_grid_ ={'alpha': [0, 1.0, 2.0, 3.0, 4.0]}
    # to understand alpha param https://stackoverflow.com/questions/33830959/multinomial-naive-bayes-parameter-alpha-setting-scikit-learn
    grid(mnb_classifier, param_mnb_grid_, X_train, X_test, y_train, y_test)

    grid(logreg_classifier, param_logreg_grid_, X_train, X_test, y_train, y_test)
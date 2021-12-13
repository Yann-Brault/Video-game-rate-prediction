import pandas as pd
from sklearn.externals._packaging.version import CmpKey
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def rate_csv_creator(myData):
    myData.sort_values(['note'], ascending=False, inplace=True)
    colummns_names = myData.columns
    good_games_df = pd.DataFrame(columns=colummns_names)
    bad_games_df = pd.DataFrame(columns=colummns_names)
    select_good_grade = myData.loc[myData['note'] > 11]
    select_bad_grades = myData.loc[myData['note'] < 12]
    good_games_df = pd.concat([good_games_df, select_good_grade], ignore_index = True, axis = 0)
    good_games_df.to_csv("./dataset/good_games.csv")
    bad_games_df = pd.concat([bad_games_df, select_bad_grades], ignore_index=True, axis=0)
    bad_games_df.to_csv("./dataset/bad_games.csv")

def review_avg_len(csv):
    df = pd.read_csv(csv)
    acc = 0
    for avis in df['avis']:
        acc += len(str(avis))
    mean = acc / len(df['avis'])
    print(f"average length of {csv} reviews is {mean:.2f} characters")

def determine_good_or_bad(data): #here we will assign a class number according to the rate. 1 for good games and 2 for bad games.
    M = data['note'].max()
    cat = 1 
    grade_class = []
    predicted_grade = []

    for grade in data['note']:
        if 11 < grade <= M:
            cat = 1
        else:
            cat = 2
        grade_class.append(cat)
        predicted_grade.append(predicte_grade())
    data['grade_class'] = grade_class
    data['predicted_grade'] = predicted_grade
    data.to_csv("./dataset/dataset_with_grade_class.csv")

def predicte_grade(): 
    return 15

def accuracy(TN, TP, FN, FP):
    size_list = len(TN)
    accuracy_sum = 0
    for i in range(size_list):
        accuracy_sum += (TP[i] + TN[i]) / (TP[i] + TN[i] + FN[i] + FP[i])
    return accuracy_sum / size_list

def recall(TP, FN):
    size_list = len(TP)
    recall_sum = 0
    for i in range(size_list):
        recall_sum += TP[i] / (TP[i] + FN[i])
    return recall_sum / size_list


def CM(data):
    m_size = 2
    cp_pp = 1
    M = data['note'].max()
    TP = [0] * m_size
    TN = [0] * m_size
    FP = [0] * m_size
    FN = [0] * m_size
    matrix = [[0 for i in range(m_size)] for j in range(m_size)]

    for i, pg in enumerate(data['predicted_grade']):
        cp = data['grade_class'][i]
        if 11 < pg <= 20:
            cp_pp = 1
        else:
            cp_pp = 2
        matrix[cp - 1][cp_pp - 1] += 1

    total = 0
    for i in range(m_size):
        sum_fn = 0
        sum_fp = 0
        TP[i] = matrix[i][i]
        for j in range(m_size):
            sum_fn += matrix[i][j]
            sum_fp += matrix[j][i]
            total += matrix[i][j]
        sum_fn -= matrix[i][i]
        sum_fp -= matrix[i][i]
        FN[i] = sum_fn
        FP[i] = sum_fp

    for i in range(m_size):
        TN[i] = total - FP[i] - FN[i] - TP[i]

    print(f"TP values : {TP}\n")
    print(f"FP values : {FP}\n")
    print(f"FN values : {FN}\n")
    print(f"TN values : {TN}\n")
    print(matrix)

    accu = accuracy(TN, TP, FN, FP)
    rec = recall(TP, FN)

    print(f"the globla accuracy is {accu:.2f}\n")
    print(f"the globla recall is {rec:.2f}\n")



    
        

if __name__ == '__main__': 
    myData = pd.read_csv("./dataset/dataset_original.csv")
    rate_csv_creator(myData)
    review_avg_len("./dataset/good_games.csv")
    review_avg_len("./dataset/bad_games.csv")
    determine_good_or_bad(myData)
    my_data = pd.read_csv("./dataset/dataset_with_grade_class.csv")
    CM(my_data)
from numpy import matrix
from tqdm import trange
import pandas as pd


def predict(data: pd.DataFrame) -> tuple[list[int], list[int]]:
    classes_predicted = []
    classes_base = []

    for i in trange(data.shape[0]):
        row = data.iloc[i]
        neg = row['negative_words']
        pos = row['positive_words']
        
        predict = 1
        if neg > pos: 
            predict = 0
        
        base = row['classe_bon_mauvais']
        classes_base.append(int(base))
        classes_predicted.append(int(predict))
    
    return classes_base, classes_predicted
    
def compute_confusion_matrix(classes_base, classes_predicted):
    
    M = [[0, 0], [0, 0]]

    for i in range(len(classes_base)):
        M[classes_base[i]][classes_predicted[i]] += 1
    
    return M



if __name__ == "__main__":
    data = pd.read_csv('dataset/dataset_clean.csv')

    classes_base, classes_predicted = predict(data)

    M = compute_confusion_matrix(classes_base, classes_predicted)
    
    TP = [0,0]
    TN = [0,0]
    FP = [0,0]
    FN = [0,0]
    Total = M[0][0] + M[0][1] + M[1][0] + M[1][1]

    for i in range(2):
        TP[i] = M[i][i]
        for j in range(2):
            FN[i] += M[i][j]
            FP[i] += M[j][i]
        
        FN[i] -= M[i][i]
        FP[i] -= M[i][i]
        TN[i] = Total - FP[i] - FN[i] + TP[i]

    for i in range(2):
        print(f"cat {i}: TP:{TP[i]}, TN:{TN[i]}, FP:{FP[i]}, FN:{FN[i]}\n")

        
        
    

    



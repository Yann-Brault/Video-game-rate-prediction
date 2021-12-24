
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
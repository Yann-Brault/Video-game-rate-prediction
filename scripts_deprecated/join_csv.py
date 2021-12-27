import pandas as pd
from pandas.core.frame import DataFrame
from tqdm import trange


jeux = []
note = []
cbm = []
neg = []
pos = []
avis = []

acc = 0
for i in range(103):
    print(i)
    try:
        data_tmp = pd.read_csv(f'ant-csv/data_part_{i}.csv')

        for j in trange(data_tmp.shape[0]):
            jeux.append(data_tmp['jeux'][j])
            note.append(data_tmp['note'][j])
            cbm.append(data_tmp['classe_bon_mauvais'][j])
            neg.append(data_tmp['negative_words'][j])
            pos.append(data_tmp['positive_words'][j])
            avis.append(data_tmp['avis'][j])

    except Exception as e:
        print(i, e)

data = pd.DataFrame(data={'jeux': jeux, 'note': note, 'classe_bon_mauvais': cbm, 'negative_words': neg, 'positive_words': pos, 'avis': avis})
data.to_csv('dataset/csv/dataset_clean_2.0.csv')


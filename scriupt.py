import pandas as pd

from src.utils.clean_data import CleanData
from autocorrect import Speller


DATASET = 'dataset/csv/data_original_4_rep.csv'
MAX_WORD = 200

def show_repartition(df) -> None:
    print(df.groupby(['classe_bon_mauvais'], as_index=False).count())

df = pd.read_csv(DATASET, na_values=['', 'nan', 'NaN', 'Nan', 'None'])
c = CleanData(MAX_WORD, df)


#show_repartition(c.df)

# classe_bon_mauvais = []
# for i in range(df.shape[0]):
#     if df['note'][i] < 5 :
#         classe_bon_mauvais.append(0)
#     elif df['note'][i] < 10:
#         classe_bon_mauvais.append(1)
#     elif df['note'][i] < 15:
#         classe_bon_mauvais.append(2)
#     else:
#         classe_bon_mauvais.append(3)

# classe_bon_mauvais = []
# for i in range(df.shape[0]):
#     if df['note'][i] < 12 :
#         classe_bon_mauvais.append(0)
#     else:
#         classe_bon_mauvais.append(1)

# c.df['classe_bon_mauvais'] = classe_bon_mauvais
# c.df = c.df[['classe_bon_mauvais', 'avis']]


c.replace_nan()

#c.fix_repartition_for_4_classes()

show_repartition(c.df)

# c.filter_long_review()

for i in c.df.index:
    c.df.at[i, 'avis'] = c.remove_urls(c.df['avis'][i])
    c.df.at[i, 'avis'] = c.clean_stop_words(c.df['avis'][i])
    c.df.at[i, 'avis'] = c.clean_str(c.df['avis'][i])
    if type(c.df.at[i, 'avis']) != str:
        print("drop !")
        c.df = c.df.drop([i])

c.replace_nan()

# c.fix_repartition_for_4_classes()


c.save_data('dataset/csv/dataset_0-3.csv')
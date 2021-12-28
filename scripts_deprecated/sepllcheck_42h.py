from spellchecker import SpellChecker
import pandas as pd
import re
from tqdm import trange

spell = SpellChecker(language='fr')


def correction_spelling(avis):    
    avis_list = avis.split(" ")
    
    bad = []
    for word in avis_list:
        if word != " ":
            bad = spell.unknown(avis_list)

    new_list = []
    for word in avis_list:
        if word in bad:
            new_list.append(spell.correction(word))
        else:
            new_list.append(word)
    
    
    return ' '.join(new_list)


if __name__ == "__main__":

    avis = "très gros joueur créateur stadium moi formule avec abonnement non absolu an standard préfère payer l'avoir vie plutôt formule abonnement où va sérieusement falloir songer baisser prix quelque chose comme l'année encore prix là pas l'accès clubs skins désolé ubisoft étiez grandement remontés dans estime derniers temps mais là juste pas possible"

    data = pd.read_csv('./dataset_clean_for_vector_connecting_less.csv')
    new_data = pd.DataFrame(data={'jeux': [], 'note': [], 'classe_bon_mauvais': [], 'negative_words': [], 'positive_words': [], 'avis': []})

    index = 0
    acc = 0
    num = 0
    for i in trange(data.shape[0]):
        if (acc == 1000):
            acc = 0
            new_data.to_csv(f'./csv/data_part_{num}.csv', index=False)
            new_data = pd.DataFrame(data={'jeux': [], 'note': [], 'classe_bon_mauvais': [], 'negative_words': [], 'positive_words': [], 'avis': []})
            num += 1

        avis = data['avis'][i]

        if not pd.isna(avis) and type(avis) == str:
            jeux = data['jeux'][i]
            note = int(data['note'][i])
            classe_bon_mauvais = data['classe_bon_mauvais'][i]
            negative_words = data['negative_words'][i]
            positive_words = data['positive_words'][i]
            
            avis = correction_spelling(avis)

            new_data.loc[index] = [jeux, note, classe_bon_mauvais, negative_words, positive_words, avis]
            index += 1
        acc+=1

    new_data.to_csv('test.csv')   
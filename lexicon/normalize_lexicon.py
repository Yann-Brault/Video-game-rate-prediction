import unidecode

neg_word: str
pos_word: str


with open('neg_words.txt', 'r', encoding='utf-8') as f:
    neg_word = f.read()
    f.close()
    
with open('pos_words.txt', 'r', encoding='utf-8') as f:
    pos_word = f.read()
    f.close()

neg_word = unidecode.unidecode(neg_word)
pos_word = unidecode.unidecode(pos_word)

with open('neg_words_normalized.txt', 'w', encoding='utf-8') as f:
    f.write(neg_word)
    f.close()

with open('pos_words_normalized.txt', 'w', encoding='utf-8') as f:
    f.write(pos_word)
    f.close()

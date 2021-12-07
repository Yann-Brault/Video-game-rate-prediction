from sklearn.feature_extraction.text import CountVectorizer
corpus = [
     "Coucou c'est moi bérnard",
     'salut mon pote... :D comment ça va ?',
     "Hey comment ça va ? t'as la forme?",
     "Salaud ! enculé !... ::!"
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())
# array(['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third',
#        'this'], ...)

print(X.toarray())

vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
X2 = vectorizer2.fit_transform(corpus)
vectorizer2.get_feature_names_out()

print(X2.toarray())

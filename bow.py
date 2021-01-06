from nltk.tokenize import word_tokenize
sentence = "Man Utd and Liverpool are at the top of league at the start of 2021"
#splitting string using pandas method
sentence.split()

#splitting using word_tokenize
word_tokenize(sentence)


#CountVectorizer from sckit learn
from sklearn.feature_extraction.text import CountVectorizer

corpus=["Man City is out of top4", "Spurs are npwhere in consideration", "Bielsa at leads is creating fine art of football", "Arsenal are not even winning FA cup this year","Chelsea will yet again sack a manger"]

countVect=CountVectorizer()

#fitting it to corpus to create sparse matrix
countVect.fit(corpus)

new_corpus=countVect.transform(corpus)

#to get an understanding of the resulting sparse vector
print(countVect.vocabulary_)

#employing word tokenize
from nltk.tokenize import word_tokenize
countVect1=CountVectorizer(tokenize=word_tokenize, token_pattern=None)
countVect1.fit(corpus)
print(countVect1.vocabulary_)

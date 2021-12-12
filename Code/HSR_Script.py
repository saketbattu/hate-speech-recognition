import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
 
tweets_df = pd.read_csv('HSR_Dataset.csv')

import string
string.punctuation

import nltk
from nltk.corpus import stopwords
stopwords.words('english')

def message_cleaning(message):
    test_punc_removed = []
    for char in message:
        if char not in string.punctuation:
            test_punc_removed.append(char)
    test_punc_removed_join = ''.join(test_punc_removed)
    test_stopword_removed = []
    for word in test_punc_removed_join.split():
        if word.lower() not in stopwords.words('english'):
            test_stopword_removed.append(word)
    return test_stopword_removed

from sklearn.feature_extraction.text import CountVectorizer
tweets_countvectorizer = CountVectorizer(analyzer = message_cleaning, dtype = 'uint8').fit_transform(tweets_df['tweet']).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tweets_countvectorizer, tweets_df['label'], test_size=0.2)

from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

import joblib
filename = "trained_model.joblib"
joblib.dump(NB_classifier, filename)


from sklearn.metrics import classification_report, confusion_matrix
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)

filename = "trained_model.joblib"
loaded_model = joblib.load(filename)

results = classification_report(y_test, y_predict_test)

f = open("results.txt", "w")
f.write(results)
f.close()

print(results)

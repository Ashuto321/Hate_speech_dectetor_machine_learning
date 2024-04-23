# hate_speech_detection.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import re
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import string

nltk.download('stopwords')
stopword = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

def train_model():
    df = pd.read_csv("C:\\Users\\Ashutosh\\Desktop\\HatespeechdetectorML\\twitter_data.csv")
    df['labels'] = df['class'].map({0: "hate speech detected", 1: "offensive language detected", 2: "no hate and offensive speech"})
    df = df[['tweet', 'labels']]
    df["tweet"] = df["tweet"].apply(clean)

    x = np.array(df["tweet"])
    y = np.array(df["labels"])

    if pd.isna(y).any():
        imputer = SimpleImputer(strategy='most_frequent')
        y_imputed = imputer.fit_transform(y.reshape(-1, 1))
    else:
        y_imputed = y

    cv = CountVectorizer()
    x = cv.fit_transform(x)
    x_train, _, y_train, _ = train_test_split(x, y_imputed, test_size=0.33, random_state=42)
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    return cv, clf


#examples can be experimented y user itself
# test_data="i will kill you"
# df = cv.transform([test_data]).toarray()
# print(clf.predict(df))

#%%
import tweepy
import pandas as pd
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import pandas_profiling
import sys
import os
import seaborn as sb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#%% data import

tweets = pd.read_csv("Data\labeling_dataset_v2.csv")
tweets.head()

def labels_numeric(label):
    if label == "Positive":
        return(0)
    elif label == "Neutral":
        return(1)
    elif label == "Negative":
        return(2)
    elif label == "Issue":
        return(3)
    else:
         return(4)

tweets['Sentiment'] = tweets['Sentiment_Label'].apply(labels_numeric)
tweets.head()

#%% splitting data

y = tweets['Sentiment']
X = tweets[['tweetLength', 'negSentiment', 'neuSentiment', 'posSentiment', 'comSentiment']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=5)



#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

#%%
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#%%
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(40),max_iter=50000)

mlp.fit(X_train, y_train)

#%%
from sklearn.metrics import classification_report,confusion_matrix
predictions = mlp.predict(X_test)

#%%
confusion_matrix(y_test, predictions)
print(classification_report(y_test,predictions))

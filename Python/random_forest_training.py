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
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=1000)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

# prediction on test set
y_pred=clf.predict(X_test)


#%%
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


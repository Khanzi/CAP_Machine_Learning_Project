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
import pickle
from sklearn.preprocessing import StandardScaler


#%%
apple_nn = pickle.load(open("pickles/apple_nn.sav" ,"rb"))
facebook_nn = pickle.load(open("pickles/facebook_nn.sav" ,"rb"))
fpu_nn = pickle.load(open("pickles/fpu_nn.sav" ,"rb"))
tesla_nn = pickle.load(open("pickles/tesla_nn.sav" ,"rb"))
wellsfargo_nn = pickle.load(open("pickles/wellsfargo_nn.sav" ,"rb"))
youtube_nn = pickle.load(open("pickles/youtube_nn.sav" ,"rb"))



# Authentication

CONSUMER_KEY ='Lp7p3I3Yc35DUg5x8ToGUxVtV'
CONSUMER_SECRET = 'Ltm3JEJTnT7w1pY12FjQvOVwi1WWt5rFowD1gqw2fcDjY5HZAs'

ACCESS_KEY = '724658688061902848-UPUXPU4H8SlSWe7Z0mh8GJXSdfQm9FM'
ACCESS_SECRET = 'ujK7JhUOf7o6Lva093YGT6TVComkrplT7oUJHOInolTxm'


auth = tweepy.OAuthHandler(consumer_key=CONSUMER_KEY, 
    consumer_secret=CONSUMER_SECRET)

#Connect to the Twitter API using the authentication
api = tweepy.API(auth)

sentimenter = SentimentIntensityAnalyzer()

#%%
scaler = StandardScaler()


def labels_cat(label):
    if label == 0:
        return("Positive")
    elif label == 1:
        return("Neutral")
    elif label == 2:
        return("Negative")
    
def search(query):
    
    tweets = api.search(q=query, lang = 'en', tweet_mode='extended', count = 200)

    DataSet = pd.DataFrame()

    DataSet['tweetID'] = [tweet.id for tweet in tweets]
    DataSet['tweetText'] = [tweet.full_text for tweet in tweets]
    DataSet['tweetRetweetCt'] = [tweet.retweet_count for tweet 
    in tweets]
    DataSet['tweetFavoriteCt'] = [tweet.favorite_count for tweet 
    in tweets]
    DataSet['tweetSource'] = [tweet.source for tweet in tweets]
    DataSet['tweetCreated'] = [tweet.created_at for tweet in tweets]


    DataSet['userID'] = [tweet.user.id for tweet in tweets]
    DataSet['userScreen'] = [tweet.user.screen_name for tweet 
    in tweets]
    DataSet['userName'] = [tweet.user.name for tweet in tweets]
    DataSet['userCreateDt'] = [tweet.user.created_at for tweet 
    in tweets]
    DataSet['userDesc'] = [tweet.user.description for tweet in tweets]
    DataSet['userFollowerCt'] = [tweet.user.followers_count for tweet 
    in tweets]
    DataSet['userFriendsCt'] = [tweet.user.friends_count for tweet 
    in tweets]
    DataSet['userLocation'] = [tweet.user.location for tweet in tweets]
    DataSet['userTimezone'] = [tweet.user.time_zone for tweet 
    in tweets]

    return DataSet

def sent_compound(text):
    return(sentimenter.polarity_scores(text)['compound'])
def sent_neutral(text):
    return(sentimenter.polarity_scores(text)['neu'])
def sent_positive(text):
    return(sentimenter.polarity_scores(text)['pos'])
def sent_negative(text):
    return(sentimenter.polarity_scores(text)['neg'])


def prepare(data):
    #data['tweetText'] = data['tweetText'].apply(clean_tweet)
    data['tweetLength'] = data['tweetText'].apply(len)
    data['sentimentPolarity'] = data['tweetText'].apply(sentimenter.polarity_scores)
    data['negSentiment'] = data['tweetText'].apply(sent_negative)
    data['neuSentiment'] = data['tweetText'].apply(sent_neutral)
    data['posSentiment'] = data['tweetText'].apply(sent_positive)
    data['comSentiment'] = data['tweetText'].apply(sent_compound)
    return(data[['tweetText','tweetCreated','tweetLength','negSentiment','neuSentiment','posSentiment','comSentiment']])



def apple_pipeline():
    data = prepare(search("@AppleSupport"))
    tweets = data[['tweetLength', 'negSentiment', 'neuSentiment', 'posSentiment', 'comSentiment']]
    data["num_label"] = apple_nn.predict(tweets)
    data['Label'] = data['num_label'].apply(labels_cat)
    print(data[['tweetText','Label']])
def facebook_pipeline():
    data = prepare(search("@Facebook"))
    tweets = data[['tweetLength', 'negSentiment', 'neuSentiment', 'posSentiment', 'comSentiment']]
    data["num_label"] = facebook_nn.predict(tweets)
    data['Label'] = data['num_label'].apply(labels_cat)
    print(data[['tweetText','Label']])
def fpu_pipeline():
    data = prepare(search("@FLPolyU"))
    tweets = data[['tweetLength', 'negSentiment', 'neuSentiment', 'posSentiment', 'comSentiment']]
    data["num_label"] = fpu_nn.predict(tweets)
    data['Label'] = data['num_label'].apply(labels_cat)
    print(data[['tweetText','Label']])
def tesla_pipeline():
    data = prepare(search("@TeslaSupport"))
    tweets = data[['tweetLength', 'negSentiment', 'neuSentiment', 'posSentiment', 'comSentiment']]
    scaler = StandardScaler()
    scaler.fit(tweets)
    data["num_label"] = tesla_nn.predict(tweets)
    data['Label'] = data['num_label'].apply(labels_cat)
    print(data[['tweetText','comSentiment','num_label','Label']])
def wellsfargo_pipeline():
    data = prepare(search("@WellsFargo"))
    tweets = data[['tweetLength', 'negSentiment', 'neuSentiment', 'posSentiment', 'comSentiment']]
    data["num_label"] = wellsfargo_nn.predict(tweets)
    data['Label'] = data['num_label'].apply(labels_cat)
    print(data[['tweetText','Label']])
def youtube_pipeline():
    data = prepare(search("@TeamYouTube"))
    tweets = data[['tweetLength', 'negSentiment', 'neuSentiment', 'posSentiment', 'comSentiment']]
    data["num_label"] = youtube_nn.predict(tweets)
    data['Label'] = data['num_label'].apply(labels_cat)
    print(data[['tweetText','comSentiment','Label']])


#%%
tesla_pipeline()
#%%
if __name__ == "__main__":
    while True:
        print("Please Choose a Company:")
        print("[1] Apple \n [2] Facebook \n [3] Florida Poly \n [4] Tesla \n [5] Wells Fargo \n [6] Youtube")
        selection = int(input("Selection:"))
        if selection == 1:
            apple_pipeline()
        elif selection == 2:
            facebook_pipeline()
        elif selection == 3:
            fpu_pipeline()
        elif selection == 4:
            tesla_pipeline()
        elif selection == 5:
            wellsfargo_pipeline()
        elif selection == 6:
            youtube_pipeline()
        else:
             print("Invalid, Please choose a valid option.")
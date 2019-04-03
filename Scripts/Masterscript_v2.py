#!/usr/bin/env python
# coding: utf-8

# In[46]:


#Libraries
import tweepy
import pandas as pd
import json
from textblob import TextBlob
import re
import pandas_profiling


# In[9]:


# Authentication

CONSUMER_KEY ='Lp7p3I3Yc35DUg5x8ToGUxVtV'
CONSUMER_SECRET = 'Ltm3JEJTnT7w1pY12FjQvOVwi1WWt5rFowD1gqw2fcDjY5HZAs'

ACCESS_KEY = '724658688061902848-UPUXPU4H8SlSWe7Z0mh8GJXSdfQm9FM'
ACCESS_SECRET = 'ujK7JhUOf7o6Lva093YGT6TVComkrplT7oUJHOInolTxm'


# Authenticate 
auth = tweepy.OAuthHandler(consumer_key=CONSUMER_KEY, 
    consumer_secret=CONSUMER_SECRET)

#Connect to the Twitter API using the authentication
api = tweepy.API(auth)


# ## Functions

# In[27]:


def search(query):
    
    tweets = api.search(q=query)

    DataSet = pd.DataFrame()

    DataSet['tweetID'] = [tweet.id for tweet in tweets]
    DataSet['tweetText'] = [tweet.text for tweet in tweets]
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
    DataSet['lang'] = [tweet.lang for tweet in tweets]

    return DataSet


# In[11]:


def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


# In[17]:


def get_sentiment(text):
    tb = TextBlob(text)
    return(tb.polarity)


# In[40]:


def prepare(data):
    data['tweetText'] = data['tweetText'].apply(clean_tweet)
    data['tweetLength'] = data['tweetText'].apply(len)
    data['sentimentPolarity'] = data['tweetText'].apply(get_sentiment)
    data_english = data.loc[data['lang'] == "en"]
    return(data_english[['tweetText','tweetCreated','tweetLength','sentimentPolarity']])


# In[42]:


test = search("Hello")
p = prepare(test)


# In[50]:


pandas_profiling.ProfileReport(p)


# In[ ]:





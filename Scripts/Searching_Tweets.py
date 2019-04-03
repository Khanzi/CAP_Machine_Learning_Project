#!/usr/bin/env python
# coding: utf-8

# # Twitter Sentiment Analysis
# ## Machine Learning | Dr. Samarah
# - Kahlil Wehmeyer
# - Luke Rhon
# - Richard Cruz
# - Diego De La Torre
# - Jackie G.

# In[ ]:





# In[2]:


# Libraries
import tweepy
import pandas as pd
import json
from textblob import TextBlob


# In[3]:


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


# In[3]:


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

    return DataSet


# ### search function
# The search function takes a string of the topic you are interested in and returns a data frame of all the recent tweets about that specific topic

# In[4]:


Florida_Poly_Tweets = search("Florida Poly")


# In[5]:


Florida_Poly_Tweets


# In[6]:


Florida_Poly_Tweets['tweetText'][5]


# In[7]:


text = Florida_Poly_Tweets['tweetText'][5]
tokenized = TextBlob(text)
tokenized.sentiment.subjectivity


# In[8]:


def get_sentiment(text):
    tb = TextBlob(text)
    return(tb.sentiment)


# In[9]:


Florida_Poly_Tweets['sentiment'] = Florida_Poly_Tweets['tweetText'].apply(get_sentiment)


# In[10]:


Florida_Poly_Tweets.sentiment


# In[1]:


Florida_Poly_Tweets.head(10)


# In[7]:


tweets = api.search("Uganda")
tweets


# In[ ]:





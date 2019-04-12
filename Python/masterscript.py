#%%
import tweepy
import pandas as pd
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import pandas_profiling


#%%
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

sentimenter = SentimentIntensityAnalyzer()

#%%
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

#%% 
def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


#%%
def sent_compound(text):
    return(sentimenter.polarity_scores(text)['compound'])
def sent_neutral(text):
    return(sentimenter.polarity_scores(text)['neu'])
def sent_positive(text):
    return(sentimenter.polarity_scores(text)['pos'])
def sent_negative(text):
    return(sentimenter.polarity_scores(text)['neg'])


#%%
def prepare(data):
    #data['tweetText'] = data['tweetText'].apply(clean_tweet)
    data['tweetLength'] = data['tweetText'].apply(len)
    data['sentimentPolarity'] = data['tweetText'].apply(sentimenter.polarity_scores)
    data['negSentiment'] = data['tweetText'].apply(sent_negative)
    data['neuSentiment'] = data['tweetText'].apply(sent_neutral)
    data['posSentiment'] = data['tweetText'].apply(sent_positive)
    data['comSentiment'] = data['tweetText'].apply(sent_compound)
    return(data[['tweetText','tweetCreated','tweetLength','negSentiment','neuSentiment','posSentiment','comSentiment']])


#%%
p = search("Test")
p


#%%
# Youtube, Apple, Tesla, Florida Polytechnic, Wells Fargo, and Facebook
youtube =  prepare(search("@TeamYouTube"))
apple = prepare(search("@AppleSupport"))
flpoly =prepare(search("@FLPolyU"))
wellsfargo = prepare(search("@WellsFargo"))
facebook = prepare(search("@Facebook"))
tesla = prepare(search("@TeslaSupport"))


#%%
youtube['Company'] = "Youtube"
apple['Company'] = "Apple"
flpoly['Company'] = "Florida Poly"
wellsfargo['Company'] = "Wells Fargo"
facebook['Company'] = "Facebook"
tesla['Company'] = "Tesla"


#%%
writer = pd.ExcelWriter('Data\labeling_individual.xlsx', engine='xlsxwriter')

youtube.to_excel(writer, sheet_name = "Youtube")
apple.to_excel(writer, sheet_name = "Apple")
flpoly.to_excel(writer, sheet_name = "Florida Poly")
wellsfargo.to_excel(writer, sheet_name = "Wells Fargo")
facebook.to_excel(writer, sheet_name = "Facebook")
tesla.to_excel(writer, sheet_name = "Tesla")

#%%
writer.save()

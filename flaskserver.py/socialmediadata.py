import tweepy
import nltk
import re
import string
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import SnowballStemmer
#yfrom sklearn.feature_extraction.text import CountVectorizer
def socialmediaData(fighter1):
   
    # Authentication Keys
    consumerKey = "7xzXFbwLqoiDO7lyJgNxGwdFG"   # enter your own credentials
    consumerSecret = "JiC0zq4wCiXLXzXFI93ijZNRcCkQamQZhN3GoZZOb790fT9kdB"  # enter your own credentials

    accessToken = "1278718386143399936-3fZE59ZIuHgG23XUVajG3IHkCCvHpD"    # enter your own credentials
    accessTokenSecret = "0cv5oxL8XzfoPu90q8Lc36IApYEqr5KIKbZoZi8lw1LOw"    # enter your own credentials
    auth = tweepy.OAuthHandler(consumerKey, consumerSecret) 
    auth.set_access_token(accessToken, accessTokenSecret)
    api = tweepy.API(auth)
    
    tweetnum = 25
    tweets=api.search_tweets(q=fighter1,count=tweetnum)
   
    tweet_list = []
    for tweet in tweets:
        tweet_list.append(tweet.text)  # transfer all text into variable tweet list
   
    totalnumber=tweetnum
    
    df = pd.DataFrame(tweet_list, columns =['initial text'])
    df['cleaned text']= df['initial text']
   # print(df['initial text'])
    #Removing RT, Punctuation etc
    clean_tweets = []
    for tweet in df['cleaned text']:
        tweet = re.sub("RT @[A-Za-z0-9]+","",tweet) #Remove @ sign
        ##Here's where all the cleaning takes place
        clean_tweets.append(tweet)
    df['cleaned text'] = clean_tweets
    df.drop_duplicates(subset ="cleaned text",
                         keep = False, inplace = True)

    #Specify location
    #Calculating Negative, Positive, Neutral and Compound values
    if(len(df['cleaned text'])==0):
        return 0

    cols=["neg","neu","pos","compound"]
    sentimentdf=pd.DataFrame(columns=cols)
    sia= SentimentIntensityAnalyzer()
    sentimentdf["neg"]=  df.apply(lambda row: (sia.polarity_scores(row["cleaned text"])['neg']*100), axis=1)
    sentimentdf["pos"]=  df.apply(lambda row: (sia.polarity_scores(row["cleaned text"])['pos']*100), axis=1)

  
    
    totaltot =(df["cleaned text"].count())
  
    negneg= sentimentdf['neg'].sum()
  
    pospos=sentimentdf['pos'].sum()
    
    totalpol= pospos-negneg
    return totalpol
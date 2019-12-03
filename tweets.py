# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 2019

@author: tg
"""
import tweepy
from textblob import TextBlob
import csv

a=[]

consumer_key= ''
consumer_secret= ''

access_token=''
access_token_secret=''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth,wait_on_rate_limit=True)

# Open/Create a file to append data
csvFile = open('enjoy.csv', 'w',newline='')
#Use csv Writer
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.search,q="#enjoy",count=200,lang="en",since="2017-04-03").items():
    analysis = TextBlob(tweet.text)
    print (tweet.created_at, tweet.text)
    
    csvWriter.writerow([tweet.created_at, tweet.text.encode('ascii','ignore')])
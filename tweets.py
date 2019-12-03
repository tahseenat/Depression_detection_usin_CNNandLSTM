# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 2019

@author: tg
"""
import tweepy
from textblob import TextBlob
import csv

a=[]

consumer_key= 'DmN7LBwPjwYEieqMwCsZykAtW'
consumer_secret= 'eHKkyASGlvgOAU7w5Yh0tEUYa4i9zXM5p3Nketg7J5YVc525rf'

access_token='2263786051-DApjJRFNvj6KJEE9aTdcxAVOwrOjeQQNxl0V7w7'
access_token_secret='oxtFitZVnRfWWGUFE6RXb1BQmalfFo202c6JsTuKVvry7'

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
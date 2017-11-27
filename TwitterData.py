#!/usr/bin/env python

"""
Uses Twython module to access Twitter API
Use Twitter API to get old tweets for a particular keyword
and for a particular duration; 
export csv file
"""

import sys, string, got, codecs, tweepy, time, datetime
import json as simplejson
from twython import Twython
from tweepy import OAuthHandler

reload(sys)
sys.setdefaultencoding('utf-8')

consumer_key = 'cEiRuD4KjEoS46ZrvHPe5hkpm'
consumer_secret = 'mJUyZ8NUhdyO0v71OtBYx46I8pcvo6fhknTmYaCnrkjpV2oEJo'
access_token = '887287866-hK6uOncwQl4Dg4HtCuVCJBwciu1LFKGN6innHx5J'
access_secret = 'DBl2ESDWREevd9xvGljDYxJYaEny5PVTpc3ErZzlUg7DR'
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth)


t = Twython(app_key='cEiRuD4KjEoS46ZrvHPe5hkpm', 
    app_secret='mJUyZ8NUhdyO0v71OtBYx46I8pcvo6fhknTmYaCnrkjpV2oEJo',
    oauth_token='887287866-hK6uOncwQl4Dg4HtCuVCJBwciu1LFKGN6innHx5J',
    oauth_token_secret='DBl2ESDWREevd9xvGljDYxJYaEny5PVTpc3ErZzlUg7DR')
   
outputFile = codecs.open("Microsoft.csv", "w+", "utf-8")
outputFile.write('username;friendsCount;followerCount;date;retweets;favorites;permalink;text')

'''
def receiveBuffer(tweets):
	for t in tweets:
	   while True:
	     try:
		user = api.get_user(t.username)
		outputFile.write(('\n%s;%d;%d;"%s";%s;%s;%s;%s' % (t.username, user.friends_count, user.followers_count, 			t.date.strftime("%Y-%m-%d %H:%M"), t.retweets, t.favorites, t.text, t.permalink)))
		outputFile.flush();
		print("One Record written")
	     except tweepy.TweepError:
        	time.sleep(60 * 15)
        	continue
	     break
	print 'More %d saved on file...\n' % len(tweets)
'''
print("starting download...")
day = 1
year = 2017
month = 03
date1 = datetime.date(year,month,day)
#print(date1)
for k in range(60):
	date2 = date1+datetime.timedelta(days=1)
	
	try:
		#print(date2)
		tweetCriteria = got.manager.TweetCriteria().setQuerySearch('SAP').setSince(str(date1)).setUntil(str(date2)).setMaxTweets(5000)
		tweet = got.manager.TweetManager.getTweets(tweetCriteria)
		#print("processing")
		for t in tweet:
			try:
				user = api.get_user(t.username)
				outputFile.write(('\n%s;%d;%d;"%s";%s;%s;%s;%s' % (t.username, user.friends_count, user.followers_count, t.date.strftime("%Y-%m-%d %H:%M"), t.retweets, t.favorites, 					t.permalink, t.text)))
				outputFile.flush();
				#print("One Record written")
			except tweepy.TweepError:
				print('Sleep...')        		
				time.sleep(60 * 15)
				continue
	except:
		continue
	
	print('Records written for date %s' % str(date1))
	date1=date1+datetime.timedelta(days=1)
	#outputFile.flush()


outputFile.close()
print 'Done. Output file generated "SAP.csv".'

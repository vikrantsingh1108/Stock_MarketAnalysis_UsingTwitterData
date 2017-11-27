import csv
import pandas as pd
import re
import nltk
import math
import numpy as np

src = pd.read_csv('IBM.csv',error_bad_lines=False,low_memory=False)
data = src.values

def isEnglish(s):
    try:
		if type(s) == float:
			return True
		s.decode('ascii')
		
    except UnicodeDecodeError:
        return False
    else:
        return True

def processTweet(tweet):
    # process the tweets

    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    #tweet = nltk.stem.PorterStemmer()
    porter_stemmer = nltk.stem.PorterStemmer()
    
    words = nltk.word_tokenize(tweet)
    for i, word in enumerate(words):
		try:
			words[i] = porter_stemmer.stem(word)
		except IndexError:
			words[i] = ""  # set empty string for those words which are not present in english dictionary. e.g. "aed" gives index error
    return tweet


f = csv.writer(open("Sap_output.csv", "wb+"))

for in_row in data:
	if isEnglish(in_row[7]):
		try:
			in_row[7] = processTweet(in_row[7])
		except AttributeError:
			if math.isnan(in_row[7]):		# for not a number in tweet, exclude the row
				continue
		for i,w in enumerate(in_row):
			if str(w) == "nan":
				in_row[i] = ""
		f.writerow(in_row)



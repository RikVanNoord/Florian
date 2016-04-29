#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys,re, datetime,numpy,json
reload(sys)
sys.setdefaultencoding("utf-8")

inFile = sys.argv[1]

## obtain data files

filePer = 'periodic_events_Florian.txt'
fileTypes = 'approved_types_dbpedia.txt'
allTypes = [line.strip() for line in open(fileTypes,'r')]

## Obtain input-file with all labeled examples. This way we know what the important features are. If a word is not in the labeled examples, the classifier cannot learn from it anyway.
## Then it does not have to be added in a new feature.

data_all = [line.strip() for line in open(inFile,'r')]
data = [x for x in data_all if x]

def dump_dict(name, d):
	with open(name, 'w') as f:
		json.dump(d, f)
		f.close()

def addToDict(dictje, counter, lijst):
	for item2 in lijst:
		item = unicode(item2, 'utf-8')
		if item not in dictje:
			dictje[item.strip()] = counter
			counter += 1
	return dictje, counter

def create_periodicity_dict(filePer):
	perDict = dict()
	for line in open(filePer,'r'):
		keywords = line.split('\t')[1].split(',')
		for key in keywords:
			key2 = key.strip()
			if key2 not in perDict:
				perDict[key2] = [line]
			else:
				perDict[key2].append(line)		##multiple options for a keyword
	return perDict

def fixDict(dict1, value):
	counter = 0
	newDict = dict()
	for key, value2 in sorted(dict1.items()):
		newDict[key] = counter + value
		counter += 1
	return (value +len(newDict)), newDict

def getIndices(data):
	counter = [0,0,0,0,0,0]
	count = 0
	
	indexDictKeywords = dict()
	indexDictWords = dict()
	indexDictUser = dict()
	indexDictDateTweet = dict()
	indexDictDateEvent = dict()
	indexDictTypes = dict()
	
	for line in data:
		count += 1
		splitLine = line.split('\t')
		
		keywords = splitLine[3].strip().split(',')
		indexDictKeywords, counter[0] = addToDict(indexDictKeywords, counter[0], keywords)		## create dict for keywords
		
		dateEventString = splitLine[1].strip()
		indexDictDateEvent, counter[4] = addToDict(indexDictDateEvent, counter[4], [dateEventString])	## create dict for dates
		oldTweets = splitLine[5].split('-----')
		allTweetsTemp = splitLine[6].split('-----')
		allTweetsAdded = oldTweets + allTweetsTemp
		
		for tweet in allTweetsAdded:
			if tweet != 'NA':							## if annotated with 'n' extra tweets become just 'NA', skip those
				splitTweet = tweet.strip().split(',')
				
				if len(splitTweet) > 2:					## filter out errors
					neededTweet = ",".join(splitTweet[2:]).split()
					wordsTweet = [x for x in neededTweet if len(x) > 1 or x.isalpha() or x.isdigit()]	## select words
					user = splitTweet[0].strip()
					dateTweetString = splitTweet[1].strip()
					
					if len(user) < 16 and '-' in dateTweetString:			## cheap way to filter out wrong tweets
						## create dicts for words, users, date tweets
						indexDictWords, counter[1] = addToDict(indexDictWords, counter[1], wordsTweet)
						indexDictUser, counter[2] = addToDict(indexDictUser, counter[2], [user])
						indexDictDateTweet, counter[3] = addToDict(indexDictDateTweet, counter[3], [dateTweetString])
		
		## allTypes are all approved DBpedia types so that we can add them
		
		for item in allTypes:
			indexDictTypes, counter[5] = addToDict(indexDictTypes, counter[5], [item])		
			
	
	#printSortedDict(indexDictWords)
	return indexDictKeywords, indexDictWords, indexDictUser, indexDictDateTweet, indexDictDateEvent, indexDictTypes
	
perDict = create_periodicity_dict(filePer)

## first create the periodicity dict so that we can check whether or not event were periodic

perDict = create_periodicity_dict(filePer)

## Now create all different dictionaries. This is to have a feature for each value and also remember which feature is actually contains what information.
## This way, we know that feature 34674 is for example the word 'morgen'. We create dicts for each type of information. We add features for every user, keyword, word, dateTweet, dateEvent and DBpedia.
## I do this in advance before the actual feature extraction. However, I know for a fact that I will never get dictionary errors since I loop over all my data first.
## However, in your case, you will detect a lot of new words/users that are not in one of the dictionaries yet. So I think it needs some general check if the value is actually in the dictionary.
	
indexDictKeywords, indexDictWords, indexDictUser, indexDictDateTweet, indexDictDateEvent, indexDictTypes = getIndices(data)

otherFeatures = 22	## number of 'fixed' features

### Get feature index numbers and put in list. Every feature has its own value for the feature-list, so that we can actually add the word 'morgen' on place 34674.

newValue1, indexDictDateEvent = fixDict(indexDictDateEvent, otherFeatures)		
newValue2, indexDictDateTweet = fixDict(indexDictDateTweet, newValue1)			
newValue3, indexDictUser = fixDict(indexDictUser, newValue2) 					
newValue4, indexDictKeywords = fixDict(indexDictKeywords, newValue3)			
newValue5, indexDictWords = fixDict(indexDictWords, newValue4)					
newValue6, indexDictTypes = fixDict(indexDictTypes, newValue5)	

print newValue6

## dump all dictionaries to file

dump_dict('dicts/indexDictDateEvent.txt',indexDictDateEvent)
dump_dict('dicts/indexDictDateTweet.txt',indexDictDateTweet)
dump_dict('dicts/indexDictUser.txt',indexDictUser)
dump_dict('dicts/indexDictKeywords.txt',indexDictKeywords)
dump_dict('dicts/indexDictWords.txt',indexDictWords)
dump_dict('dicts/indexDictTypes.txt',indexDictTypes)
dump_dict('dicts/perDict.txt',perDict)


	

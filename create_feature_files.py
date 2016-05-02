#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys,re, datetime,numpy, random, csv, math, json
reload(sys)
sys.setdefaultencoding("utf-8")
from scipy.sparse import dok_matrix
from pattern.nl import sentiment
from SPARQLWrapper import SPARQLWrapper, JSON
from collections import Counter
from sklearn import preprocessing, feature_selection
from sklearn.feature_selection import SelectKBest, chi2
import cPickle as pickle

inFile = sys.argv[1]
outFile = sys.argv[2]

## read in the event(s) that need to be put in feature format

data_all = [line.strip() for line in open(inFile,'r')]
data = [x for x in data_all if x]

## read in approved DBpedia types

fileTypes = 'approved_types_dbpedia.txt'
allTypes = [line.strip() for line in open(fileTypes,'r')]

### Functions ####

def load_dict(name):
	with open(name) as infile:
		dictje = json.load(infile) 
	infile.close()
	return dictje

def create_dok_matrix(finalList):
	array_temp2 = numpy.array(finalList)
	array_temp = array_temp2.astype(float)

	## for feature 4 (average days to event) add number of days (for every feature-value) of the biggest negative difference
	## this way we make sure there are no negative feature values (some classifiers can't handle those)

	minimum = min(array_temp[:,4])
	new_row = [float(x + abs(minimum)) for x in array_temp[:,4]]
	array_temp[:,4] = new_row

	array = numpy.ma.masked_values(array_temp, float(missing_value))
	aray,labels = get_labels(array, False)												## get data and labels	
	array = preprocessing.normalize(array, axis=0)										## normalize features	
	
	new_array = array.tolist()
	for x in range(0, len(array)):														## categories get accidently normalized as well, fix that
		new_array[x].append(labels[x])

	new_array_temp2 = numpy.array(new_array)
	new_array_temp = new_array_temp2.astype(float)	
	dok_array = dok_matrix(new_array_temp)

	return dok_array, new_array_temp, labels

def getPeriodicityFeatures(keywordsFixed, keywordScores, dateEvent, perDict, missing_value):
	threshold = 3 ## 3 day threshold
	allPos = []
	
	## check if keyword also occurred in a periodic event
	
	for idx, key in enumerate(keywordsFixed):
		if key in perDict:
			keyScore = keywordScores[idx]
			for item in perDict[key]:
				splitItem = item.split('\t')
				datesTemp =  splitItem[3].split('>')
				dates = [x.strip() for x in datesTemp]
				score = splitItem[2].split(',')[0]
				confidence = splitItem[2].split(',')[1]
				support = splitItem[2].split(',')[2]
				typePer = splitItem[0].split(',')[0].replace('<','')
				for date in dates:
					datePer =  datetime.datetime.strptime(date.strip(),"%Y-%m-%d")
					diff = abs((datePer - dateEvent).days)
					if diff <= 7 :
						typePerAppend = 1 if typePer == 'e' else 2										## typePer is 'e' or 'v' only
						allPos.append([score, keyScore, diff, confidence, support, typePerAppend])		## keep track of all possible periodic events 
			
	if allPos:
		sortedPer = sorted(allPos, key = lambda x: (x[0], x[1],-x[2],x[3],x[4]),reverse=True) 	## sort periodic events by features
		bestPer = ['1'] + sortedPer[0]				## add binary feature that we found something
	else:
		bestPer = ['0'] + [missing_value for x in range(6)]			## else return the missing value
	
	## return the set of periodic features (it turned out only the binary feature was actually informative)
	
	return bestPer

## function to do the disambiguation in DBpedia (provided code)

def buildAnchorHash():
	page = csv.reader(open("page.csv", "r"), delimiter=",")
	pages = dict()
	for row in page :
		if len(row) == 3:
			pages[row[0]] = row[1]

	afile = csv.reader(open("anchor_summary.csv", "r"), delimiter=",")
	anchors = dict()
	for row in afile :
		if len(row) == 2:
			ids = []
			key = row[0]
			refs = row[1]
			altrefs = refs.split(";")
			for ref in altrefs:
				pair = ref.split(":")
				if pair[0] in pages :
					ids.append(pages[pair[0]])
			if len(ids) > 0:
				anchors[key] = ids
	return anchors

def filterResults(allResults):
	keepResults = []
	for result in allResults:
		cleanResult = result.split('/')[-1]
		digits = re.match("[0-9]+",cleanResult[1:])
		if (cleanResult[0] == 'Q' and digits):			## filter weird types that are always a Q followed by a number of digits
			continue
		else:												## fix links that are owl.#Place or owl.#Location
			if '#' in cleanResult:
				cleanResult = cleanResult.split('#')[1]
			keepResults.append(cleanResult)
	return keepResults			
			
		
def runSparql(pages,select, answer, dbpedia, rdfType, sparql, indexDictType, featureList, label): 
	global teller, allTypes
	foundResult = False
	db_list =[]
	categories = ['Sport','Politiek','Uitzending','Publieksevenement','Software','Bijzondere dag','Sociale actie','Celebrity nieuws','Reclame','Overig']
	
	for page in pages:
		finalQuery = select + dbpedia + page + '>' + rdfType + answer		## create the query
		sparql.setQuery(finalQuery)
		sparql.setReturnFormat(JSON)
		results = sparql.query().convert()	
		allResults = []
		for result in results["results"]["bindings"]:						## check results from query
			resultaat = result["answer"]["value"]
			allResults.append(resultaat)
			foundResult = True
			allResults = filterResults(allResults)							## filter the results to only obtain meaningful ones
		for item in allResults:
			if item in indexDictType:										## check if it occurred in the approved types
				db_list.append([str(item), page, categories[label]])
				featureList[indexDictType[item]] += 1						## add in the right location
					
	return featureList, db_list	
					
def fixDbpediaKeywords(keywords):
	pages = []
	
	## fix keywords for dbpedia search, problems with capitals, underscores etc
	
	for key2 in keywords:
		key = key2.strip()
		newKey = key.replace('@','').replace('#','')
		if ' ' in key: 
			keySplit = key.split()
			for x in range(0, len(keySplit)):
				keySplit[x] = keySplit[x][0].upper() + keySplit[x][1:]
			pages.append("_".join(keySplit))	
				
		else:	
			newKey = newKey[0].upper() + newKey[1:]
			pages.append(newKey)
			if len(newKey) < 5:
				addItem = newKey.upper()
				pages.append(addItem)
	return pages			

def fixAmbigiousPages(pages, anchors):
	finalPages = []
	for page in pages:
		if page in anchors:
			addItem = anchors[page][0].replace(' ','_')
			finalPages.append(addItem)
		else: 
			re_page = re.match("[A-Za-z0-9_]+",page)		## filter strange links to avoid sparql errors
			if re_page:
				finalPages.append(page)			
	finalPagesSet = list(set(finalPages))
	return finalPagesSet

def getDbpediaFeatures(keywords, indexDictTypes, featureList, label):
	anchors = buildAnchorHash() ## get the anchors
	
	## set the general query information
	
	sparql = SPARQLWrapper("http://nl.dbpedia.org/sparql")
	select = """SELECT DISTINCT ?answer
		WHERE { """
	answer = answer = """?answer .
		} """	
	dbpedia = '<http://nl.dbpedia.org/resource/'
	rdfType = ' <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> '
	
	pages = fixDbpediaKeywords(keywords)			## get all pages
	finalPages = fixAmbigiousPages(pages, anchors)	## fix ambigious pages (doorverwijspaginas)
	featureList, db_list = runSparql(finalPages, select, answer, dbpedia, rdfType, sparql, indexDictTypes, featureList, label)							## run sparql on all pages
	
	return featureList, db_list

def most_occuring_label(feature,array, f_value, binary):
	labels = []
	#print(feature, f_value)
	for x in range(len(array)):
		if binary:
			#print(array[x][feature])
			if array[x][feature] != 0.0:
				#print(array[x][feature])
				labels.append(array[x][-1])
		else:
			labels.append(array[x][-1])		
	
	max_item = max(set(labels), key=labels.count)		
	total_max = labels.count(max_item)
	
	return max_item, total_max, len(labels)	

def addDateInformation(featureList, beforeAfter, diffTotal, absDiffTotal, totalTweets):
	totalBeforeAfter = beforeAfter[0] + beforeAfter[1] + beforeAfter[2]
	featureList[4] = round(float(diffTotal[0]) / float(diffTotal[1]),2)			#calculate the averages
	featureList[5] = round(float(absDiffTotal[0]) / float(absDiffTotal[1]),2)
	
	## adding absolute and percentual values for before/during/after event
	
	featureList[6] = beforeAfter[0]
	featureList[7] = beforeAfter[1]
	featureList[8] = beforeAfter[2]
	featureList[9] = round(float(beforeAfter[0]) / float(totalBeforeAfter),2)
	featureList[10] = round(float(beforeAfter[1]) / float(totalBeforeAfter),2)
	featureList[11] = round(float(beforeAfter[2]) / float(totalBeforeAfter),2)
	return featureList	

def getDateInformation(dateTweet, dateEvent):
	diffTotal = [0,0]			## calculate diff in dates average
	absDiffTotal = [0,0]		## calculate absolute difference average
	beforeAfter = [0,0,0]
	
	diff = (dateTweet - dateEvent).days		## difference
	diffTotal[0] += diff
	absDiffTotal[0] += abs(diff)
	diffTotal[1] += 1
	absDiffTotal[1] += 1
	
	## split tweets regarding occurence before, during and after events
	
	if diff < 0:
		beforeAfter[0] += 1
	elif diff == 0:
		beforeAfter[1] += 1
	else:
		beforeAfter[2] += 1
	return beforeAfter, diff, diffTotal, absDiffTotal

def get_labels(array, shuffle):
	if shuffle:
		numpy.random.shuffle(array)
	labels = []
	
	for x in range(len(array)):
		labels.append(array[x][-1])
	return array, labels

def get_key(item, dictje):
	foundKey = False
	for key in dictje:
		if dictje[key] == item:
			foundKey = True
			k = key
			break
	if not foundKey:
		print('Niets gevonden voor', item)
	return k
	
def get_feature_information(array,labels):
	features = feature_selection.chi2(array[0:], labels[0:])
	sortList = []
	
	for x in range(len(features[0])):
		if math.isnan(features[0][x]):
			sortList.append([0.0, features[1][x], x])
		else:
			sortList.append([float(features[0][x]), features[1][x], x])

	feature_num = []

	sortedList = sorted(sortList, key = lambda x: (x[0]),reverse=True)
	
	for idx,item in enumerate(sortedList):
		feature_num.append([item[2],item[0]])
	
	return feature_num	
	
def get_event_information(splitLine, categories):
	dateEvent =  datetime.datetime.strptime(splitLine[1].strip(),"%Y-%m-%d")
	eventScore = int(round(float((splitLine[2].strip())),0))
	oldTweets = splitLine[5].split('-----')
	category = categories[int(splitLine[8]) -1]	
	dateEventString = splitLine[1].strip()					## get date information in sparse format
	
	keywords = splitLine[3].strip().split(',')
	keywordScoresTemp = splitLine[4].strip().split(',')					
	keywordsFixed = [x.strip() for x in keywords]
	keywordScores = [x.strip() for x in keywordScoresTemp]
	
	newTweets = splitLine[6].split('-----')
	
	if splitLine[7].strip() == 'j':								## check if we add the extra tweets
		allTweets = oldTweets + newTweets
	else:
		allTweets = oldTweets									## get all tweets we want to use
	
	return dateEvent, eventScore, oldTweets, category, dateEventString, keywords, keywordsFixed, keywordScores, newTweets, allTweets

def show_best_features(array, labels, min_occ = [5,10,20,50], cutoff = 50):

	## get feature information

	feature_numbers = get_feature_information(array,labels)								## get best features by chi-squared value

	feature_names = []
	
	featureList = [x.strip() for x in open('featurelist_categories.txt','r')]          ## so we can print the right name of the feature for the fixed features
	
	nonZeroDict = dict()

	for x in range(len(array[0])):
		non_zero = 0
		for y in range(len(array)):		
			if array[y][x] != 0.0 and not math.isnan(array[y][x]):
				non_zero += 1	
		nonZeroDict[x] = non_zero
				
	c = 0

	categories = ['Sport','Politiek','Uitzending','Publieksevenement','Software','Bijzondere dag','Sociale actie','Celebrity nieuws','Reclame','Overig']
	
	## show best features that occur at least 'value' times
	
	newValue1 = other_features + len(indexDictDateEvent)
	newValue2 = newValue1 + len(indexDictDateTweet)
	newValue3 = newValue2 + len(indexDictUser)
	newValue4 = newValue3 + len(indexDictKeywords)
	newValue5 = newValue4 + len(indexDictWords)
	newValue6 = newValue5 + len(indexDictTypes)
	
	for value in min_occ:
		feature_names = []
		for item in feature_numbers:
			counter = nonZeroDict[item[0]]
			if counter > value:	
				c += 1
				
				## check what kind of feature it is to print extra information regarding the feature. Also print its most occuring category.
				
				if item[0] < 22:
					label, amt, total = most_occuring_label(item[0],array, str(item[1]), False)
					feature_names.append(featureList[item[0]] + ' ' + str(item[1]))
				elif item[0] < newValue1:
					label, amt, total = most_occuring_label(item[0],array, get_key(item[0],indexDictDateEvent), True)
					feature_names.append('DateEvent '+ get_key(item[0],indexDictDateEvent)+ ' ' + str(item[1]) + ' voor ' + categories[int(label)] + ' ' + str(amt) + '/' + str(total) + ' (' + str(round(float(amt) / float(total) * 100,2)) +'%)' )
				elif item[0] < newValue2:
					label, amt, total = most_occuring_label(item[0],array, get_key(item[0],indexDictDateTweet), True)
					feature_names.append('DateTweet '+ get_key(item[0],indexDictDateTweet)+ ' ' + str(item[1])+ ' voor ' + categories[int(label)] + ' ' + str(amt) + '/' + str(total) + ' (' + str(round(float(amt) / float(total) * 100,2)) +'%)')
				elif item[0] < newValue3:
					label, amt, total = most_occuring_label(item[0],array, get_key(item[0],indexDictUser), True)
					feature_names.append('User ' + get_key(item[0],indexDictUser)+ ' ' + str(item[1])+ ' voor ' + categories[int(label)] + ' ' + str(amt) + '/' + str(total) + ' (' + str(round(float(amt) / float(total) * 100,2)) +'%)')
				elif item[0] < newValue4:
					label, amt, total = most_occuring_label(item[0],array, get_key(item[0],indexDictKeywords), True)
					feature_names.append('Keyword ' +get_key(item[0],indexDictKeywords)+ ' ' + str(item[1])+ ' voor ' + categories[int(label)] + ' ' + str(amt) + '/' + str(total) + ' (' + str(round(float(amt) / float(total) * 100,2)) +'%)')
				elif item[0] < newValue5:
					label, amt, total = most_occuring_label(item[0],array, get_key(item[0],indexDictWords), True)
					feature_names.append('Word ' + get_key(item[0],indexDictWords)+ ' ' + str(item[1])+ ' voor ' + categories[int(label)] + ' ' + str(amt) + '/' + str(total) + ' (' + str(round(float(amt) / float(total) * 100,2)) +'%)')
				elif item[0]< newValue6:
					label, amt, total = most_occuring_label(item[0],array, get_key(item[0],indexDictTypes), True)
					feature_names.append('DBpedia' + get_key(item[0],indexDictTypes) + ' ' + str(item[1])+ ' voor ' + categories[int(label)] + ' ' + str(amt) + '/' + str(total) + ' (' + str(round(float(amt) / float(total) * 100,2)) +'%)')
	
		for idx, item in enumerate(feature_names):
			if idx < cutoff:		## only print first few
				print item

def getFeatureValues(indexDictKeywords, indexDictWords, indexDictUser, indexDictDateTweet, indexDictDateEvent, maxList, perDict, indexDictTypes, otherFeatures, missing_value):
	finalList = []
	featureList = maxList * [0]			## create list with zero's
	categories = ['Sport','Politiek','Uitzending','Publieksevenement','Software','Bijzondere dag','Sociale actie','Celebrity nieuws','Reclame','Overig']
	total_db_list = []
	
	## add the actual feature values
	
	for idx,line in enumerate(data):	
		if idx % 50 == 0:
			print idx
		#if idx > 200:
		#	break
		splitLine = line.split('\t')
		if splitLine[8] != 'NA':			## check if it actually was an event (and not a non-event)
			featureList = maxList * [0]	
			
			## get all event information
			
			dateEvent, eventScore, oldTweets, category, dateEventString, keywords, keywordsFixed, keywordScores, newTweets, allTweets = get_event_information(splitLine, categories)
			
			## add 1 (positive) at the right place in the feature-file using the dictionary
			
			featureList[0] = eventScore									## first feature is event score
			featureList[indexDictDateEvent[dateEventString]] = 1	
	
			for keyword in keywordsFixed:
				keyword = unicode(keyword, 'utf-8')
				featureList[indexDictKeywords[keyword]] = 1
				
			allTweetsText = ''
			
			## loop over all tweets to obtain the right values for the tweet-features
			
			for tweet in allTweets:
				splitTweet = tweet.strip().split(',')
				if len(splitTweet) > 2:
					user = splitTweet[0].strip()				## add user information
					dateTweetString = splitTweet[1].strip()
					
					if len(user) < 16 and '-' in dateTweetString:
						neededTweet = ",".join(splitTweet[2:]).split()
						finalTweet = [x.strip() for x in neededTweet if len(x) > 1 or x.isalpha() or x.isdigit()]	## delete everything that is only 1 character and non-letter/digit
						allTweetsText += ' ' + (" ".join(finalTweet))
						keywordsInTweet = 0
						
						for word in finalTweet:
							add_word = unicode(word, 'utf-8')
							featureList[indexDictWords[add_word]] = 1   ## add information about the word (note: can be binary or total number of occurences, right now it is binary)
							if word in keywordsFixed:
								keywordsInTweet += 1				
						
						featureList[1] += keywordsInTweet			## add how often a keyword occured
						
						featureList[2] += len(finalTweet)			## keep track of total number of words as a feature
						
						user = unicode(user, 'utf-8')
						featureList[indexDictUser[user]] += 1		## add username as same way as bag-of-words
						
						## add date information to the featurelist
						
						featureList[indexDictDateTweet[dateTweetString]] += 1
						dateTweet = datetime.datetime.strptime(splitTweet[1].strip(),"%Y-%m-%d")
						beforeAfter, diff, diffTotal, absDiffTotal = getDateInformation(dateTweet, dateEvent)
						
			totalTweets = len(allTweets)
			featureList[3] = round(float(featureList[2]) / float(totalTweets),1)								## add average number of words per tweet
			featureList = addDateInformation(featureList, beforeAfter, diffTotal, absDiffTotal, totalTweets) 	## add all information regarding dates

			senti = sentiment(allTweetsText)						## add sentiment and subjective information
			featureList[12] = senti[0] + 1							## add +1 due to some classifiers not able to handle negative numbers (polarity)
			featureList[13] = senti[1]								## subjectivity
			
			featureList[14] = len(allTweets)						## add number of tweets
			
			per = getPeriodicityFeatures(keywordsFixed, keywordScores, dateEvent, perDict, missing_value)	
			for x in range(0, len(per)):		## add the features
				featureList[x+15] = per[x]
			
			label = int(splitLine[8])-1 		## set label
			
			featureList, db_list = getDbpediaFeatures(keywords, indexDictTypes, featureList, label)
		
			featureList.append(int(splitLine[8]) -1)					## add the label as number
			finalList.append(featureList)								## keep track of the final featureList over all events
			
	return finalList

## load dictionaries

indexDictDateEvent = load_dict('dicts/' + 'indexDictDateEvent.txt')
indexDictDateTweet = load_dict('dicts/' + 'indexDictDateTweet.txt')
indexDictUser = load_dict('dicts/' + 'indexDictUser.txt') 
indexDictKeywords = load_dict('dicts/' + 'indexDictKeywords.txt') 
indexDictWords = load_dict('dicts/' + 'indexDictWords.txt') 
indexDictTypes = load_dict('dicts/' + 'indexDictTypes.txt')
perDict = load_dict('dicts/' + 'perDict.txt')  

other_features = 22
missing_value = 1234567			## create artificial missing value to indicate masked values in numpy
max_list = len(indexDictDateEvent) + len(indexDictDateTweet) + len(indexDictUser) + len(indexDictKeywords) + len(indexDictWords) + len(indexDictTypes) + other_features

## This is the actual feature extraction.

finalList = getFeatureValues(indexDictKeywords, indexDictWords, indexDictUser, indexDictDateTweet, indexDictDateEvent, max_list, perDict, indexDictTypes, other_features, missing_value)	

final_matrix, array, labels = create_dok_matrix(finalList)

## dump the final array using pickle

with open(outFile, 'wb') as outfile_part:
	pickle.dump(final_matrix, outfile_part, protocol=0)

show_best_features(array, labels) ## show information regarding the best features (optional)	

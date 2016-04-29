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

inFile = sys.argv[1]

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

finalList = getFeatureValues(indexDictKeywords, indexDictWords, indexDictUser, indexDictDateTweet, indexDictDateEvent, newValue6, perDict, indexDictTypes, otherFeatures, missing_value)	

final_matrix, array, labels = create_dok_matrix(finalList)

with open(outFile, 'wb') as outfile_part:
	pickle.dump(dok_array, outfile_part, protocol=0)

show_best_features(array, labels) ## show information regarding the best features (optional)	

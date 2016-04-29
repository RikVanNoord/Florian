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

## load dictionaries

indexDictDateEvent = load_dict('dicts/' + 'indexDictDateEvent.txt')
indexDictDateTweet = load_dict('dicts/' + 'indexDictDateTweet.txt')
indexDictUser = load_dict('dicts/' + 'indexDictUser.txt') 
indexDictKeywords = load_dict('dicts/' + 'indexDictKeywords_.txt') 
indexDictWords = load_dict('dicts/' + 'indexDictWords.txt') 
indexDictTypes = load_dict('dicts/' + 'indexDictTypes.txt')  


other_features = 22
max_list = len(indexDictDateEvent) + len(indexDictDateTweet) + len(indexDictUser) + len(indexDictKeywords) + len(indexDictWords) + len(indexDictTypes) + other_features

print max_list

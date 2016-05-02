#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys,re, numpy, ast, random, math, time, json
from scipy import delete, stats
from sklearn.metrics import *
from sklearn import svm
from sklearn import linear_model
from sklearn import feature_selection
from sklearn import preprocessing

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cross_validation import LeaveOneOut, cross_val_score
from scipy.sparse import dok_matrix
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.multiclass import OutputCodeClassifier
from collections import Counter

## function to obtain array and list of labels

def get_array_and_labels(array, shuffle):
	if shuffle:
		numpy.random.shuffle(array)
	labels = []

	for x in range(len(array)):
		if int(array[x][-1]) == 7 or int(array[x][-1]) == 9:					## filter celebrity news
			labels.append(7.0)													## add "overig" instead
		else:
			labels.append(array[x][-1])		
	
	new_array = numpy.delete(array, -1, 1)
	new_array2 = numpy.delete(new_array, -1, 1)	
	
	labels_int = [int(i) for i in labels]	
	return new_array2, labels_int

## splitting the word features from the other features

def split_array_words(array):
	boundaries = [x.strip() for x in open('word_feature_boundaries.txt','r')]
	b1 = boundaries[0].split()[0]
	b2 = boundaries[0].split()[1]
	
	word_array = []
	other_array = []		
	
	for x in range(len(array)):
		keep_in = []
		keep_out = []
		for y in range(len(array[x])):
			if	(y >= b1 and y < b2):
				keep_in.append(array[x][y])
			else:
				keep_out.append(array[x][y])			
		word_array.append(keep_in)
		other_array.append(keep_out)
	
	word_array_final = numpy.array(word_array).astype(float)	
	other_array_final = numpy.array(other_array).astype(float)
	
	return word_array_final, other_array_final

## function for feature selection, only keeps the best [keep_number] features
	
def delete_bad_features_chi2(array,labels, keep_number):
	if keep_number < len(array[x][0]):	## check if pruning number is possible
		array_new = SelectKBest(chi2, k=keep_number).fit_transform(array, labels)
	else:
		print 'No feature pruning performed (num_features < keep_number)'
		array_new = array
	return array_new

## printing the classification report
	
def do_clf_report(pred, labels, printer):
	pred_list_str = []
	for x in range(len(pred)):
		pred_list_str.append(str(int(pred[x])))
	
	labels_str = [str(x) for x in labels]
	print(printer)
	print(classification_report(labels_str, pred_list_str))	
	

## function for down-sampling, find number of instances we keep (simply returns the value of second biggest class)

def find_keep_samples(array):
	labels = []
	for x in range(len(array)):
		labels.append(array[x][-1])
	
	label_dict = Counter(labels)	
	sorted_list = sorted(label_dict.values(), reverse=True)
	return sorted_list[1]

## svm grid search

def train_svm(labels,array, num_folds, num_jobs, params = 2):
	#obtain the best parameter settings for an svm outputcode classifier
	bestParameters = dict()
	if len(labels) > 2:
		print("outputcodeclassifier")
		#param_grid = {'estimator__C': [0.001, 0.005, 0.01,0.1, 0.5, 1,2.5, 5, 10,15,25, 50,75, 100, 500, 1000],
		#	'estimator__kernel': ['linear','rbf','poly'], 
		#	'estimator__gamma': [0.0005,0.001, 0.002, 0.008,0.016, 0.032,0.064, 0.128,0.256, 0.512, 1.024, 2.048],
		#	'estimator__degree': [1,2,3,4]}
		param_grid = {'estimator__C': [0.001, 0.005],
			'estimator__kernel': ['linear','rbf'], 
			'estimator__gamma': [0.0005,0.001],
			'estimator__degree': [1]}
		model = OutputCodeClassifier(svm.SVC(probability=True))
	else:
		print("svc model")
		param_grid = {'C': [0.001, 0.005, 0.01, 0.5, 1, 5, 10, 50, 100, 500, 1000],
			'kernel': ['linear','rbf','poly'], 
			'gamma': [0.0005, 0.002, 0.008, 0.032, 0.128, 0.512, 1.024, 2.048],
			'degree': [1,2,3,4]}
		model = svm.SVC(probability=True)
	
	paramsearch = RandomizedSearchCV(model, param_grid, cv=num_folds, verbose=2,n_iter = params,n_jobs=num_jobs) 
	print("Grid search...")
	paramsearch.fit(array,numpy.asarray(labels))
	print("Prediction...")
	parameters = paramsearch.best_params_
	
	for parameter in parameters.keys():
		print(parameter + ": " + str(parameters[parameter]) + "\n")
	print("best score: " + str(paramsearch.best_score_) + "\n\n")
	
	#for score in paramsearch.grid_scores_:
	#	print 'mean score:',score.mean_validation_score
	#	print 'list scores:',score.cv_validation_scores
	#train an svm outputcode classifier using the best parameters
	
	if len(labels) > 2:
		test = svm.SVC(probability=True, C=parameters['estimator__C'],
			kernel=parameters['estimator__kernel'],gamma=parameters['estimator__gamma'],
			degree=parameters['estimator__degree'])
		out_test = OutputCodeClassifier(test,n_jobs=1)
		out_test.fit(array,labels)
	else:
		test = svm.SVC(probability=True, C=parameters['C'],
			kernel=parameters['kernel'],gamma=parameters['gamma'],
			degree=parameters['degree'])
		#test.fit(array,labels)
	return test	

## Function for down-sampling the most dominant class (always public event in my case, so therefore we can just check for value 3.0)

def down_sample_array(new_array, keepSamples):
	rest_data = [] 
	publiek_data = []
	
	for x in range(len(new_array)):
		if int(new_array[x][-1]) == 3.0:			## if public event add to publiek_data
			publiek_data.append(new_array[x])
		else:																	
			rest_data.append(new_array[x])
	
	numpy.asarray(publiek_data)
	numpy.asarray(rest_data)
	numpy.random.shuffle(publiek_data)		   ## randomize the downsampling	
	
	keep_publiek = publiek_data[0:keepSamples] ## downsampling happens here
	down_array = keep_publiek + rest_data
	numpy.random.shuffle(down_array)		   ## randomize final downsampled array (not sure if necessary)
	
	labels = []
	
	for x in range(len(down_array)):		   ## get labels again
		labels.append(down_array[x][-1])
	
	new_array = numpy.delete(down_array, -1, 1)	## remove labels from feature set
	
	labels_int = [int(i) for i in labels]	
	return new_array, labels_int

def add_clf_features(array, clf_list):
	for x in range(len(set(list(labels)))):					## add binary feature for each category
		add_list = []
		for item in clf_list:
			if item == x:
				add_list.append(1.0)
			else:
				add_list.append(0.0)
		cat_array = numpy.array(add_list).reshape(len(add_list),1)
		print array.shape
		print cat_array.shape
		print 'done'
		array = numpy.append(array, cat_array, axis = 1)			## add binary feature to feature set
	return array	
	
## I made my own functions for cross validation, since the built-in CV functions do not let you inspect all the output you want (especially necessary for the secondary social actions)

def get_cv_data_labels(fold_list, label_list, nr):	
	train_data = []
	train_labels = numpy.asarray([])
	
	for x in range(len(fold_list)):
		if x == nr:
			test_data = fold_list[x]
			test_labels = label_list[x]
		else:
			temp_list = list(fold_list[x])
			train_data += temp_list
			train_labels = numpy.append(train_labels, label_list[x])
	
	train_data_array = numpy.asarray(train_data)
		
	return train_data_array, test_data, train_labels, test_labels			
	
def cross_validation_own(array, labels, num_folds, down, test, print_res):
	fold_list = []
	label_list = []
	list_num = 0
	
	if down:
		col_labels = numpy.asarray(labels).reshape(len(labels),1)									## reshape
		array_with_label = numpy.append(array, col_labels, axis = 1)				## add labels	
		keepSamples = find_keep_samples(array_with_label)							## find number of samples we keep
		array, labels = down_sample_array(array_with_label, keepSamples)			## create randomized down-sampled array		
	
	
	for x in range(0, num_folds):
		list_num_new = list_num + int(len(array) / num_folds)		## keep track of the division in equal parts
		array_part = array[list_num:list_num_new]					## actual division for array and labels
		label_part = labels[list_num:list_num_new]
		fold_list.append(array_part)								## add the parts in list of lists
		label_list.append(label_part)
		list_num = list_num_new										## update where we are in dividing the data
		print list_num
	
	pred_list = []
	
	for x in range (len(fold_list)):
		print 'do i even get here'
		train_data, test_data, train_labels, test_labels = get_cv_data_labels(fold_list, label_list,x)		## obtain train and test data
		test.fit(train_data, train_labels)
		pred = test.predict(test_data)
		pred_temp = list(pred)
		pred_list += pred_temp
		print len(pred_list)
	
	labels = labels[0:list_num_new]  ## delete labels that were just outside X equal folds (sometimes losing few instances, it is possible to save them and classify with leave-one-out anyway, or simply add them to last part)
	
	## print lot of information regarding the results
	
	if print_res:
		print 'Accuracy:',accuracy_score(list(labels), pred_list)
		print 'f1-weighted:',f1_score(list(labels),pred_list, average='weighted',pos_label = None),'\n' 
		print 'precision-weighted:',precision_score(list(labels),pred_list, average='weighted',pos_label = None),'\n' 
		print 'recall-weighted:',recall_score(list(labels),pred_list, average='weighted',pos_label = None),'\n'    
		
		do_clf_report(pred_list, list(labels), 'Classification report:\n')
	
		#print 'Predictions:'			 ## print actual predictions
		#pred_c = Counter(pred_list)
		#for key in pred_c:
		#	print 'Cat',key,':', pred_c[key]
	
	return pred_list	


#### Main

## Ik heb nog geen mooie argumentenstructuur gemaakt voor welke test, downsamplen, num_folds voor CV, etc. Dat wil je (dacht ik) toch het liefst zelf regelen.

inFile = sys.argv[1]

down_sample = True
print_res = True
shuffle_data = True
num_folds = 5
num_jobs = 16 			## for svm number of parallel jobs

array = numpy.load(inFile)
array, labels = get_array_and_labels(array, shuffle_data)		## obtain data
array = preprocessing.normalize(array, axis=0)					## normalize feature values

## For doing bag-of-words in advance, we need to know what the word-features are. I did this by just saving a while with the boundaries after creating the dictionaries.
## This means that it is important to only do this when the feature-file is obtained by using the latest version of the dictionaries.

word_array, other_array = split_array_words(array)

## Different tests

## Bayes normal

test = MultinomialNB()
pred = cross_validation_own(array, labels, num_folds, down_sample, test, print_res)

## SVM

#test = train_svm(labels,array, num_folds, num_jobs)		## grid search (takes a long time usually)
#cross_validation_own(array, labels, num_folds, down_sample, test, print_res)

## Bag-of-words in advance for Bayes (only change test variable to do so for SVM)

test = MultinomialNB()
pred = cross_validation_own(word_array, labels, num_folds, down_sample, test, False) ## first do only words, don't print
clf_array = add_clf_features(other_array, pred)
cross_validation_own(clf_array, labels, num_folds, down_sample, test, print_res)


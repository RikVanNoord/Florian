#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys,re, numpy, ast, random, math, time, json
from scipy import delete, stats
from sklearn import metrics
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
	print len(array)
	print len(array[0])
	for x in range(len(array)):
		if int(array[x][-1]) == 7 or int(array[x][-1]) == 9:					## filter celebrity news
			labels.append(7.0)													## add "overig" instead
		else:
			labels.append(array[x][-1])		
	
	new_array = numpy.delete(array, -1, 1)
	new_array2 = numpy.delete(new_array, -1, 1)	
	
	labels_int = [int(i) for i in labels]	
	return new_array2, labels_int

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
		col_labels = labels.reshape(len(labels),1)									## reshape
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
	
	pred_list = []
	
	for x in range (len(fold_list)):
		train_data, test_data, train_labels, test_labels = get_cv_data_labels(fold_list, label_list,x)		## obtain train and test data
		test.fit(train_data, train_labels)
		pred = test.predict(test_data)
		pred_temp = list(pred)
		pred_list += pred_temp
	
	labels = labels[0:list_num_new]  ## delete labels that were just outside X equal folds (sometimes losing few instances, it is possible to save them and classify with leave-one-out anyway, or simply add them to last part)
	
	## print lot of information regarding the results
	
	if print_res:
	
		print 'Accuracy:',accuracy_score(list(labels), pred_list)
		print 'f1-weighted:',f1_score(list(labels),pred_list, average='weighted',pos_label = None),'\n' 
		print 'precision-weighted:',precision_score(list(labels),pred_list, average='weighted',pos_label = None),'\n' 
		print 'recall-weighted:',recall_score(list(labels),pred_list, average='weighted',pos_label = None),'\n'    
		
		print precision_recall_fscore_support(list(labels), pred_list, average='micro', pos_label=None)
		do_clf_report(pred_list, list(labels), 'Classification report:\n')
	
		#print 'Predictions:'			 ## print actual predictions
		#pred_c = Counter(pred_list)
		#for key in pred_c:
		#	print 'Cat',key,':', pred_c[key]


#### Main

## Ik heb nog geen mooie argumentenstructuur gemaakt voor welke test, downsamplen, num_folds voor CV, etc. Dat wil je (dacht ik) toch het liefst zelf regelen.

inFile = sys.argv[1]

down_sample = True
print_res = True
shuffle_data = True
num_folds = 5

dok_array = numpy.load(inFile)
array = dok_array.todense()
array, labels = get_array_and_labels(array, shuffle_data)		## obtain data
array = preprocessing.normalize(array, axis=0)						## normalize feature values

test = MultinomialNB()
cross_validation_own(array, labels, num_folds, down_sample, test, print_res)		
	

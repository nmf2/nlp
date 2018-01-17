#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:00:32 2018

@author: nmf2
"""
import nltk
import random
from nltk.corpus import names

# 1.1   Gender Identification

# Get sample names

labeled_names = ( [(name, 'male') for name in names.words('male.txt')] +
                  [(name, 'female') for name in names.words('female.txt')] )

random.shuffle(labeled_names)


# Define feature extractor function.

def gender_features(word):
    return {
            'last_letter': word[-1],
            'fisrt_letter': word[0],
            'last_two_leters': word[-2:],
            'last_three_leters': word[-3:]
            }


# Separate training and testing sets
  	
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier, test_set))

classifier.show_most_informative_features(15)
- [Learning to Classify Text](#learning-to-classify-text)
    - [Supervised Classification](#supervised-classification)
        - [Gender Identification](#gender-identification)
        - [Choosing The Right Features](#choosing-the-right-features)
        - [POS Tagging](#pos-tagging)
        - [Sequence Classification](#sequence-classification)
    - [Further Examples of Supervised Classification](#further-examples-of-supervised-classification)
        - [Sentence Segmentation](#sentence-segmentation)
    - [Decision Trees](#decision-trees)
    - [Naive Bayes Classifiers](#naive-bayes-classifiers)

# Learning to Classify Text

Let the Machine Learning begin. :)

## Supervised Classification

Classification is the task of choosing the correct class label for a given input.
The basic classification task has a number of interesting variants. For example, in multi-class classification, each instance may be assigned multiple labels; in open-class classification, the set of labels is not defined in advance; and sequence classification involves the algorithmic assignment of a categorical label to each member of a sequence of observed values.

A classifier is called supervised if it is built based on training corpora containing the correct label for each input.

![alt text](http://www.nltk.org/images/supervised-classification.png)

### Gender Identification

As an exercise let's build a gender classifier for words.
First it's necessary to decide which **features** of the input will be used and how to **encode** them. For this classifier we'll use the last latter of the words as the only feature.

Using this feature with a Naive Bayes classifier (seen later) and train data it's possible to get 77% of accuracy. Adding new features such as the last two letters of the name and the first letter it's possible to get 84% of accuracy.

### Choosing The Right Features

Identifying the right features and deciding how to represent them is one of the most important steps in building a classifier because they have great impact on the performance of the classifier given that they are the "parameters" used to classify the data.

An important step in choosing features is to do some **error analysis** on the trained classifier. It consists of manually analyzing the errors made by the classifier to find patterns which could lead to removing or adding features. The available data should be divided between the training set and the test set. The training set should be further divided between train data and dev-test data. This way the classifier can be trained with the train data and then tested with the dev-test data for error analysis. After each dev-test session the training set (and **only** the training set) should be scrambled and re-split. Only when the model is ready (the features have been chosen) should the test set be used to check the performance of the classifier, thus is very important to leave the test set untouched since the very beginning.

### POS Tagging

For POS Tagging, the context should most certainly be taken into account so one way to build a classifier is to use, apart from suffixes, the previous and next words as features for the classifier.
This yields an accuracy of ~79%.

### Sequence Classification

In POS Tagging **consecutive classification** or **greedy sequence classification** consists of finding the most likely tag for the first input the use that to find the tag for the next input. This approach was used by the bigram tagger.

In order to implement this the feature extractor function needs to have a history of the last tags for each word.

## Further Examples of Supervised Classification

### Sentence Segmentation

This consists of a classification task to determine whether a sentence has ended or not based on punctuation analysis.

It's necessary to get some data which has already been segmented into sentences. To make it suitable for extracting features there should be a *tokens* list and a *boundaries* set to save the index where one sentence begins and another ends. *tokens* should be a merged list of tokens from the individual sentences.

Features such as if the next word is capitalized, the previous word, the punctuation itself and if the previous word is one char are quite useful and yield an accuracy of ~94%.

## Decision Trees

A decision tree is a simple flowchart that selects labels for input values. This flowchart consists of decision nodes, which check feature values, and leaf nodes, which assign labels. To choose the label for an input value, we begin at the flowchart's initial decision node, known as its root node. This node contains a condition that checks one of the input value's features, and selects a branch based on that feature's value. Following the branch that describes our input value, we arrive at a new decision node, with a new condition on the input value's features. We continue following the branch selected by each node's condition, until we arrive at a leaf node which provides a label for the input value.

![alt text](http://www.nltk.org/images/decision-tree.png)

In order to understand how to build a decision tree that models a given training set first we need to understand what is a *decision stump*.
A decision stump is a decision tree with a single node that decides how to classify inputs based on a single feature. The easiest approach to decide which feature to use is to simply create a decision stump for each possible feature, and see which one achieves the highest accuracy on the training data.


## Naive Bayes Classifiers

In naive Bayes classifiers, every feature gets a say in determining which label should be assigned to a given input value. To choose a label for an input value, the naive Bayes classifier begins by calculating the prior probability of each label, which is determined by checking frequency of each label in the training set. The contribution from each feature is then combined with this prior probability, to arrive at a likelihood estimate for each label. The label whose likelihood estimate is the highest is then assigned to the input value

Process:
1. Calculate probability of the label by checking the frequency of the label in the training set.
2. Calculate the contribution of each feature
3. Combine the previous two and the label with highest likelihood is the one.

Individual **features** make their contribution by "voting against" labels tat don't occur with that **feature** very often. The **likelihood for each label** is reduced by multiplying it by the probability that an input value with that **label** would have the **feature**.

![alt text](http://www.nltk.org/images/naive_bayes_bargraph.png)


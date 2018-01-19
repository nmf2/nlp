- [Learning to Classify Text](#learning-to-classify-text)
    - [Supervised Classification](#supervised-classification)
        - [Gender Identification](#gender-identification)
        - [Choosing The Right Features](#choosing-the-right-features)
        - [POS Tagging](#pos-tagging)

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


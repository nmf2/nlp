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


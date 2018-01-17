# Categorizing and Tagging Words

## POS-Tagger

Processes a sequence of words attaching part of speech tags to each word.

    This:
    "And now for something completely different"

    Becomes this:
    [('They', 'PRP'), ('refuse', 'VBP'), ('to', 'TO'), ('permit', 'VB'), ('us', 'PRP'),
    ('to', 'TO'), ('obtain', 'VB'), ('the', 'DT'), ('refuse', 'NN'), ('permit', 'NN')]

In NLKT first we tokenize the text and then tag it.
***

## Tagged Corpora

### Universal POS Tagset

| Tag  | Meaning             | English Examples                       |
| ---- | ------------------- | -------------------------------------- |
| ADJ  | adjective           | new, good, high, special, big, local   |
| ADP  | adposition          | on, of, at, with, by, into, under      |
| ADV  | adverb              | really, already, still, early, now     |
| CONJ | conjunction         | and, or, but, if, while, although      |
| DET  | determiner, article | the, a, some, most, every, no, which   |
| NOUN | noun                | year, home, costs, time, Africa        |
| NUM  | numeral             | twenty-four, fourth, 1991, 14:24       |
| PRT  | particle            | at, on, out, over per, that, up, with  |
| PRON | pronoun             | he, their, her, its, my, I, us         |
| VERB | verb                | is, say, told, given, playing, would   |
| .    | punctuation marks   | . , ; !                                |
| X    | other               | ersatz, esprit, dunno, gr8, univeristy |
***

## Automatic Tagging

The tag of a word depends on the word and its context within a sentence.

### The Default Tagger

The simplest tagger assigns the same tag to all the words. The most likely tag, that is, the tag that has the maximum frequency. The **DefaultTagger** in the **nltk** module does exactly that.
This yields aproximately 13% of accuracy.

### The Regex Tagger

These taggers use patterns to tag words. Such as:
``` python
patterns = [
     (r'.*ing$', 'VBG'),               # gerunds
     (r'.*ed$', 'VBD'),                # simple past
     (r'.*es$', 'VBZ'),                # 3rd singular present
     (r'.*ould$', 'MD'),               # modals
     (r'.*\'s$', 'NN$'),               # possessive nouns
     (r'.*s$', 'NNS'),                 # plural nouns
     (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
     (r'.*', 'NN')                     # nouns (default)
]
```
This yields aproximately 20% of accuracy.

### The Lookup Tagger

First let's find the most frequent words and save their most likely tags (from a corpus) and then use this information as a model for a "lookup tagger" (an NLTK *UnigramTagger*). Here's the how to:

1. Take the frequency distribution of the words of the curpus.
2. Take the conditional frequency distribution of the corpus with the words as the conditions for the tags.
3. Take the X most frequent words (i.e. X = 100)
4. Now for each word in the list of most frequent words assign the most likely tag to the word (i.e. the tag that has been assigned to that word most of the time)
5. Now use this as a model for the UnigramTagger.

This tagger yields an accuracy of ~45%.
***

## N-Gram Tagging

### Unigram Tagger

These taggers use a simple statistical algorithm: for each token assign the tag most like for that token. This tagger behaves just like a Lookup Tagger except this tagger can be *trained* rather than folloing a model.
The process to train is simple: provide tagged sentences data as a parameter when initializing the tagger. Then the same process of made to create a Lookup Tagger is made automaticaly.

### Separating Training and Testing Data

The data used to train the tagger can't be the same to test the tagger. Otherwise the tagger will just "memorize" it all and do great on the testing. Hence the data should be split with some proportion such as 90% for training and 10% for testing.

### General N-Gram Tagging

In Unigram Tagging only the word to be tagged is used as a parameter for the tagging, the word is the only piece of context that's used.

The context of a N-Gram tagger is the current word together with the N-1 POS-tags that came before it. For example, if we're using a 4-gram tagger, the curret word is *car* and the last 3 POS-tags were PRON, VERB, DET then PRON, VERB, DET and car will be used to tag *car*.

### Combining Taggers

N-Gram taggers can be combined to get a satisfactory result. For example, a Default Tagger can be used in the begginig, then a UnigramTagger with the DefaultTagger as backoff then a BigramTagger with the UnigramTagger as backoff, etc..

### Unknown words

In this situation a special tag "UNK" should be used so the taggers can learn how to deal with it. For example: in the beggining the tagger would probably tag UNK as nouns but if it has a the word "to" before it then it would likely be tagged as a verb (i.e. "the blog" and "to blog", the same unknown word should be tagged differently).

### Storing Taggers

Tagging can take a significant amout of time depending on the input. It's possible to store them and rapidly recover the with *prickle*:

```python
from pickle import dump
output = open('POS-Tagger.pkl', 'wb')
dump(pos_tagger, output, -1)
output.close()
```

To get it back:

```python
from pickle import load
input = open('POS-tagger.pkl', 'rb')
pos_tagger = load(input)
input.close()
```

***

## Transformation-Based Tagging

A a few issues can be found in n-gram tagging: the size of the n-gram's table, and the lack of context. And approach appointed 
to solve these issues is called Brill tagging or *transformation-based learning*.

The idea of this kind of tagging is quite simple: guess the tag of every word, then go back and fix the mistakes. With n-gram we use *supervised learning* since we use some data to train the tagger, with Brill Tagging instead of counting observations a list of transformational correction rules are used.

We can use a unigram tag to first guess the tags of the words and then apply a set of rules such as (a) Replace NN with VB when the previous word is TO; (b) Replace TO with IN when the next tag is NNS. Let's se an example:

|             |     |          |        |     |        |     |            |                |
| ----------- | --- | -------- | ------ | --- | ------ | --- | ---------- | -------------- |
| **Phrase**  | to  | increase | grants | to  | states | for | vocational | rehabilitation |
| **Unigram** | TO  | NN       | NNS    | TO  | NNS    | IN  | JJ         | NN             |
| **Rule 1**  |     | VB       |        |     |        |     |            |                |
| **Rule 2**  |     |          |        | IN  |        |     |            |                |
| **Output**  | TO  | VB       | NNS    | IN  | NNS    | IN  | JJ         | NN             |
| **Gold**    | TO  | VB       | NNS    | IN  | NNS    | IN  | JJ         | NN             |

All the rules in Brill Tagging are of the following type:
Replace T1 with T2 in context C
Usually the context depends on following or previous word.
***

## Summary

* Words can be grouped into classes, such as nouns, verbs, adjectives, and adverbs. These classes are known as lexical categories or parts of speech. Parts of speech are assigned short labels, or tags, such as NN, VB.
* Backoff is a method for combining models: when a more specialized model (such as a bigram tagger) cannot assign a tag in a given context, we backoff to a more general model (such as a unigram tagger).
* Part-of-speech tagging is an important, early example of a sequence classification task in NLP: a classification decision at any one point in the sequence makes use of words and tags in the local context.
* N-gram taggers can be defined for large values of n, but once n is larger than 3 we usually encounter the sparse data problem; even with a large quantity of training data we only see a tiny fraction of possible contexts.
* Transformation-based tagging involves learning a series of repair rules of the form "change tag s to tag t in context c", where each rule fixes mistakes and possibly introduces a (smaller) number of errors.

# Categorizing and Tagging Words

## POS-Tagger

Processes a sequence of words attaching part of speech tags to each word.

    This:
    "And now for something completely different"

    Becomes this:
    [('They', 'PRP'), ('refuse', 'VBP'), ('to', 'TO'), ('permit', 'VB'), ('us', 'PRP'),
    ('to', 'TO'), ('obtain', 'VB'), ('the', 'DT'), ('refuse', 'NN'), ('permit', 'NN')]

In NLKT first we tokenize the text and then tag it.

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

## N-Gram Tagging

### Unigram Tagger

These taggers use a simple statistical algorithm: for each token assign the tag most like for that token. This tagger behaves just like a Lookup Tagger except this tagger can be *trained* rather than folloing a model.
The process to train is simple: provide tagged sentences data as a parameter when initializing the tagger. Then the same process of made to create a Lookup Tagger is made automaticaly.

### Separating Training and Testing Data

The data used to train the tagger can't be the same to test the tagger. Otherwise the tagger will just "memorize" it all and do great on the testing. Hence the data should be split with some proportion such as 90% for training and 10% for testing.

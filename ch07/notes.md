# Extracting Information from Text

This section is about getting structured data from an unstructured text input. That is, how to get entities and intents described in a text.

## Information Extraction

One important form of information is **structured data**, where there is a regular and predictable organization of entities and relationships. For example, to associate a **company** with the **locations** where it does business.

In Python this could be represented as a tuple (entity, relation, entity).

**Information Extraction** consists of converting unstructured data into structured data. To solve this problem it's possible to build a very general representation of meaning, which would be ideal (Chapter 10). For now, it's easier to look for specific kinds of information in text.

### Information Extraction Architecture

![alt text](http://www.nltk.org/images/ie-architecture.png)

A simple IE architecture to process a document is to first to use sentence fragmentation with the raw text, then to tokenize each sentence and feed this into a POS tagger). The next step would be to do **named entity detection**. In this step we search for mentions of potentially interesting entities in each sentence. Finally, we use relation detection to search for likely relations between different entities in the text.

In named entity detection, we segment and label the entities that might participate in interesting relations with one another. Usually, these entities are definite noun phrases (*the knights who say "ni"*) or proper names(*Monty Python*).

## Chunking

This is a technique for **entity detection**. It consists of segmenting and labeling multi-token sequences. Each of the larger boxes in the image is called a **chunk**. Chunking usually selects a subset of the tokens. Also, chunks don't overlap each other.

![alt text](http://www.nltk.org/images/chunk-segmentation.png)

### Noun Phrase Chunking

Also known as **NP-chunking**, it consists of searching for chunks corresponding to individual noun phrases.

One of the most useful sources of information for NP-chunking is POS tags. To demonstrate this well need to first define a **chunk grammar**, which is a set of rules that indicate how sentences should be chunked. 

Initially a single rule shall be used: a NP chunk should be formed whenever the chunker finds an optional determiner (DT) followed by any number of adjectives (JJ) and then a noun (NN).

### Tag Patterns

A chunk grammar uses tag patterns to describe sequences of tagged words. A **tag pattern** is a sequence of POS tags delimited using angle brackets, e.g. <DT>?<JJ>*<NN>. They resemble regex patterns.

### Chinking

Chinking is the process of removing a sequence of tokens from a chunk. A chink is a sequence of tokens that that is not included in a chunk. In the following example barked/VBD at/IN is a chink.

>[ the/DT little/JJ yellow/JJ dog/NN ] barked/VBD at/IN [ the/DT cat/NN ]

If the matching sequence of tokens spans an entire chunk, then the whole chunk is removed; if the sequence of tokens appears in the middle of the chunk, these tokens are removed, leaving two chunks where there was only one before. If the sequence is at the periphery of the chunk, these tokens are removed, and a smaller chunk remains.

|           | Entire chunk            | Middle of a chunk         | End of a chunk          |
| --------- | ----------------------- | ------------------------- | ----------------------- |
| Input     | [a/DT little/JJ dog/NN] | [a/DT little/JJ dog/NN]   | [a/DT little/JJ dog/NN] |
| Operation | Chink "DT JJ NN"        | Chink "JJ"                | Chink "NN"              |
| Pattern   | }DT JJ NN{              | }JJ{                      | }NN{                    |
| Output    | a/DT little/JJ dog/NN   | [a/DT] little/JJ [dog/NN] | [a/DT little/JJ] dog/NN |

### Representing Chunks (IOB Tagging)

Chunk structures can be represented using either tags or trees. The most widespread file representation uses IOB tags. In this scheme, each token is tagged with one of three special chunk tags, I (inside), O (outside), or B (begin). A token is tagged as B if it marks the beginning of a chunk. Subsequent tokens within the chunk are tagged I. All other tokens are tagged O. The B and I tags are suffixed with the chunk type, e.g. B-NP, I-NP. Of course, it is not necessary to specify a chunk type for tokens that appear outside a chunk, so these are just labeled O.

![alt text](http://www.nltk.org/images/chunk-tagrep.png)

## Developing and Evaluating Chunkers

To evaluate chunks it's necessary to have a corpus annotated for the task. The CoNNL corpus has this functionality and thus will be used to evaluate chunkers.

### Simple Evaluation and Baselines

When we use a chunker that creates no chunks (i.e. marks everything with the 'O' IOB tag) we get the baseline accuracy which is that 43.4% of the chunks are tagged with 'O'. And since the tagger didn't find any chunks its precision, recall and f-measure are all zero.

By using a naive regular expression approach we can get fairly good results:

    IOB Accuracy:  87.7%
    Precision:     70.6%
    Recall:        67.8%
    F-Measure:     69.2%

Which can be improved by using a Unigram Tagger:

```python
class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.UnigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
                     in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags
```

    IOB Accuracy:  92.9%
    Precision:     79.9%
    Recall:        86.8%
    F-Measure:     83.2%

### Training Classifier-Based Chunkers

Unfortunately the POS tags of a sentence are not enough to chunk it correctly because sometimes two phrases with the same POS tags must be chunked differently. Thus, to maximize chunking precision it's necessary to use information about the content of the words.



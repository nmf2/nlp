import nltk
# 1. Using a Tagger
text = nltk.word_tokenize("And now for something completely different")
nltk.pos_tag(text)

#
from nltk.corpus import brown
brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)
tag_fd.most_common()
tag_fd.plot(["ADP", "ADV"], cumulative=True)
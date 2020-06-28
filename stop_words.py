# Stop words
# importing stop words from English language.
import spacy
from spacy.lang.en import English

# Setting also the stop words list
stop_words = spacy.lang.en.stop_words.STOP_WORDS
nlp = spacy.load('en')

print('Number of total stop words: %d' % len(stop_words))
print('All stop words: %s' % list(stop_words))

import pandas as pd
import string

df_amazon = pd.read_csv(r"/home/renos/Desktop/datasets_39657_61725_amazon_alexa.tsv", sep="\t")
given_sentence = df_amazon.iloc[2]['verified_reviews']

# Setting the parser into English tokenizer
parser = English()

# Tokenize the text
mytoken_text = parser(given_sentence)
# Showing the tokenized words
print(mytoken_text)
# The punctuation characters
punct = string.punctuation
print(punct)
# Lemmatize the words and convert into lower case words and strip them
# from the empty fileds.
my_tokens = [word.lemma_.lower().strip() for word in mytoken_text]
# Removing the stop words and the punctuations.
my_tokens = [word for word in my_tokens if word not in stop_words and word not in punct]

print(my_tokens)


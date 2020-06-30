# Import libraries from spacy

from spacy.lang.en import English
import string
import spacy
import numpy as np

# Import scikit-learn libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Logistic Regression Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Steps that we need to follow for text prepossessing
punctuations = string.punctuation

# Create our list of stopwords
nlp = spacy.load('en')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()

# Loading TSV file
df_amazon = pd.read_csv(r"/home/renos/Desktop/datasets_39657_61725_amazon_alexa.tsv", sep="\t")
given_sentence = df_amazon.iloc[2]['verified_reviews']
print(df_amazon.shape)
print("1", len(df_amazon[df_amazon.feedback == 1]))
print("0", len(df_amazon[df_amazon.feedback == 0]))
print(df_amazon.feedback.value_counts())

X = df_amazon['verified_reviews']
y = df_amazon['feedback']

from stop_words import tokenizer_text


# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase format
    return text.strip().lower()


# Setting the bag of words
bags_of_words_vector = CountVectorizer(tokenizer=tokenizer_text, ngram_range=(1, 1))

# Creating the TF-iDF vector
tfidf_vector = TfidfVectorizer(tokenizer=tokenizer_text)
# Splitting the data into 80 - 20. Namely, 80% training data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Setting the classifier
classifier = RandomForestClassifier(criterion='entropy', random_state=0)

# Create pipeline for
# 1. Creating the Bag of Words and applying Cleaning the text
# 2. Applying the classification
pipeline = Pipeline([('vectorizer', bags_of_words_vector),
                     ('classifier', classifier)])

# Model generation on our training data
pipeline.fit(X_train, y_train)

# Predicting with testing data set
predict_label = pipeline.predict(X_test)

# Show the model Accuracy
print("Accuracy : {:.2f}%".format(accuracy_score(y_test, predict_label) * 100))

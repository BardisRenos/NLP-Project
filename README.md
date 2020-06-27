# NLP-Project
<p align="center"> 
<img src="https://github.com/BardisRenos/NLP-Project/blob/master/imagesNLP.jpeg" width="350" height="200" style=centerme>
</p>

## Intro


  
**Natural language processing** (NLP) is a subfield of linguistics, computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data.

Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation. Can read more regarding NLP from this [link](https://en.wikipedia.org/wiki/Natural_language_processing) and also this [link](https://www.tutorialspoint.com/artificial_intelligence/artificial_intelligence_natural_language_processing.htm)


## What you can do with this Data ?
You can use this data to analyze Amazon’s Alexa product, discover insights into consumer reviews and assist with machine learning models.You can also train your machine models for sentiment analysis and analyze customer reviews how many positive reviews ? and how many negative reviews ?


## Data set

This dataset consists of 3150 Amazon customer reviews (input text), starts with ratings, date of review, variation, verified_reviews and feedback of various amazon Alexa products like Alexa Echo, Echo dots, Alexa Firesticks etc. for learning how to train Machine for sentiment analysis.


## Tools 

In order to solve this nlp problem. I used python3 as programming language and spacy and scikit-learn libraries. 

* spacy:


spaCy is a free and open-source library for Natural Language Processing (NLP) in Python with a lot of in-built capabilities. It’s becoming increasingly popular for processing and analyzing data in NLP. Unstructured textual data is produced at a large scale, and it’s important to process and derive insights from unstructured data. To do that, you need to represent the data in a format that can be understood by computers. NLP can help you do that. You can read more from this [link](https://realpython.com/natural-language-processing-spacy-python/)


<p align="center"> 
<img src="https://github.com/BardisRenos/NLP-Project/blob/master/spacyImage.webp" width="350" height="200" style=centerme>
</p>

* scikit-learn:

Scikit-learn (formerly scikits.learn and also known as sklearn) is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy. For more information can find from this [link](https://scikit-learn.org/stable/index.html)

<p align="center"> 
<img src="https://github.com/BardisRenos/NLP-Project/blob/master/Scikit_learn_logo_small.svg.png" width="350" height="200" style=centerme>
</p>


### Installation

You have to install the aforementioned libraries.

* spaCy

```shell
  pip install spacy
```

The default model for the English language is en_core_web_sm. Download models and data for the English language:

```shell
  python -m spacy download en_core_web_sm
```

* Scikit-learn

```shell
  pip3 install -U scikit-learn
```
For validation the library insgtallation

```shell
python3 -m pip show scikit-learn # to see which version and where scikit-learn is installed
python3 -m pip freeze # to see all packages installed in the active virtualenv
python3 -c "import sklearn; sklearn.show_versions()"
```

## Analyzing and Processing Text With spaCy

Cleaning up the text data is necessary to highlight attributes that we’re going to want our machine learning system to pick up on. Cleaning (or pre-processing) the data typically consists of a number of steps:


**Step 1:** Word Tokenization & Sentence Tokenization

<p align="justify"> 
The first is called word tokenization, which means breaking up the text into individual words. This is a critical step for many language processing applications, as they often require input in the form of individual words rather than longer strings of text.

Also, If someone wants, it is also possible to break the text into sentences rather than words. This is called sentence tokenization. When performing sentence tokenization, the tokenizer looks for specific characters that fall between sentences, like periods, exclaimation points, and newline characters. For sentence tokenization, we will use a preprocessing pipeline because sentence preprocessing using spaCy includes a tokenizer, a tagger, a parser and an entity recognizer that we need to access to correctly identify what’s a sentence and what isn’t.
</p>

```python
  import pandas as pd

  df_amazon = pd.read_csv(r"/home/renos/Desktop/datasets_39657_61725_amazon_alexa.tsv", sep="\t")

  given_sentence = df_amazon.iloc[2]['verified_reviews']
  print(given_sentence)
  
```
```text
# The given test: 

  Sometimes while playing a game, you can answer a question correctly but Alexa says you got it wrong and answers the same as you.  I like being able to turn       lights on and off while away from home.  
```

```python
  # For word Tokenization
  from spacy.lang.en import English

  # Because we have english corpus. Load English tokenizer and word vectors
  nlp = English()
  my_given_doc = nlp(given_sentence)

  # The "nlp" Object is used to create documents with linguistic annotations.
  list_of_tokens = []
  for tokens in my_given_doc:
      list_of_tokens.append(tokens)

  print(list_of_tokens)
```

```text
# The resultafter the stemming.

  [Sometimes, while, playing, a, game, ,, you, can, answer, a, question, correctly, but, Alexa, says, you, got, it, wrong, and, answers, the, same, as, you, .,  , I, like, being, able, to, turn, lights, on, and, off, while, away, from, home, .]
```


**Step 2:** Text Lemmatization (Lexicon Normalization)

<p align="justify">
Next step is **Lexicon normalization** is another step in the text data cleaning process. In the big picture, normalization converts high dimensional features into low dimensional features which are appropriate for any machine learning model. For our purposes here, we’re only going to look at lemmatization, a way of processing words that reduces them to their roots. Lemmatization is a way of dealing with the fact that while words like connect, connection, connecting, connected, etc. aren’t exactly the same, they all have the same essential meaning: connect. The differences in spelling have grammatical functions in spoken language, but for machine processing, those differences can be confusing, so we need a way to change all the words that are forms of the word connect into the word connect itself.

One method for doing this is called **stemming**. Stemming involves simply lopping off easily-identified prefixes and suffixes to produce what’s often the simplest version of a word. Connection, for example, would have the -ion suffix removed and be correctly reduced to connect. This kind of simple stemming is often all that’s needed, but lemmatization—which actually looks at words and their roots (called lemma) as described in the dictionary—is more precise
</p>

**Step 3:** Removing Stop Words
<p align="justify">
Most text data that we work with is going to contain a lot of words that aren’t actually useful to us. These words, called stopwords, are useful in human speech, but they don’t have much to contribute to data analysis. Removing stopwords helps us eliminate noise and distraction from our text data, and also speeds up the time analysis takes (since there are fewer words to process).
 </p>


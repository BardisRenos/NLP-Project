# NLP-Project
<p align="center"> 
<img src="https://github.com/BardisRenos/NLP-Project/blob/master/imagesNLP.jpeg" width="350" height="200" style=centerme>
</p>

## Intro

**Natural language processing** (NLP) is a subfield of linguistics, computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data.

Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation. Can read more regarding NLP from this [link](https://en.wikipedia.org/wiki/Natural_language_processing) and also this [link](https://www.tutorialspoint.com/artificial_intelligence/artificial_intelligence_natural_language_processing.htm)

## Goal


## Data set


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

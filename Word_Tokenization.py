import pandas as pd

df_amazon = pd.read_csv(r"/home/renos/Desktop/datasets_39657_61725_amazon_alexa.tsv", sep="\t")

given_sentence = df_amazon.iloc[2]['verified_reviews']
print(given_sentence)

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

# sentence tokenization

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = English()

# Create the pipeline 'sentencizer' component
sbd = nlp.create_pipe('sentencizer')

# Add the component to the pipeline
nlp.add_pipe(sbd)

#  The "nlp" Object is used to create documents with linguistic annotations.
document = nlp(given_sentence)

# create list of sentence tokens
list_of_sentences = []
for sent in document.sents:
    list_of_sentences.append(sent.text)
print(list_of_sentences)

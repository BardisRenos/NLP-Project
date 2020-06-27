import spacy

nlp = spacy.load('en_core_web_sm')

# Implement lemmatization on a sentence
lemmatization = nlp(u'compute computer computed computing')

for words in lemmatization:
    print(words.text, words.lemma_)

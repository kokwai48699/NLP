from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import csv
from string import punctuation
import random


with open("imdb_master.csv", "r", encoding="utf-8", errors="replace") as firstfile:
    files = csv.reader(firstfile)
    positiveFiles = []
    negativeFiles = []
    for sentence in files:
        if sentence[3] in "pos":
            positiveFiles.append(sentence[2])
        if sentence[3] in "neg":
            negativeFiles.append(sentence[2])

def clean_sentence(sentence):
    sentence = [[character for character in word.split() if character not in punctuation]for word in sentence]
    return sentence     

cleanedPositiveFiles = clean_sentence(positiveFiles[0:10])
cleanedNegativeFiles = clean_sentence(negativeFiles[0:10])
totalfiles = cleanedNegativeFiles + cleanedPositiveFiles
totalfiles = random.sample(totalfiles,len(totalfiles))

taggedPositiveFiles = [TaggedDocument(sentence, ["positive"])for i, sentence in enumerate(cleanedPositiveFiles)]
taggedNegativeFiles = [TaggedDocument(sentence, ["negative"])for i, sentence in enumerate(cleanedNegativeFiles)]

totalTagged = taggedNegativeFiles + taggedPositiveFiles


model = Doc2Vec(totalTagged, min_count = 1, workers=1, vector_size=3)
model.build_vocab(totalTagged, update=True)
model.train(totalTagged, total_examples=1, epochs=1)

print(model.docvecs.most_similar("positive"))
print(model.docvecs.most_similar("negative"))
    

    

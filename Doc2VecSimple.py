from gensim.models.doc2vec import Doc2Vec, TaggedDocument


file = [["I have a pet"], ["They have a pet"], ["she has no pet"], ["no pet"], ["many have pet"], ["Some have no pet"], ["they no pet"], ["no pet"], ["We have no pet"]]

total = [word.split() for sentence in file for word in sentence]


totalTagged = [TaggedDocument(sentence,[i]) for i, sentence in enumerate(total)]


model = Doc2Vec(totalTagged, min_count = 1, workers=1, vector_size=3)
model.build_vocab(totalTagged, update=True)
model.train(totalTagged,total_examples=1, epochs=1000000)
print(model.wv.most_similar("have"))
print(model.wv.most_similar("Some"))

print(model.docvecs.most_similar(0))
print(model.docvecs.most_similar(8))

#print(model.wv)
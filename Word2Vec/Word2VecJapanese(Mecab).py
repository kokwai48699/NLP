import MeCab
from collections import Counter
import numpy as np
import tensorflow as tf


sentences = open("Data/100.txt", "r", encoding = "utf-8").read().split()
#print(sentences)

totalSentences = []
for sent in sentences:
    totalSentences.append(MeCab.Tagger("-Owakati").parse(sent).rstrip().split(" "))
    
    
def dictSentence(totalSentences):
    sentence = [word for sent in totalSentences for word in sent]
    count = Counter(sentence)
    dicstring = {c:i for i,c in enumerate(count)}
    reverse_dicstring = {i:c for i,c in enumerate(count)}
    return dicstring, reverse_dicstring    

def generateWindowData(totalSentences):
    window_size = 2
    windowData = []
    for lines in totalSentences:
        for index,word in enumerate(lines):
            for words in lines[max(index-window_size,0):min(index+window_size,len(totalSentences)+1)]:
                if words != word:
                    windowData.append([word,words])
    return windowData

def generateOnehotData(windowDataIndex,length):
    temp = np.zeros(length)
    temp[windowDataIndex] = 1
    return temp

sentenceDic, reverseSentenceDic = dictSentence(totalSentences)
lengthSentenceDic = len(sentenceDic)
wordToID = [[sentenceDic[word] for word in sentence] for sentence in totalSentences]
windowData = generateWindowData(totalSentences)


x_train = [] # Input data
y_train = [] # Output data



for wordIndex in windowData:
    x_train.append(generateOnehotData(sentenceDic[wordIndex[0]],lengthSentenceDic))
    y_train.append(generateOnehotData(sentenceDic[wordIndex[1]],lengthSentenceDic))


x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

x = tf.placeholder(tf.float32,[None,lengthSentenceDic])
y_label = tf.placeholder(tf.float32,[None, lengthSentenceDic])

embedding_dim = 10

w1 = tf.Variable(tf.random_normal([lengthSentenceDic,embedding_dim]))
w2 = tf.Variable(tf.random_normal([embedding_dim,lengthSentenceDic]))
b1 = tf.Variable(tf.random_normal([embedding_dim]))
b2 = tf.Variable(tf.random_normal([lengthSentenceDic]))

hidden_representation = tf.add(tf.matmul(x,w1),b1)
prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_representation,w2),b2))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)
n_iters = 10
# train for n_iter iterations
for _ in range(n_iters):
    sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
    print('loss is :', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))
    
vectors = sess.run(w1+b1)
print("For example, our うどん vectors are\n", vectors[sentenceDic["うどん"]])


def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))

def find_closest(word_index, vectors):
    min_dist = 10000 # to act like positive infinity
    min_index = -1
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vector, query_vector)
            min_index = index
    return min_index

print("The closest to うどん would be \n", reverseSentenceDic[find_closest(sentenceDic['うどん'], vectors)])
print("The closest to も would be \n", reverseSentenceDic[find_closest(sentenceDic['も'], vectors)])
print("The closest to 良い would be \n", reverseSentenceDic[find_closest(sentenceDic['昨日'], vectors)])

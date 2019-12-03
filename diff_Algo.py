# -*- coding: utf-8 -*-
"""
Created on Tue Nov 4 2019

@author: tahseenat0
"""
import warnings
warnings.filterwarnings("ignore")
from gensim.models import KeyedVectors

from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from clean_tweet_fun import clean_tweets
from model import brain
import numpy as np
import pandas as pd
%matplotlib inline

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
warnings.filterwarnings("ignore")

# Reproducibility
np.random.seed(1234)

RANDOM_NROWS = 20000
MAX_SEQUENCE_LENGTH = 140
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300

# Splitting the arrays into test (60%), validation (20%), and train data (20%)
TRAIN_SPLIT = 0.7
TEST_SPLIT = 0.2
LEARNING_RATE = 0.1
EPOCHS = 15

CREATED_DATASET = "final-output.csv"

EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin'

collected_tweet_df = pd.read_csv(CREATED_DATASET, encoding="ISO-8859-1", usecols=range(0, 4), nrows=RANDOM_NROWS)

print(collected_tweet_df.head())

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

# Applying the pre processing clean_text function to every element in the depressive tweets and random tweets data.

collected_tweet_arr = [x for x in collected_tweet_df["tweet"]]

X_c = clean_tweets(collected_tweet_arr)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(X_c)

sequences_c = tokenizer.texts_to_sequences(X_c)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data_c = pad_sequences(sequences_c, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data_r tensor:', data_c.shape)

nb_words = min(MAX_NB_WORDS, len(word_index))

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))

for (word, idx) in word_index.items():
    if word in word2vec.vocab and idx < MAX_NB_WORDS:
        embedding_matrix[idx] = word2vec.word_vec(word)

# Assigning labels to the depressive tweets and random tweets data

labels_c = [x for x in collected_tweet_df["label"]]

data_train = data_c[0:int(5000*TRAIN_SPLIT)]
labels_train = labels_c[0:int(5000*TRAIN_SPLIT)]

data_test = data_c[int(5000*TRAIN_SPLIT):int(5000*(TEST_SPLIT+TRAIN_SPLIT))]
labels_test = labels_c[int(5000*TRAIN_SPLIT):int(5000*(TEST_SPLIT+TRAIN_SPLIT))]

data_val = data_c[int(5000*(TEST_SPLIT+TRAIN_SPLIT)):5000]
labels_val = labels_c[int(5000*(TEST_SPLIT+TRAIN_SPLIT)):5000]

#loading model
model = brain(embedding_matrix, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH)

early_stop = EarlyStopping(monitor='val_loss', patience=3)

model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

hist = model.fit(data_train, labels_train, validation_data=(data_val, labels_val), epochs=EPOCHS, batch_size=100,
                 shuffle=True, callbacks=[early_stop])

#predict
labels_c_pred = model.predict(data_test)

labels_pred = np.round(labels_c_pred.flatten())

accuracy = accuracy_score(labels_test, labels_pred)
print("Accuracy: %.2f%%" % (accuracy*100))
print(classification_report(labels_test, labels_pred))
print(roc_auc_score(labels_test,labels_pred))#base model


#logic regression
clf = LogisticRegression(random_state=0, solver='saga', multi_class='ovr').fit(data_train, labels_train)
Y_predit = clf.predict(data_test)
print(sum(Y_predit == labels_test) / len(labels_test))
matrix = confusion_matrix(labels_test,Y_predit)
print(matrix)
print(classification_report(labels_test,Y_predit))
print(roc_auc_score(labels_test,Y_predit))


#Decision tree
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(data_train, labels_train)
pred=clf.predict(data_test)
matrix = confusion_matrix(labels_test,pred)
print(matrix)
print(classification_report(labels_test,pred))
print(roc_auc_score(labels_test,pred))#base model


#naive bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb_model = gnb.fit(data_train, labels_train)
pred = gnb_model.predict(data_test)


matrix = confusion_matrix(labels_test,pred)
print(matrix)
print(classification_report(labels_test,pred))
print(roc_auc_score(labels_test,pred))#base model

#SVM
from sklearn import svm
clf = svm.SVC(gamma='scale')
clf = clf.fit(data_train, labels_train)
pred = clf.predict(data_test)

matrix = confusion_matrix(labels_test,pred)
print(matrix)
print(classification_report(labels_test,pred))
print(roc_auc_score(labels_test,pred))#base model

#KNN
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(data_train, labels_train)
#KNeighborsClassifier(...)
pred = neigh.predict(data_test)

matrix = confusion_matrix(labels_test,pred)
print(matrix)
print(classification_report(labels_test,pred))
print(roc_auc_score(labels_test,pred))#base model
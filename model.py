# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 2019

@author: tahseenat
"""

from keras.models import Model, Sequential
from keras.layers import Conv1D, Dense, LSTM, Embedding, Dropout, MaxPooling1D


def brain(embedding_matrix, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH):
    model = Sequential()
    model.add(Embedding(len(embedding_matrix), EMBEDDING_DIM, weights=[embedding_matrix],
                        input_length=MAX_SEQUENCE_LENGTH, trainable=False))

    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.3))

    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.3))

    model.add(LSTM(3500))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
    return model

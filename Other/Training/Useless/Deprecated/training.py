# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:50:46 2016

@author: Joanna
"""

import numpy as np
import pandas
import time
from time import sleep
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import regularizers
from keras.callbacks import ModelCheckpoint

import theano
import theano.tensor as T

epsilon = 1.0e-9
def custom_objective(y_true, y_pred):
    '''Just another crossentropy'''
    #remove all neutral predictions from y_pred and y_true
    y_true = y_true[y_pred[:,1]!=True]
    y_pred = y_pred[y_pred[:,1]!=True]    

    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    y_pred /= y_pred.sum(axis=-1, keepdims=True)
    cce = T.nnet.categorical_crossentropy(y_pred, y_true)
    return cce

def compute_return (model, X, y):
    try:
        pred = model.predict_classes(X)
    
        pos_pred = pred[pred!=1]
        pos_y = y[pred!=1]
    
        ret = 0.0
        hits = 0
        misses = 0
        for j in range(len (pos_pred)):
            if pos_pred [j]== 0:
                if pos_y[j,0] == True:
                    ret += 5
                    hits += 1
                else:
                    ret += -3
                    misses += 1
            else:
                if pos_y[j,2] == True:
                    ret += 5
                    hits += 1
                else:
                    ret += -3
                    misses += 1
        return ret, hits, misses
    except:
        return 0, 0, 0
#path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
#text = open(path).read().lower()
#print('corpus length:', len(text))

chars = [-1,0,1]
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, stateful=True, batch_input_shape=(batch_size, 14, 1)))
#model.add(Dropout(0.2))
#model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss=custom_objective, optimizer='rmsprop')
try:
    #pass
    model.load_weights ("./trained_model.hdf5")
except:
    pass
test_and_cv_samples = 1000
no_series = 116
batch_size = 32

feat_path = 'C:/Users/Joanna/Desktop/Renato/Data Science/Projects/DeepLearning/DeepLearning Trading/Datasets/Fx/Featured'
parsed_path = 'C:/Users/Joanna/Desktop/Renato/Data Science/Projects/DeepLearning/DeepLearning Trading/Datasets/Fx/Parsed'
label_path = 'C:/Users/Joanna/Desktop/Renato/Data Science/Projects/DeepLearning/DeepLearning Trading/Datasets/Fx/Labeled'

for series_no in range (1, no_series+1,1):
    parsed_filename = 'ccy_hist_ext_'+str(series_no)+'.txt'
    feat_filename = 'ccy_hist_feat_'+str(series_no)+'.csv'     
    label_filename = 'ccy_hist_feat_'+str(series_no)+'.csv' #need to fix this name
    
    my_df = pandas.read_csv(feat_path+'/'+feat_filename, converters={'Change':p2f})
    my_df['Date'] = pandas.to_datetime(my_df['Date'],dayfirst=True)
    my_df.index = my_df['Date']
    my_df = my_df.sort(columns='Date', ascending=True)
    del my_df['Date']
    
    train_df = my_df[:-test_and_cv_samples]
    test_and_cv_df = my_df[-test_and_cv_samples:]
    
    my_df = pandas.read_csv(label_path+'/'+label_filename, converters={'Change':p2f})
    my_df['Date'] = pandas.to_datetime(my_df['Date'],dayfirst=True)
    my_df.index = my_df['Date']
    my_df = my_df.sort(columns='Date', ascending=True)
    del my_df['Date']
    
    train_labels_df = my_df[:-test_and_cv_samples]
    test_and_cv_labels_df = my_df[-test_and_cv_samples:]
    
    sentences = np.zeros ((len(train_df),len(train_df.columns)-1))
    next_chars = np.zeros (len(train_df))
    test_sentences = np.zeros ((len(test_and_cv_df),len(test_and_cv_df.columns)-1))
    test_next_chars = np.zeros (len(test_and_cv_df))
    
    for i in range(len(train_df)):
        sentences[i,:] = np.array(train_df.ix[:,'Close':].irow(i).values)
        next_chars [i] = train_labels_df ['Labels'][i]

    for i in range(len(test_and_cv_df)):
        test_sentences[i,:] = np.array(test_and_cv_df.ix[:,'Close':].irow(i).values)
        test_next_chars [i] = test_and_cv_labels_df ['Labels'][i]
    print('nb sequences:', len(sentences))
    
    print('Vectorization...')
    #X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    X = np.zeros((len(sentences), 14, 1))
    y = np.zeros((len(sentences),len(chars)), dtype=np.bool)
    test_X = np.zeros((len(test_sentences), 14, 1))
    test_y = np.zeros((len(test_sentences),len(chars)), dtype=np.bool)
    
    for i in range(len(sentences)):
        #for t, char in enumerate(sentence):
        #    X[i, t, char_indices[char]] = 1
        #y[i, char_indices[next_chars[i]]] = 1
        for j in range(np.size(sentences,1)):
            sentences[300:,j] = (sentences[300:,j] - np.mean(sentences[300:,j])) / np.std(sentences[300:,j])
        X[i,:,0] = sentences[i,:]
        
        if next_chars[i] == -1:
            y[i,:] = [1,0,0]
        elif next_chars[i] == 0:
            y[i,:] = [0,1,0]
        elif next_chars[i] == 1:
            y[i,:] = [0,0,1]
    
    for i in range(len(test_sentences)):
        #for t, char in enumerate(sentence):
        #    X[i, t, char_indices[char]] = 1
        #y[i, char_indices[next_chars[i]]] = 1
        for j in range(np.size(test_sentences,1)):
            test_sentences[:,j] = (test_sentences[:,j] - np.mean(sentences[300:,j])) / np.std(sentences[300:,j])
        test_X[i,:,0] = test_sentences[i,:]
        
        if test_next_chars[i] == -1:
            test_y[i,:] = [1,0,0]
        elif test_next_chars[i] == 0:
            test_y[i,:] = [0,1,0]
        elif test_next_chars[i] == 1:
            test_y[i,:] = [0,0,1]

    checkpoint = ModelCheckpoint(model, monitor='val_acc', save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    print ("Training will start")
    # Fit the model
    #hist = model.fit(X, y, batch_size=128, nb_epoch=1, callbacks=callbacks_list,
    #     validation_data=(test_X, test_y))
    try:
        print ("Series no: "+ str(series_no))
        for iteration in range(1, 5):
            print()
            print('-' * 50)
            print('Iteration', iteration)
            model.fit(X[300:-np.mod(len(X) - 300, batch_size),:,:], y[300:-np.mod(len(X) - 300, batch_size)], batch_size=batch_size, nb_epoch=1)
                      #validation_data=(test_X[:-np.mod(len(test_X), batch_size),:,:], test_y[:-np.mod(len(test_X), batch_size)]))
            ret, hits, misses = compute_return(model,test_X[:-np.mod(len(test_X),batch_size),:,:], test_y[:-np.mod(len(test_X),batch_size)])
            print ("Return: "+str(ret))
            print ("Hits: "+str(hits))
            print ("Misses: "+str(misses))
        model.save_weights("./trained_model.hdf5",overwrite=True)
        
        model.reset_states()
    except:
        pass
        
#def sample(a, temperature=1.0):
#    # helper function to sample an index from a probability array
#    a = np.log(a) / temperature
#    a = np.exp(a) / np.sum(np.exp(a))
#    return np.argmax(np.random.multinomial(1, a, 1))
#
## train the model, output generated text after each iteration
#for iteration in range(1, 60):
#    print()
#    print('-' * 50)
#    print('Iteration', iteration)
#    model.fit(X[300:,:,:], y[300:], batch_size=128, nb_epoch=1)
#
#    start_index = random.randint(0, len(text) - maxlen - 1)
#
#    for diversity in [0.2, 0.5, 1.0, 1.2]:
#        print()
#        print('----- diversity:', diversity)
#
#        generated = ''
#        sentence = text[start_index: start_index + maxlen]
#        generated += sentence
#        print('----- Generating with seed: "' + sentence + '"')
#        sys.stdout.write(generated)
#
#        for i in range(400):
#            x = np.zeros((1, maxlen, len(chars)))
#            for t, char in enumerate(sentence):
#                x[0, t, char_indices[char]] = 1.
#
#            preds = model.predict(x, verbose=0)[0]
#            next_index = sample(preds, diversity)
#            next_char = indices_char[next_index]
#
#            generated += next_char
#            sentence = sentence[1:] + next_char
#
#            sys.stdout.write(next_char)
#            sys.stdout.flush()
#        print()
#    model.save_weights ("./random_text_gen_weights.hdf5",overwrite="True")

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from collections import defaultdict
import gensim
from sklearn import naive_bayes as nb
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier as mlpc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from numpy import mean
import numpy as np

stopwords = []
project_dir = "../"
stopwords_txt = project_dir + "stopwords"

'''
    for 2-class classification
    dataset not included as it was given by the mentor
'''
# data_dir = "cleaned/"
# classes_dir = ["pos/", "neg/"]
# classes = ["pos", "neg"]

'''
    for 4-class classification
'''
data_dir = "four_class_devanagari/"
classes_dir = ["E/", "A/", "C/", "D/"]
classes = ["E", "A", "C", "D"]

target = []
songs = []

def getCorpus():
    with open(stopwords_txt) as stops:
        for stop in stops:
            stopwords.append(stop.strip())

    for dir in classes_dir:
        folder = project_dir + data_dir + dir
        for file in os.listdir(folder):
            with open(folder + file) as f:
                song = f.read().strip().split()
                song = [word for word in song if word not in stopwords]
                songs.append(song)
                target.append(classes[classes_dir.index(dir)])

getCorpus()
frequency = defaultdict(int)
for song in songs:
    for word in song:
        frequency[word] += 1

songs = [[word for word in song if frequency[word] > 1] for song in songs]

dictionary = gensim.corpora.Dictionary(songs)
n_unique_tokens = len(dictionary)

bag_of_words = [dictionary.doc2bow(song) for song in songs]

#saving for persistency
# gensim.corpora.MmCorpus.serialize('bag_of_words.mm', bag_of_words)
# dictionary.save('dictionary.dict')

#bag of words
dense_bow = gensim.matutils.corpus2dense(bag_of_words, num_terms = n_unique_tokens).transpose()

#tfidf
tfidf = gensim.models.TfidfModel(bag_of_words)
records = tfidf[bag_of_words]
dense_tfidf = gensim.matutils.corpus2dense(records, num_terms = n_unique_tokens).transpose()

#assign the feature set that you want to use for training and classification to dataset
dataset = dense_tfidf

kf = KFold(n_splits = 10, shuffle = True)

accuracies = []
scores = []

for it in range(2):
    print ("Iteration ", it)
    for train, test in kf.split(dataset):
        train_set = []
        train_labels = []
        test_set = []
        test_labels = []
        for i in train:
            train_set.append(dataset[i])
            train_labels.append(target[i])
        for i in test:
            test_set.append(dataset[i])
            test_labels.append(target[i])

        '''
            uncomment the classifier you want to use. Comment out the others
        '''
        # classifier = KNeighborsClassifier()
        # classifier = nb.GaussianNB()
        # classifier = nb.MultinomialNB()
        # classifier = svm.SVC()
        classifier = mlpc(solver = 'lbfgs', hidden_layer_sizes = (5, 15), max_iter = 200)

        predicted = classifier.fit(train_set, train_labels).predict(test_set)

        score = f1_score(test_labels, predicted, average = 'weighted')
        scores.append(score)
        incorrect = (test_labels != predicted).sum()
        accuracy = (len(test_set) - incorrect) / len(test_set) * 100.
        accuracies.append(accuracy)
    print("Maximum accuracy attained ", max(accuracies))
    print("f1score  ", scores[np.argmax(accuracies)])
    print('\n')

print("Maximum accuracy attained ", max(accuracies))
print("f1score  ", scores[np.argmax(accuracies)])

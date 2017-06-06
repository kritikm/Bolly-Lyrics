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

# suffixes = {
#     1: ["ो", "े", "ू", "ु", "ी", "ि", "ा"],
#     2: ["कर", "ाओ", "िए", "ाई", "ाए", "ने", "नी", "ना", "ते", "ीं", "ती", "ता", "ाँ", "ां", "ों", "ें"],
#     3: ["ाकर", "ाइए", "ाईं", "ाया", "ेगी", "ेगा", "ोगी", "ोगे", "ाने", "ाना", "ाते", "ाती", "ाता", "तीं", "ाओं", "ाएं", "ुओं", "ुएं", "ुआं"],
#     4: ["ाएगी", "ाएगा", "ाओगी", "ाओगे", "एंगी", "ेंगी", "एंगे", "ेंगे", "ूंगी", "ूंगा", "ातीं", "नाओं", "नाएं", "ताओं", "ताएं", "ियाँ", "ियों", "ियां"],
#     5: ["ाएंगी", "ाएंगे", "ाऊंगी", "ाऊंगा", "ाइयाँ", "ाइयों", "ाइयां"],
# }
#
# def stem(word):
#     for L in 5, 4, 3, 2, 1:
#         if len(word) > L + 1:
#             for suf in suffixes[L]:
#                 if word.endswith(suf):
#                     return word[:-L]
#     return word


stopwords = []
project_dir = "/home/rkb/Documents/IASNLP/"	#path where the project files are
stopwords_txt = project_dir + "stopwords"

# data_dir = "cleaned/"
# classes_dir = ["pos/", "neg/"]
# classes = ["pos", "neg"]
data_dir = "four_class_data/"				#path where the dataset is
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

#saving for persistency, can be avoided
# gensim.corpora.MmCorpus.serialize('bag_of_words.mm', bag_of_words)
# dictionary.save('dictionary.dict')

# tfidf = gensim.models.TfidfModel(bag_of_words)
# records = tfidf[bag_of_words]
# dense_tfidf = gensim.matutils.corpus2dense(records, num_terms = n_unique_tokens).transpose()
dense_bow = gensim.matutils.corpus2dense(bag_of_words, num_terms = n_unique_tokens).transpose()

kf = KFold(n_splits = 10, shuffle = True)

accuracies = []
scores = []

for it in range(10):
    print ("Iteration ", it)
    for train, test in kf.split(dense_bow):
        train_set = []
        train_labels = []
        test_set = []
        test_labels = []
        for i in train:
            train_set.append(dense_bow[i])
            train_labels.append(target[i])
        for i in test:
            test_set.append(dense_bow[i])
            test_labels.append(target[i])

        classifier = KNeighborsClassifier()
        # classifier = nb.GaussianNB()
        # classifier = nb.MultinomialNB()
        # classifier = svm.SVC()
        # classifier = mlpc(solver = 'lbfgs', hidden_layer_sizes = (15, 5), max_iter = 500)
        predicted = classifier.fit(train_set, train_labels).predict(test_set)

        score = f1_score(test_labels, predicted, average = 'weighted')
        scores.append(score)
        incorrect = (test_labels != predicted).sum()
        accuracy = (len(test_set) - incorrect) / len(test_set) * 100.
        accuracies.append(accuracy)
        # print ("Number of mislabeled points out of a total %d points : %d" %(len(test_set), incorrect))
        # print ("Accuracy : ", accuracy)

print ("Maximum accuracy attained ", max(accuracies))
print ("fscore  ", scores[np.argmax(accuracies)])

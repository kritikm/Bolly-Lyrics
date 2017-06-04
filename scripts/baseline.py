import os
from collections import defaultdict
import gensim
from sklearn import naive_bayes as nb
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier as mlpc
from numpy import mean

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
# frequency = defaultdict(int)
# for song in songs:
#     for word in song:
#         frequency[word] += 1
#
# songs = [[word for word in song if frequency[word] > 1] for song in songs]

dictionary = gensim.corpora.Dictionary(songs)
n_unique_tokens = len(dictionary)

bag_of_words = [dictionary.doc2bow(song) for song in songs]

#saving for persistency, can be avoided
gensim.corpora.MmCorpus.serialize('bag_of_words.mm', bag_of_words)
dictionary.save('dictionary.dict')

tfidf = gensim.models.TfidfModel(bag_of_words)
records = tfidf[bag_of_words]
dense_tfidf = gensim.matutils.corpus2dense(records, num_terms = n_unique_tokens).transpose()
dense_bow = gensim.matutils.corpus2dense(bag_of_words, num_terms = n_unique_tokens).transpose

kf = KFold(n_splits = 10, shuffle = True)

incorrects = []

for train, test in kf.split(dense_tfidf):
    train_set = []
    train_labels = []
    test_set = []
    test_labels = []
    for i in train:
        train_set.append(dense_tfidf[i])
        train_labels.append(target[i])
    for i in test:
        test_set.append(dense_tfidf[i])
        test_labels.append(target[i])

    # gnb_tfidf = nb.GaussianNB()
    # predicted = gnb_tfidf.fit(train_set, train_labels).predict(test_set)

    # gnb_bow = nb.GaussianNB()
    # predicted = gnb_bow.fit(train_set, train_labels).predict(test_set)

    # mnb_tfidf = svm.SVC()
    # predicted = mnb_tfidf.fit(train_set, train_labels).predict(test_set)

    mlp_tfidf = mlpc(solver = 'lbfgs', hidden_layer_sizes = (15, 2), max_iter = 500)
    predicted = mlp_tfidf.fit(train_set, train_labels).predict(test_set)

    #
    # for i in range (len(predicted)):
    #     print (predicted[i], " ", test_labels[i])
    # print ("****************************")

    # bern_tfidf = nb.

    incorrect = (test_labels != predicted).sum()
    incorrects.append(incorrect)
    accuracy = (len(test_set) - incorrect) / len(test_set) * 100.
    print ("Number of mislabeled points out of a total %d points : %d" %(len(test_set), incorrect))
    print ("Accuracy : ", accuracy)

# print ("Average accuracy %d" %((float(len(test_set)) - mean(incorrects)) / len(test_set)))

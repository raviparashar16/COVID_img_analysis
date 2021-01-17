"""
    Experimentation with RFC on a dataset of MRI covid/non-covid images.
"""

import cv2
import numpy as np
import math
import pandas as pd

from DataHandler import DataHandler

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from skimage import feature
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import FastICA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

dh = DataHandler()  # data handler object. See DataHandler.py for more information.

dataset = dh.get_all_data_labeled(shuffle=True)
split_point = int(len(dataset) * .8)  # 80 / 20 split
train_data = dataset[0:split_point]
test_data = dataset[split_point:]

train_X = np.array([dh.load_image(fp, resize=True, grayscale=True) for (fp, label) in train_data])
train_y = np.array([label for (fp, label) in train_data])
test_X = np.asarray([dh.load_image(fp, resize=True, grayscale=True) for (fp, label) in test_data]).astype('uint8')
test_y = np.asarray([np.asarray(label) for (fp, label) in test_data]).astype('float16')

train_X = []
train_y = []
test_X = []
test_y = []
train_X_2d = []
test_X_2d = []

for (fp, label) in train_data:
    img = dh.load_image(fp, resize=True, grayscale=True)  # optionally use noise adder or rotate images
    # also combine into a single array because scikit learn requires that
    img_2d = np.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    img_1d = img_2d.flatten(order="C")  # flatten in row major
    train_X.append(img_1d)
    train_X_2d.append(img_2d)
    train_y.append(label)

for (fp, label) in test_data:
    img = dh.load_image(fp, resize=True, grayscale=True)
    img_2d = np.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    img_1d = img_2d.flatten(order="C")  # flatten in row major
    test_X.append(img_1d)
    test_X_2d.append(img_2d)
    test_y.append(label)

train_X_std = []
for t in train_X:
    t = np.array(t)
    t = (t - t.mean()) / t.std()
    train_X_std.append(t)
train_X = train_X_std

test_X_std = []
for t in test_X:
    t = np.array(t)
    t = (t - t.mean()) / t.std()
    test_X_std.append(t)
test_X = test_X_std

pca = PCA(n_components=150, svd_solver='full')

X_pca = pca.fit_transform(train_X)
testX_pca = pca.fit_transform(test_X)

X_edges = []
for i in train_X_2d:
    edges = feature.canny(i, sigma=1)
    X_edges.append(edges)
testX_edges = []
for i in test_X_2d:
    edges = feature.canny(i, sigma=1)
    testX_edges.append(edges)

X_edges_1d = []
testX_edges_1d = []
for i in X_edges:
    img_1d = i.flatten(order="C")  # flatten in row major
    X_edges_1d.append(img_1d)
for i in testX_edges:
    img_1d = i.flatten(order="C")  # flatten in row major
    testX_edges_1d.append(img_1d)
X_edges = X_edges_1d
testX_edges = testX_edges_1d

# accuracy of using 1-D images as input
clf = RandomForestClassifier()
clf.fit(train_X, train_y)
prediction = clf.predict(test_X)
print("1d image acc: " + str(accuracy_score(test_y, prediction)))

# accuracy using pca
clf = RandomForestClassifier()
clf.fit(X_pca, train_y)
prediction = clf.predict(testX_pca)
print("pca acc: " + str(accuracy_score(test_y, prediction)))

# accuracy using edges
clf = RandomForestClassifier()
clf.fit(X_edges, train_y)
prediction = clf.predict(testX_edges)
print("edges acc: " + str(accuracy_score(test_y, prediction)))

# accuracy using lda
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit(train_X, train_y).transform(train_X)
testX_lda = lda.fit(test_X, test_y).transform(test_X)
clf = RandomForestClassifier()
clf.fit(X_lda, train_y)
prediction = clf.predict(testX_lda)
print("lda acc: " + str(accuracy_score(test_y, prediction)))

# accuracy using ica
ica = FastICA(n_components=1)
X_ica = ica.fit_transform(train_X)
testX_ica = ica.fit_transform(test_X)
clf = RandomForestClassifier()
clf.fit(X_ica, train_y)
prediction = clf.predict(testX_ica)
print("ica acc: " + str(accuracy_score(test_y, prediction)))

# appending lda data and normal image data
X_ldaimg = []
for i in range(len(train_X)):
    X_ldaimg.append(X_lda[i] + train_X[i])
testX_ldaimg = []
for i in range(len(test_X)):
    testX_ldaimg.append(testX_lda[i] + test_X[i])

# # grid search for hyperparameters
# ests = [100, 200, 300, 400, 500, 600, 700]
# crit = ["gini", "entropy"]
# max_d = [None, 100, 500, 1000]
# for i in ests:
#     for j in crit:
#         for k in max_d:
#             clf = RandomForestClassifier(n_estimators=i, criterion=j, max_depth=k)
#             clf.fit(X_ldaimg, train_y)
#             prediction = clf.predict(testX_ldaimg)
#             print("number of estimators: " + str(i) + " criterion: " + j + " max depth: " + str(k))
#             print("edge and lda acc: " + str(accuracy_score(test_y, prediction)))

# accuracy using appended approach
clf = RandomForestClassifier(n_estimators=200, criterion="entropy", max_depth=1000)
clf.fit(X_ldaimg, train_y)
prediction = clf.predict(testX_ldaimg)
print("final model (edge and lda) acc: " + str(accuracy_score(test_y, prediction)))

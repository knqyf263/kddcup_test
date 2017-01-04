#!/usr/bin/python
#coding:utf-8

from __future__ import print_function
import argparse
import math
import sys, time, os

import numpy as np
from numpy.random import *
import six
import pandas

import chainer
from chainer import optimizers
from chainer import serializers
import chainer.functions as F
import chainer.links as L

class Kdd(chainer.Chain):
    def __init__(self, in_units, hidden_units, out_units, train=True):
        super(Kdd, self).__init__(
            l1=L.Linear(in_units, hidden_units),
            l2=L.Linear(hidden_units, out_units),
        )

    def __call__(self, x):
        h1 = self.l1(x)
        y = self.l2(h1)
        return y

# パラメータ設定
in_units = 41
hidden_units = 50
out_units = 5

# テストデータの準備
kdd_data = pandas.read_csv("kddcup99/corrected.csv")

# モデルの準備
kdd = Kdd(in_units, hidden_units, out_units)
# このようにすることで分類タスクを簡単にかける
model = L.Classifier(kdd)
model.compute_accuracy = False

serializers.load_npz('model/my.model', model)

x = chainer.Variable(np.asarray(kdd_data.ix[:, :-1], dtype=np.float32))
#t = chainer.Variable(np.asarray(kdd_data.ix[:, -1], dtype=np.int32))

answer = np.asarray(kdd_data.ix[:, -1])
TP = TN = FP = FN = 0
for i, predicts in enumerate(kdd(x).data):
    predict = np.argmax(predicts)
    if answer[i] == 4:
        if answer[i] == predict:
            TN += 1
        else:
            FP += 1
    else:
        if answer[i] == predict:
            TP += 1
        else:
            FN += 1

#print(1.0 * correct/len(answer))
print("Accuracy: {}".format(1.0 * (TP+TN)/(TP+FP+FN+TN)))
print("Precision: {}".format(1.0 * TP/(TP+FP)))
print("False positive: {}".format(1.0 * FP/(TN+FP)))
print("False negative: {}".format(1.0 * FN/(TP+FN)))





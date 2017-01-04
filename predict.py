#!/usr/local/bin/python
#coding:utf-8

from __future__ import print_function
import argparse
import math, sys, time, os, re

import numpy as np
from numpy.random import *
import six, pandas
from kdd import *

import chainer
from chainer import optimizers
from chainer import serializers
import chainer.functions as F
import chainer.links as L

def option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=20, help='Number of examples in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=39, help='Number of sweeps over the dataset to train')
    parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=650, help='Number of LSTM units in each layer')
    parser.add_argument('--target', '-t', type=str, default="kdd", help='Target name')
    return parser.parse_args()

args = option()

class_name = re.sub("_(.)",lambda x:x.group(1).upper(), args.target.capitalize())
cls = getattr(sys.modules["kdd"], class_name)

# パラメータ設定
in_units = 41
hidden_units = 50
out_units = 5

# テストデータの準備
kdd_data = pandas.read_csv("kddcup99/corrected.csv")

# モデルの準備
kdd = cls(in_units, hidden_units, out_units)
model = L.Classifier(kdd)
model.compute_accuracy = False

serializers.load_npz('model/{}/my.model'.format(args.target), model)

x = np.asarray(kdd_data.ix[:, :-1], dtype=np.float32)
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
print("TP: {} TN: {} FP: {} FN: {}".format(TP, TN, FP, FN))

#!/usr/local/bin/python
#coding:utf-8

from __future__ import print_function
import argparse
import math
import sys, time, os, re

import numpy as np
from numpy.random import *
import six
import pandas
from kdd import *

import chainer
from chainer import optimizers
from chainer import serializers
from chainer import training, datasets
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L

def option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=10000, help='Number of examples in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=5, help='Number of sweeps over the dataset to train')
    parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=650, help='Number of LSTM units in each layer')
    parser.add_argument('--target', '-t', type=str, default="kdd", help='Target name')
    return parser.parse_args()

args = option()

# snake => camel (kdd_relu => KddRelu)
class_name = re.sub("_(.)",lambda x:x.group(1).upper(), args.target.capitalize()) 
cls = getattr(sys.modules["kdd"], class_name)

# パラメータ設定
in_units = 41
hidden_units = 50
out_units = 5
model_file = "model/{}/my.model".format(args.target)
optimizer_file = "model/{}/my.optimizer".format(args.target)

# データの準備
kdd_train_data = pandas.read_csv("kddcup99/train.csv")
kdd_test_data = pandas.read_csv("kddcup99/corrected.csv")

# モデルの準備
kdd = cls(in_units, hidden_units, out_units)
model = L.Classifier(kdd)
#model.compute_accuracy = False

if os.path.isfile(model_file):
    print("model load!!")
    serializers.load_npz(model_file, model)
else:
    for param in model.params():
        data = param.data
        data[:] = np.random.uniform(-0.2, 0.2, data.shape)

# optimizerの設定
optimizer = optimizers.Adam()
optimizer.setup(model)

if os.path.isfile(optimizer_file):
    print("optimizer load!!")
    serializers.load_npz(optimizer_file, optimizer)

batchsize = args.batchsize
epoch = args.epoch
resume = args.resume

# 訓練データの準備
X = np.asarray(kdd_train_data.ix[:, :-1], dtype=np.float32)
Y = np.asarray(kdd_train_data.ix[:, -1], dtype=np.int32)
train = datasets.TupleDataset(X,Y)

# テストデータの準備
X = np.asarray(kdd_test_data.ix[:, :-1], dtype=np.float32)
Y = np.asarray(kdd_test_data.ix[:, -1], dtype=np.int32)
test = datasets.TupleDataset(X,Y)

# Trainerの準備
train_iter = chainer.iterators.SerialIterator(train, batchsize)
test_iter = chainer.iterators.SerialIterator(test, batchsize ,repeat=False, shuffle=False)

updater = training.StandardUpdater(train_iter, optimizer, device=-1)
trainer = training.Trainer(updater, (epoch, 'epoch'), out="result")

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.snapshot(), trigger=(epoch, 'epoch'))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
	['epoch', 'main/loss', 'validation/main/loss',
	'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
trainer.extend(extensions.ProgressBar())

if resume:
    chainer.serializers.load_npz(resume, trainer)

trainer.run()

serializers.save_npz(model_file, model)
serializers.save_npz(optimizer_file, optimizer)

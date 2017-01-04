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

target = "kdd_relu"
class_name = re.sub("_(.)",lambda x:x.group(1).upper(), target.capitalize())
cls = getattr(sys.modules["kdd"], class_name)

# パラメータ設定
in_units = 41
hidden_units = 50
out_units = 5
model_file = "model/{}/my.model".format(target)
optimizer_file = "model/{}/my.optimizer".format(target)

# 訓練データの準備
kdd_data = pandas.read_csv("kddcup99/train.csv")

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

batchsize = 50000
epoch = 5
#resume = "result/snapshot_iter_494"
resume = ""

X = np.asarray(kdd_data.ix[:, :-1], dtype=np.float32)
Y = np.asarray(kdd_data.ix[:, -1], dtype=np.int32)
train, test = datasets.split_dataset_random(datasets.TupleDataset(X,Y), len(Y) - 10000)

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

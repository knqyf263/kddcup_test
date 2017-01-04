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
            l2=L.Linear(hidden_units, hidden_units),
            l3=L.Linear(hidden_units, out_units),
        )

    def __call__(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        y = self.l3(h2)
        return y

def each_slice(arr, n):
    return [arr[i:i + n] for i in range(0, len(arr), n)]

def compute(kdd_mini_data, seq):
    start = time.time()
#    for _,v in kdd_mini_data.iterrows():
#        x = chainer.Variable(np.asarray([v[:-1]], dtype=np.float32))
#        t = chainer.Variable(np.asarray([v[-1]], dtype=np.int32))
#        loss += model(x, t)  # lossの計算
    x = chainer.Variable(np.asarray(kdd_mini_data.ix[:, :-1], dtype=np.float32))
    t = chainer.Variable(np.asarray(kdd_mini_data.ix[:, -1], dtype=np.int32))
    loss = model(x, t)  # lossの計算

    # 最適化の実行
    model.zerograds()
    loss.backward()
    optimizer.update()

    # lossの表示
    cur_end = time.time()
    print("sequence:{}, loss:{}, time:{} sec".format(seq, loss.data, (cur_end - start)))
    serializers.save_npz(model_file, model)
    serializers.save_npz(optimizer_file, optimizer)

# パラメータ設定
in_units = 41
hidden_units = 50
out_units = 5
model_file = "model/2layer/my.model"
optimizer_file = "model/2layer/my.optimizer"

# 訓練データの準備
kdd_data = pandas.read_csv("kddcup99/train.csv")

# モデルの準備
kdd = Kdd(in_units, hidden_units, out_units)
# このようにすることで分類タスクを簡単にかける
model = L.Classifier(kdd)
model.compute_accuracy = False

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


# 訓練を行うループ
mini_batch = 1000000  # 何回ごとに表示するか
total_loss = 0  # 誤差関数の値を入れる変数
loss = 0
seq = 0
epoch = 10000
start = time.time()
for seq in range(epoch):
    sampler = np.random.permutation(len(kdd_data))
    sequence = 0
    for s in each_slice(sampler, mini_batch):
        sequence += len(s)
        compute(kdd_data.take(s), sequence)

serializers.save_npz(model_file, model)
serializers.save_npz(optimizer_file, optimizer)

#!/usr/bin/env python
"""Chainer example: train a multi-layer perceptron on MNIST

This is a minimal example to write a feed-forward net.

"""
from __future__ import print_function
import argparse
import time

import numpy as np
import six

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers

#import data
import net

import sys
sys.path.append("/home/mil/kanayama/chainer/chainer/optimizers/")
import myadam
import mysgd
import mymomentum_sgd

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--net', '-n', choices=('simple', 'parallel'),
                    default='simple', help='Network type')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=20, type=int,
                    help='number of epochs to learn')
parser.add_argument('--unit', '-u', default=1000, type=int,
                    help='number of units')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='learning minibatch size')
args = parser.parse_args()

batchsize = args.batchsize
n_epoch = args.epoch
n_units = args.unit

print('GPU: {}'.format(args.gpu))
print('# unit: {}'.format(args.unit))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('Network type: {}'.format(args.net))
print('')

#Prepare cifar-10 dataset
print('load cifar-10 dataset')
import cPickle
train_dataset_list = []
for i in range(5):
    filename = "/home/mil/kanayama/data/cifar-10-batches-py/data_batch_" + str(i + 1)
    fo = open(filename, 'r')
    d = cPickle.load(fo)
    train_dataset_list.append(d)
    fo.close()

filename = "/home/mil/kanayama/data/cifar-10-batches-py/test_batch"
fo = open(filename, 'r')
test_dataset = cPickle.load(fo)
fo.close()

flag = 0
for d in train_dataset_list:
    if not(flag):
        x_train = d["data"].astype(np.float32).reshape(10000, 3, 32, 32)
        y_train = np.array(d["labels"], dtype = np.int32)
        flag = 1
    else:
        x_train = np.r_[x_train, d["data"].astype(np.float32).reshape(10000, 3, 32, 32)]
        y_train = np.r_[y_train, np.array(d["labels"], dtype = np.int32)]

x_test = test_dataset["data"].astype(np.float32).reshape(10000, 3, 32, 32)
y_test = np.array(test_dataset["labels"], dtype = np.int32)

N = 50000

N_test = y_test.size

# Prepare multi-layer perceptron model, defined in net.py
if args.net == 'simple':
    model = L.Classifier(net.Net2())
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    xp = np if args.gpu < 0 else cuda.cupy
elif args.net == 'parallel':
    cuda.check_cuda_available()
    model = L.Classifier(net.MnistMLPParallel(784, n_units, 10))
    xp = cuda.cupy

# Setup optimizer
#optimizer = myadam.Adam()
#optimizer = mysgd.SGD()
optimizer = mymomentum_sgd.MomentumSGD()
optimizer.setup(model)

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)

# Learning loop
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    start = time.time()
    for i in six.moves.range(0, N, batchsize):
        #print(i)
        x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]]))
        t = chainer.Variable(xp.asarray(y_train[perm[i:i + batchsize]]))

        # Pass the loss function (Classifier defines it) and its arguments
        optimizer.update(model, x, t)
        if epoch == 1 and i == 0:
            with open('graph.dot', 'w') as o:
                variable_style = {'shape': 'octagon', 'fillcolor': '#E0E0E0',
                                  'style': 'filled'}
                function_style = {'shape': 'record', 'fillcolor': '#6495ED',
                                  'style': 'filled'}
                g = computational_graph.build_computational_graph(
                    (model.loss, ),
                    #variable_style=variable_style,
                    #function_style=function_style
                )
                o.write(g.dump())
            print('graph generated')

        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)
    end = time.time()
    elapsed_time = end - start
    throughput = N / elapsed_time
    print('train mean loss={}, accuracy={}, throughput={} images/sec'.format(
        sum_loss / N, sum_accuracy / N, throughput))

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_test, batchsize):
        x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]),
                             volatile='on')
        t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]),
                             volatile='on')
        loss = model(x, t)
        sum_loss += float(loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))

# Save the model and the optimizer
print('save the model')
serializers.save_npz('mlp.model', model)
print('save the optimizer')
serializers.save_npz('mlp.state', optimizer)

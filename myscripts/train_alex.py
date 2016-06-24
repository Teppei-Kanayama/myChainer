#!/usr/bin/env python
# -*- coding:utf-8 -*-

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

import os
from PIL import Image
import random

parser = argparse.ArgumentParser(description='Chainer example: MNIST')

parser.add_argument('train', help='Path to training image-label list file')
parser.add_argument('val', help='Path to validation image-label list file')
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
parser.add_argument('--root', '-R', default='.',
                    help='Root directory path of image files')
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
print('load imagenet dataset')

def load_image_list(path, root):
    tuples = []
    for line in open(path):
        pair = line.strip().split()
        tuples.append((os.path.join(root, pair[0]), np.int32(pair[1])))
    return tuples

train_list = load_image_list(args.train, args.root)
test_list = load_image_list(args.val, args.root)

N = 10000

N_test = len(test_list)
N_test = 1000

# Prepare multi-layer perceptron model, defined in net.py
if args.net == 'simple':
    model = L.Classifier(net.Alex())
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

cropwidth = 256 - 227 #modelにアクセスする方法がわからないのでベタ書き(ほんとは227ではなくmodel.insizeとしたかった)

def read_image(path, center=False, flip=False):
    # Data loading routine                                                                           
    image = np.asarray(Image.open(path))
    if len(image.shape) != 3:
        return None
    image = image.transpose(2, 0, 1)
    if center:
        top = left = cropwidth / 2
    else:
        top = random.randint(0, cropwidth - 1)
        left = random.randint(0, cropwidth - 1)
    bottom = 227 + top
    right = 227 + left

    image = image[:, top:bottom, left:right].astype(np.float32)
    #image -= mean_image[:, top:bottom, left:right]
    image /= 255
    #print(image)
    if flip and random.randint(0, 1) == 0:
        return image[:, :, ::-1]
    else:
        return image


# Learning loop
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N) #データ数分のランダムな順列を生成 #permはndarray
    sum_accuracy = 0
    sum_loss = 0
    start = time.time()
    for i in six.moves.range(0, N, batchsize): 
        print(i)
        x_train = []
        y_train = []
        for j in perm[i:i+batchsize]:
            tmp_img = read_image(train_list[j][0])
            if tmp_img is not None:
                x_train.append(tmp_img)
                y_train.append(train_list[j][1])
        x_train = np.array(x_train)
        y_train = np.array(y_train, dtype = np.int32)

        x = chainer.Variable(xp.asarray(x_train)) #x_train[perm[i:i+batchsize]]は、batchsize枚の画像データをランダムに持ってくる操作
        
        t = chainer.Variable(xp.asarray(y_train))

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
    x_test = []
    y_test = []
    for i in range(N_test):
        tmp_img = read_image(test_list[i][0])
        if tmp_img is not None:
            x_test.append(read_image(test_list[i][0]))                                                                                                                                                                        
            y_test.append(test_list[i][1])      

    for i in six.moves.range(0, N_test, batchsize):
        #for j in range(i, i+batchsize):
        #    x_test.append(read_image(test_list[j][0]))
        #    y_test.append(test_list[j][1])
        
        x = chainer.Variable(xp.asarray(x_test[i:i+batchsize]),
                             volatile='on')
        t = chainer.Variable(xp.asarray(y_test[i:i+batchsize]),
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

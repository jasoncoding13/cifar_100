#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:28:56 2019

@author: jason
@E-mail: jasoncoding13@gmail.com
@Github: jasoncoding13
"""

import numpy as np
import os
import pickle
import tensorflow as tf


def load_data(dataset='train'):
    module_path = os.path.dirname(__file__)
    with open(module_path+'/data/'+dataset, 'rb') as f:
        dct = pickle.load(f, encoding='bytes')
    return dct[b'data'], np.array(dct[b'fine_labels'], dtype=np.int)


def preprocess_data(X, y, n_classes=100):
    n_samples = y.shape[0]
    X = X/255.0
    X = np.reshape(X, [-1, 3, 32, 32])
    X = np.transpose(X, axes=[0, 2, 3, 1])
    X = X.astype(np.float32)
    Y = np.zeros([n_samples, n_classes], dtype=np.uint8)
    Y[np.arange(n_samples), y] = 1
    return X, Y


def conv_relu(input_, filters_shape, strides, scope_name, padding='SAME'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        filters = tf.get_variable(name='filters',
                                  shape=filters_shape,
                                  initializer=tf.initializers.he_uniform())
        biases = tf.get_variable(name='biases',
                                 shape=filters_shape[3],
                                 initializer=tf.constant_initializer(0.01))
        conv = tf.nn.conv2d(input_, filters, strides, padding=padding)
    return tf.nn.relu(conv + biases, name=scope_name)


def max_pool(input_, ksize, strides, scope_name, padding='VALID'):
    with tf.name_scope(scope_name):
        pool = tf.nn.max_pool(input_,
                              ksize=ksize,
                              strides=strides,
                              padding=padding)
    return pool


def fully_connected(input_, dims, scope_name):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        weights = tf.get_variable(name='weights',
                                  shape=dims,
                                  initializer=tf.glorot_normal_initializer)
        biases = tf.get_variable(name='biases',
                                 shape=dims[1],
                                 initializer=tf.constant_initializer(0.0))
        z = tf.matmul(input_, weights) + biases
    return z


def load_mnist():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = (X_train/255.0).astype(np.float32)[:, :, :, np.newaxis]
    X_test = (X_test/255.0).astype(np.float32)[:, :, :, np.newaxis]
    Y_train = np.zeros([60000, 10], dtype=np.uint8)
    Y_train[np.arange(60000), y_train] = 1
    Y_test = np.zeros([10000, 10], dtype=np.uint8)
    Y_test[np.arange(10000), y_test] = 1
    return X_train, Y_train, X_test, Y_test

def identity_block(input_):
    pass

def load_fashion_mnist():
    dct = np.load('/home/jason/d/Repositories/robust_classifier/robust_classifier/data/FashionMNIST0.5.npz')
    X_train = dct['Xtr']
    X_train = (X_train/255.0).astype(np.float32)[:, :, :, np.newaxis]
    X_test = dct['Xts']
    X_test = (X_test/255.0).astype(np.float32)[:, :, :, np.newaxis]
    y_train = dct['Str']
    Y_train = np.zeros([y_train.shape[0], 3])
    Y_train[np.arange(y_train.shape[0]), y_train] = 1
    y_test = dct['Yts']
    Y_test = np.zeros([y_test.shape[0], 3])
    Y_test[np.arange(y_test.shape[0]), y_test] = 1
    return X_train, Y_train, X_test, Y_test
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:32:23 2019

@author: jason
@E-mail: jasoncoding13@gmail.com
@Github: jasoncoding13
"""

import matplotlib.pyplot as plt
import os
import tensorflow as tf
import time
from .utils import load_data, load_mnist, load_fashion_mnist
from .utils import preprocess_data
from .utils import conv_relu, max_pool, fully_connected


class LeNet():

    def __init__(self,
                 n_classes,
                 batch_size=128,
                 learning_rate=0.001,
                 skip_steps=100):
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(name='global_step',
                                       initial_value=tf.constant(0),
                                       trainable=False)
        module_path = os.path.dirname(__file__)
        self.ckpt_path = module_path+'/CheckPoints/lenet'
        self.event_path = module_path+'/Events'
        self.skip_steps = skip_steps

    def _build_data(self):
        """Build dataset and iterator
        """
#        # load data
#        X_train, y_train = load_data('train')
#        X_test, y_test = load_data('test')
#        # reshape and one-hot encoding
#        X_train, Y_train = preprocess_data(X_train, y_train)
#        X_test, Y_test = preprocess_data(X_test, y_test)
#        del y_train
#        del y_test
#        X_train, Y_train, X_test, Y_test = load_mnist()
        X_train, Y_train, X_test, Y_test = load_fashion_mnist()
        self.n_test = X_test.shape[0]
        ds_train = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        ds_test = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
        ds_train = ds_train.shuffle(1000).batch(self.batch_size)
        ds_test = ds_test.batch(1)
        with tf.name_scope('data'):
            self.iterator = tf.data.Iterator.from_structure(
                    ds_train.output_types, ds_train.output_shapes)
            self.X, self.Y = self.iterator.get_next()
            self.init_op_train = self.iterator.make_initializer(ds_train)
            self.init_op_test = self.iterator.make_initializer(ds_test)

    def _build_network(self):
        # [None, 32, 32, 3]
        conv1 = conv_relu(input_=self.X,
                          filters_shape=[5, 5, 1, 6],
                          strides=[1, 1, 1, 1],
                          scope_name='conv1')
        # [None, 32, 32, 6]
        pool1 = max_pool(input_=conv1,
                         ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1],
                         scope_name='pool1')
        # [None, 16, 16, 6]
        conv2 = conv_relu(input_=pool1,
                          filters_shape=[5, 5, 6, 16],
                          strides=[1, 1, 1, 1],
                          scope_name='conv2')
        # [None, 16, 16, 16]
        pool2 = max_pool(input_=conv2,
                         ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1],
                         scope_name='pool2')
        # [None, 8, 8, 16]
        dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
        pool2 = tf.reshape(pool2, [-1, dim])
        self.logits = fully_connected(input_=pool2,
                              dims=[dim, self.n_classes],
                              scope_name='logits')
#        # [None, 8*8*16]
#        fc1 = fully_connected(input_=pool2,
#                              dims=[dim, 120],
#                              scope_name='fc1')
#        # [None, 120]
#        fc2 = fully_connected(input_=fc1,
#                              dims=[120, 84],
#                              scope_name='fc2')
#        # [None, 84]
#        self.logits = fully_connected(input_=fc2,
#                                      dims=[84, self.n_classes],
#                                      scope_name='logits')
#        # [None, 10]

    def _build_loss(self):
        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.Y,
                    logits=self.logits)
            self.loss = tf.reduce_mean(entropy, name='loss')

    def _build_optimizer(self):
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.opti_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def _build_summary(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def _build_evaluation(self):
        with tf.name_scope('evaluation'):
            Y_pred = tf.nn.softmax(self.logits)
            self.accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(Y_pred, 1), tf.argmax(self.Y, 1)), tf.float32))

    def build_graph(self):
        self._build_data()
        self._build_network()
        self._build_loss()
        self._build_optimizer()
        self._build_evaluation()
        self._build_summary()

    def train_one_epoch(self, sess, init_op, saver, writer, global_step):
        start_time = time.time()
        sess.run(init_op)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                batch_loss, _, summary = sess.run([self.loss, self.opti_op, self.summary_op])
                writer.add_summary(summary=summary, global_step=global_step)
                if global_step % self.skip_steps == 0:
                    print('step: {}, batch_loss: {}'.format(global_step, batch_loss))
                global_step += 1
                total_loss += batch_loss
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        saver.save(sess=sess,
                   save_path=self.ckpt_path,
                   global_step=global_step)
        print('average batch loss: {}, time: {:.2}'.format(total_loss/n_batches, time.time()-start_time))
        return global_step

    def eval_one_epoch(self, sess, init_op, writer, global_step):
        start_time = time.time()
        sess.run(init_op)
        total_accuracy = 0
        try:
            while True:
                batch_accuracy, summary = sess.run([self.accuracy, self.summary_op])
                writer.add_summary(summary=summary, global_step=global_step)
                total_accuracy += batch_accuracy
        except tf.errors.OutOfRangeError:
            pass
        print('test accuracy :{}, time: {:.2} '.format(total_accuracy/self.n_test, time.time()-start_time))

    def predict(self):
        pass

    def train(self, n_epochs=100):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        # createa a session
        with tf.Session(config=config) as sess:
            # initialize variables
            sess.run(tf.global_variables_initializer())
            # restore from the checkpoint
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.ckpt_path))
            if ckpt and ckpt.model_checkpoint_path:
                print(f'resotre from {ckpt.model_checkpoint_path}')
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('initialize a new checkpoint')
            # write events
            writer = tf.summary.FileWriter(self.event_path, sess.graph)
            # run global step
            global_step = sess.run(self.global_step)
            for epoch in range(n_epochs):
                self.train_one_epoch(sess=sess,
                                     init_op=self.init_op_train,
                                     saver=saver,
                                     writer=writer,
                                     global_step=global_step)
                self.eval_one_epoch(sess=sess,
                                    init_op=self.init_op_test,
                                    writer=writer,
                                    global_step=global_step)
            writer.close()

    def test(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.init_op_train)
            logits, image = sess.run([self.logits, self.X])
            print(image.shape, logits.shape)
            plt.imshow(image[0].reshape(28, 28))


if __name__ == '__main__':
    lenet = LeNet()
    lenet.build_graph()
    lenet.train()

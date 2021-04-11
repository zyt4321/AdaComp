'''Copyright (c) 2015 – Thomson Licensing, SAS

Redistribution and use in source and binary forms, with or without
modification, are permitted (subject to the limitations in the
disclaimer below) provided that the following conditions are met:

* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of Thomson Licensing, or Technicolor, nor the names
of its contributors may be used to endorse or promote products derived
from this software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
GRANTED BY THIS LICENSE.  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import tensorflow as tf
import Tf_op as df
import Communication as com
import socket as sck
import model.cifar.cifar10 as cifar10
import model.cifar.cifar10_input as cifar10_input
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import sys
import time
import numpy as np
from model.cifar.DataLoader import Dataloader
import os

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

FLAGS = tf.app.flags.FLAGS
from functools import reduce
from operator import mul

import random

class Timer():
    '''
       The EMA Timer
    '''
    def __init__(self):
        self.calcu_time = 0
        self.comm_time = 0
        self.mu = 0.9

    def update_calcu(self, t):
        self.calcu_time = self.mu * self.calcu_time + (1 - self.mu) * t

    def update_commu(self, t):
        self.comm_time = self.mu * self.comm_time + (1 - self.mu) * t

    def get_epsilon(self):
        return self.calcu_time / (self.comm_time + 1e-5)

class SpeedController():
    def __init__(self, init_bs, init_rate):
        self.bs = init_bs
        self.rate = init_rate
        self.BS_list = [16,32,64,128,256]
        self.RATE_list = [0.001,0.005,0.01,0.05,0.1]
        self.i_bs = self.BS_list.index(self.bs)
        self.i_rate = self.RATE_list.index(self.rate)
    def speed_up(self, epsilon):
        # epsilon > 1 : calcu > comm
        can_bs_smaller = self.i_bs > 0
        can_rate_smaller = self.i_rate > 0
        if not can_bs_smaller and not can_rate_smaller:
            pass
        elif epsilon >= 1 and can_bs_smaller:
            self.i_bs -= 1
        elif epsilon >= 1 and can_rate_smaller:
            self.i_rate -= 1
        elif epsilon < 1 and can_rate_smaller:
            self.i_rate -= 1
        elif epsilon < 1 and can_bs_smaller:
            self.i_bs -= 1

        return self.BS_list[self.i_bs], self.RATE_list[self.i_rate]

    def speed_down(self, epsilon):
        # epsilon > 1 : calcu > comm
        can_bs_larger = self.i_bs < len(self.BS_list) - 1
        can_rate_larger = self.i_rate < len(self.RATE_list) - 1
        if not can_bs_larger and not can_rate_larger:
            pass
        elif epsilon >= 1 and can_bs_larger:
            self.i_bs += 1
        elif epsilon >= 1 and can_rate_larger:
            self.i_rate += 1
        elif epsilon < 1 and can_rate_larger:
            self.i_rate += 1
        elif epsilon < 1 and can_bs_larger:
            self.i_bs += 1

        return self.BS_list[self.i_bs], self.RATE_list[self.i_rate]

def get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        print("get_num_params", shape)
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params

def shuffle(images, labels):
    index = [i for i in range(len(labels))]
    np.random.shuffle(index)
    return images[index], labels[index]

# Worker routine
def worker(graph=None, model_name=""):
    """ Build Tensorflow graph and run iterations """
    if graph == None:
        graph = tf.Graph()

    # Build Tensorflow graph which computes gradients of the model with one mini-batch of examples
    with graph.as_default():

        # Get input and labels for learning from D
        # inputs, labels = D
        image_size = 32
        n_channel = 3
        inputs = tf.placeholder(
            dtype=tf.float32, shape=[None, image_size, image_size, n_channel], name='images')
        labels = tf.placeholder(
            dtype=tf.int32, shape=[None], name='labels')
        batch_size = tf.placeholder(
            dtype=tf.int32, shape=[], name='batch_size')

        inputs_processed = cifar10_input.data_preprocess(inputs, batch_size)
        logits = cifar10.inference(inputs_processed, batch_size)

        # Calculate loss.
        loss = cifar10.loss(logits, labels, batch_size)

        optimizer = tf.train.GradientDescentOptimizer(0.1)
        grads = optimizer.compute_gradients(loss)
        with tf.variable_scope("", reuse=True):
            grads_var = {
                var.op.name: tf.Variable(tf.zeros(var.get_shape()), trainable=False, name=var.op.name + "_grad",
                                         collections=["W_grad"]) for _, var in grads}
        train_op = [grads_var[var.op.name].assign(grad) for grad, var in grads]

        # more debug info to show in tensorboard
        if FLAGS.more_info:
            # Show the histogram of grad
            for grad, var in grads:
                tf.summary.histogram("grad/" + var.op.name, grad)
                # do some statistic
                mean, variance = tf.nn.moments(grad, axes=-1)
                tf.summary.scalar("statistic-mean/{}".format(var.op.name), mean)
                tf.summary.scalar("statistic-variance/{}".format(var.op.name), variance)

                R_ku = tf.reduce_mean(tf.pow((grad - mean), 4) / tf.pow(variance, 2))
                tf.summary.scalar("statistic-R_ku/{}".format(var.op.name), R_ku)

                temp = tf.abs(grad) / (tf.reduce_max(tf.abs(grad)) - tf.abs(grad))
                # alpha = 1 - tf.exp(- temp)
                idx = tf.where(temp > 0.5)
                output = tf.gather_nd(grad, idx)
                tf.summary.histogram("alpha/{}".format(var.op.name), output)

            # compress rate
            compressed = tf.Variable(1, trainable=False, collections=["Summary"], name="compressed")
            uncompressed = tf.Variable(1, trainable=False, collections=["Summary"], name="uncompressed")
            tf.summary.scalar('compress_rate', compressed / uncompressed)

        # Build an initialization operation.
        init = tf.global_variables_initializer()

        # Tensorflow op to update parameters from PS
        get_W = df.get_w(graph, "W_global")

        dataloader = Dataloader(data_path=os.path.join(FLAGS.data_dir, 'cifar-10-batches-py'), sub_dataset=3)
        # train_images = dataloader.data_augmentation(dataloader.train_images, mode='train',
        #                                             flip=True, crop=True, crop_shape=(24, 24, 3), whiten=True,
        #                                             brightness=True, contrast=True,
        #                                             noise=False)
        train_images = dataloader.train_images
        train_labels = dataloader.train_labels
        # print("======================2======================")
        # print(train_images[0].reshape(-1))
        # print("======================2======================")

        train_images, train_labels = shuffle(train_images, train_labels)
        timer = Timer()
        controller = SpeedController(FLAGS.batch_size, FLAGS.compression_rate)
        with tf.Session() as sess:
            # Initialize the TF variables
            sess.run([init])
            tf.train.start_queue_runners(sess=sess)

            if FLAGS.more_info:
                summary_writer = tf.summary.FileWriter('logs_worker/{}/{}'.format(model_name, time.strftime("%Y%m%d-%H%M%S",
                                                                                      time.localtime())), sess.graph)
                merged = tf.summary.merge_all()

            iteration = 0
            s = sck.socket(sck.AF_INET, sck.SOCK_STREAM)
            s.connect((FLAGS.ip_PS, FLAGS.port))

            batch_start = 0
            batch_end = FLAGS.batch_size
            iter_times = 0
            while iteration < FLAGS.iter_max:
                # Get the parameters from the PS
                t1 = time.time()
                com.send_msg(s, "", "GET_W", FLAGS.id_worker, 0)
                t2 = time.time()
                cmd, _, command, _, round_time, data = com.recv_msg(s)

                # Change
                if FLAGS.strategy == "MY":
                    if command > 0:
                        # Speed down
                        FLAGS.batch_size, FLAGS.compression_rate = controller.speed_down(timer.get_epsilon())
                    elif command < 0:
                        # Speed up
                        FLAGS.batch_size, FLAGS.compression_rate = controller.speed_up(timer.get_epsilon())

                s.close()

                t3 = time.time()
                iteration, W = com.decode_variables(data)

                # 随机失活
                # if random.random() <= 0.01:
                #     print("【Worker {}】 Oh.... offline".format(FLAGS.id_worker))
                #     time.sleep(30)

                t4 = time.time()
                # Update the parameters
                sess.run(get_W, {key + "_delta:0": value for key, value in W.items()})
                # t5 = time.time()
                # Compute gradients stored in Tensorflow variables
                # inp, log, lab, loss_values, _ = sess.run([inputs, logits, labels, loss, train_op])
                if batch_end + FLAGS.batch_size >= dataloader.n_train:
                    print("=======================Shuffle===========================")
                    train_images, train_labels = shuffle(train_images, train_labels)
                    batch_start = 0
                    batch_end = FLAGS.batch_size
                else:
                    batch_start = batch_end
                    batch_end = batch_end + FLAGS.batch_size

                batch_images = train_images[batch_start: batch_end]
                batch_labels = train_labels[batch_start: batch_end]
                _, loss_values = sess.run([train_op, loss],
                                          feed_dict={inputs: batch_images, labels: batch_labels, batch_size: FLAGS.batch_size})
                t6 = time.time()
                timer.update_calcu(t6 - t4)

                # Encode the update with the local timer (iteration)
                update = com.encode_variables(sess, "W_grad", iteration,
                                              compression=FLAGS.compression_rate, uncompress=FLAGS.uncompress)
                t7 = time.time()

                # Push the update to PS
                s = sck.socket(sck.AF_INET, sck.SOCK_STREAM)
                s.connect((FLAGS.ip_PS, FLAGS.port))

                t8 = time.time()
                com.send_msg(s, update, "PUSH", FLAGS.id_worker, 0, FLAGS.batch_size)
                t9 = time.time()
                # timer.update_commu((t3 - t2) + (t9 - t8))
                # 迭代总时间 减去 计算时间 就是通信时间
                timer.update_commu(round_time - (t8 - t3))

                # if FLAGS.calcu_compress_rate:
                    # uncompress_grad = com.encode_variables(sess, "W_grad", iteration, compression=0.99)
                    # sess.run(rate, feed_dict={"compressed:0": sys.getsizeof(update), "uncompressed:0": sys.getsizeof(uncompress_grad)})
                # print("Loss: {:.3f} Parameter Size: {:.2f} KB".format(
                #     loss_values, sys.getsizeof(update) / 1024))
                # print("Time:\n [Iter] {:.3f} s, [Calcu] {:.3f} s, [Compress] {:.3f} s ".format(
                #     t7 - t3, t6 - t4, t7 - t6
                # ))
                print("【Worker {}】| bs: {}, cr: {} | Loss: {:.3f} Parameter Size: {:.2f} KB | [EMA Calcu] {:.3f} s, [EMA Comm] {:.3f} s, [Compress] {:.3f} s, [EMA epsilon] {:.3f}".format(
                    FLAGS.id_worker,
                    FLAGS.batch_size, FLAGS.compression_rate,
                    loss_values, sys.getsizeof(update) / 1024,
                    timer.calcu_time, timer.comm_time, t7 - t6, timer.get_epsilon()
                ))

                iter_times += 1

                if FLAGS.more_info:
                    uncompress_grad = com.encode_variables(sess, "W_grad", iteration, compression=0.99, uncompress=True)
                    merged_summary = sess.run(merged, feed_dict={"compressed:0": sys.getsizeof(update), "uncompressed:0": sys.getsizeof(uncompress_grad)})
                    summary_writer.add_summary(merged_summary, iteration)

            print("Worker", FLAGS.id_worker, " is closed")

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
import Model_DNN as mdnn
import mnist
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import sys
import time
import numpy as np

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 10,
                            """Number of images to process in a mini-batch.""")

tf.app.flags.DEFINE_float('compression_rate', 0.0000001,
                          """Compression rate of worker updates.""")

tf.app.flags.DEFINE_bool('more_info', False,
                          """If calcu the compression rate.""")
# Worker routine
def worker(D, graph=None):
    """ Build Tensorflow graph and run iterations """

    if graph == None:
        graph = tf.Graph()
    # Build Tensorflow graph which computes gradients of the model with one mini-batch of examples
    with graph.as_default():

        # Get input and labels for learning from D
        inputs, labels = D
        logits = mdnn.CNN_model(inputs, graph)

        # Calculate loss.
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

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

            # compress rate
            compressed = tf.Variable(1, trainable=False, collections=["Summary"], name="compressed")
            uncompressed = tf.Variable(1, trainable=False, collections=["Summary"], name="uncompressed")
            tf.summary.scalar('compress_rate', compressed / uncompressed)

        # Build an initialization operation.
        init = tf.global_variables_initializer()

        # Tensorflow op to update parameters from PS
        get_W = df.get_w(graph, "W_global")

        with tf.Session() as sess:
            # Initialize the TF variables
            sess.run([init])
            tf.train.start_queue_runners(sess=sess)

            if FLAGS.more_info:
                summary_writer = tf.summary.FileWriter('logs_worker/{}'.format(time.strftime("%Y%m%d-%H%M%S",
                                                                                      time.localtime())), sess.graph)
                merged = tf.summary.merge_all()

            iteration = 0
            s = sck.socket(sck.AF_INET, sck.SOCK_STREAM)
            s.connect((FLAGS.ip_PS, FLAGS.port))

            while iteration < FLAGS.iter_max:
                # Get the parameters from the PS
                com.send_msg(s, "", "GET_W")
                cmd, data = com.recv_msg(s)
                iteration, W = com.decode_variables(data)
                s.close()

                # Update the parameters
                sess.run(get_W, {key + "_delta:0": value for key, value in W.items()})

                # Compute gradients stored in Tensorflow variables
                inp, log, lab, loss_values, _ = sess.run([inputs, logits, labels, loss, train_op])

                # Encode the update with the local timer (iteration)
                update = com.encode_variables(sess, "W_grad", iteration, compression=FLAGS.compression_rate)
                # update = com.encode_variables(sess, "W_grad", iteration, compression=0.99)

                # Push the update to PS
                s = sck.socket(sck.AF_INET, sck.SOCK_STREAM)
                s.connect((FLAGS.ip_PS, FLAGS.port))

                com.send_msg(s, update, "PUSH")
                # if FLAGS.calcu_compress_rate:
                    # uncompress_grad = com.encode_variables(sess, "W_grad", iteration, compression=0.99)
                    # sess.run(rate, feed_dict={"compressed:0": sys.getsizeof(update), "uncompressed:0": sys.getsizeof(uncompress_grad)})
                print("Loss: {:.3f}  Parameter Size: {:.2f} KB".format(loss_values, sys.getsizeof(update) / 1024))

                if FLAGS.more_info:
                    uncompress_grad = com.encode_variables(sess, "W_grad", iteration, compression=0.99)
                    merged_summary = sess.run(merged, feed_dict={"compressed:0": sys.getsizeof(update), "uncompressed:0": sys.getsizeof(uncompress_grad)})
                    summary_writer.add_summary(merged_summary, iteration)

            print("Worker", FLAGS.id_worker, " is closed")

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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from Worker_new import worker
from PS_new import PS
from Supervisor_new import Supervisor
import model.cifar.cifar10 as cifar10


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('iter_max', 100000,
                            """Number of iterations to process on PS.""")
tf.app.flags.DEFINE_string('type_node', 'Worker',
                           """Worker|PS|Superisor : define the local computation node""")
tf.app.flags.DEFINE_integer('nb_workers', 100,
                            """Number of workers.""")
tf.app.flags.DEFINE_integer('id_worker', 0,
                            """ID of worker""")
tf.app.flags.DEFINE_string('ip_PS', 'localhost',
                           """The ip adresse of PS""")
tf.app.flags.DEFINE_integer('port', 2223,
                            """The port used in PS""")
tf.app.flags.DEFINE_integer('image_size', 24,
                            """The size of image""")
tf.app.flags.DEFINE_integer('image_depth', 3,
                            """The depth of image""")
tf.app.flags.DEFINE_integer('nb_classes', 10,
                            """ Number of classes""")
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'model/cifar/cifar10_data/',
                           """Path to the CIFAR-10 data directory.""")

tf.app.flags.DEFINE_bool('uncompress', False,
                          """Not compress""")

tf.app.flags.DEFINE_float('compression_rate', 0.01,
                          """Compression rate of worker updates.""")

tf.app.flags.DEFINE_bool('more_info', False,
                          """If calcu the compression rate.""")

tf.app.flags.DEFINE_string('strategy', 'MY',
                           """Server communication strategy""")

if FLAGS.type_node == "Worker":
    with tf.Graph().as_default() as graph:
        # Load training data
        data_range = (1, 2)
        # if FLAGS.nb_workers == 2:
        #     if FLAGS.id_worker == 1:
        #         data_range = (1, 3)
        #     else:
        #         data_range = (3, 6)

        # training_data = cifar10.distorted_inputs(data_range=data_range)
        # with tf.Session() as sess:
        #     # Initialize the TF variables
        #     print("============================================")
        #     tf.train.start_queue_runners(sess=sess)
        #
        #     print(tf.reshape(training_data[0][0], [-1]).eval(session=sess))
        # print("============================================")

        # Run model
        worker(graph, "cifar")

if FLAGS.type_node == "Supervisor":
    with tf.Graph().as_default() as graph:
        # Load test data
        test_data = cifar10.inputs(True)
        # Test model
        Supervisor(test_data, graph, "cifar")

if FLAGS.type_node == "PS":
    # Update model loop
    PS()

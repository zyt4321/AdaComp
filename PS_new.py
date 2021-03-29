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
import socket as sck
import numpy as np
import Communication as com
import pickle as pck
import os
from datetime import datetime
import traceback
import time
import sys
import threading

import model.cifar.cifar10 as cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('learning_rate', -0.1,
                          """ Learning rate of the gradient descent.""")
tf.app.flags.DEFINE_string('restore_from', 'parameter.pck',
                          """ Restore parameter from saved file""")

# ADACOMP - compute staleness
def count_pred(history, key, loc):
    if history == []:
        st = [0 for i in loc]
    else:
        st = np.add.reduce([[1 if i in h[key] else 0 for i in loc] for h in history], axis=0)
    return st

def learning_rate_decay(iteration, decay_rate=1, decay_every=500):
    decay_num = iteration // decay_every
    return FLAGS.learning_rate * (decay_rate ** decay_num)

class Recorder():
    def __init__(self):
        self.worker_ema_time = {}
        self.worker_last_timestamp = {}
        self.worker_last_version = {}
        self.worker_upload_times = {}
        self.worker_last_command = {}
        self.mu = 0.9
        self.cmin = 0.5
        self.cmax = 1.5

    def arrive(self, worker_id):
        '''
            Worker arrived at the server, update its EMA time according its last timestamp
        :param worker_id:
        :return:
        '''
        if worker_id not in self.worker_upload_times:
            self.worker_upload_times[worker_id] = 1

        # 某个worker梯度到达之前，一定通过get操作获取了参数，因此self.worker_last_timestamp一定存在
        last_round_takes = time.time() - self.worker_last_timestamp[worker_id]
        self.__update_worker_ema_time(worker_id, last_round_takes)
        self.worker_upload_times[worker_id] += 1

    def leave(self, worker_id, iter):
        '''
            Start send grad to workers, record the timestamp and the version (number of iter)
            Also check the change of the min(worker_last_version), it will help to keep the correct number of history
        :param worker_id:
        :return:
        '''
        self.worker_last_timestamp[worker_id] = time.time()

        last_min = 0 if len(self.worker_last_version) == 0 else min(self.worker_last_version.values())
        self.worker_last_version[worker_id] = iter
        now_min = 0 if len(self.worker_last_version) == 0 else min(self.worker_last_version.values())
        return now_min - last_min

    def update_worker_last_version(self, worker_id, iter):
        self.worker_last_version[worker_id] = iter

    def __update_worker_ema_time(self, worker_id, t):
        if worker_id not in self.worker_ema_time:
            self.worker_ema_time[worker_id] = t
        else:
            self.worker_ema_time[worker_id] = self.mu * self.worker_ema_time[worker_id] + (1 - self.mu) * t

    def get_oldest_version(self):
        return min(self.worker_last_version.values())

    def get_worker_last_command(self, worker_id):
        return self.worker_last_command.get(worker_id, 0)

    def delay_theta(self, worker_id):
        '''
            Get the delay theta
        :param worker_id:
        :return:
        '''
        all_ema = list(self.worker_ema_time.values())
        median = np.median(all_ema)
        return self.worker_ema_time[worker_id] / median

    def get_command(self, worker_id):
        '''
            Get command according to theta:
        :param theta:
        :return:
            0: cmin <= theta <= cmax;
            -1: theta > cmax;
            float: theta < cmin (means the times to wait)
        '''
        all_ema = list(self.worker_ema_time.values())
        median = np.median(all_ema)
        theta = self.worker_ema_time[worker_id] / median

        command = 0
        if theta > self.cmax:
            command = -1
        if theta < self.cmin:
            command = (self.cmin - theta) * median

        self.worker_last_command[worker_id] = command
        return command

lock = threading.Lock()
recoder = Recorder()
history = []
iteration = 0

# Parameter Server routine
def PS():
    global recoder, iteration, history
    with tf.Graph().as_default() as graph:
        # Get input and labels for learning from D
        inputs = tf.placeholder(tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, FLAGS.image_depth])
        labels = tf.placeholder(tf.float32, shape=[None, FLAGS.nb_classes])

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar10.inference(inputs)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
        init_glob = tf.variables_initializer(tf.get_collection("W_global"))

        update_op = df.apply_sparse_update(graph, "W_global")
        set_W = df.set_w(graph, "W_global")

        with tf.Session(graph=graph) as sess:

            # Initialize the Deep Neural Network
            sess.run([init, init_glob])

            # Configure socket
            tcpsock = sck.socket(sck.AF_INET, sck.SOCK_STREAM)
            tcpsock.setsockopt(sck.SOL_SOCKET, sck.SO_REUSEADDR, 1)

            tcpsock.bind(("", FLAGS.port))

            worker_latest_record = {i: 0 for i in range(1, FLAGS.nb_workers + 1)}

            # If has saved param, restore first
            if os.path.exists(FLAGS.restore_from):
                with open(FLAGS.restore_from, 'rb') as f:
                    parameters, recoder, history = pck.load(f)
                    iteration, W = com.decode_variables(parameters)
                    # iteration, updates = pck.loads(parameters)

                # Update paramters
                # assign = {}
                # for k in updates.keys():
                #     feed_dict[k[:-5]] = updates[k]
                # assign_op = df.set_w(graph, "W_global")
                # Update parameters
                sess.run(set_W, {key + "_assign:0": value for key, value in W.items()})
                # feed_dict = {}
                # for k in updates.keys():
                #     feed_dict[k + "_delta:0"] = updates[k]
                # sess.run(assign_op, feed_dict)

                print("Restored parameter from iteration: ", iteration)

            try:
                while iteration < FLAGS.iter_max + FLAGS.nb_workers - 1:
                    tcpsock.listen(1)
                    (wsocket, (ip, port)) = tcpsock.accept()
                    tr = threading.Thread(target=worker_handler,
                                          args=(wsocket, (ip, port), sess, update_op))
                    tr.start()


            except Exception as e:
                traceback.print_exc()
            finally:
                # Save parameters
                print("Server aborted at iteration: {}".format(iteration))
                parameters = com.encode_variables(sess, "W_global", iteration, compression=1)
                # Save to file
                with open(FLAGS.restore_from, "wb") as f:
                    pck.dump((parameters, recoder, history), f)
                print("Saved parameters to file: ", FLAGS.restore_from)

            print("PS is closed")


def worker_handler(wsocket, worker_addr, sess, update_op):
    global history, recoder, iteration
    worker_ip, worker_port = worker_addr
    # if iteration % 10 == 0:
    #     print("%s: iteration %d" % ((datetime.now(), iteration)))
    print("======={}=========".format(iteration))
    print(recoder.worker_ema_time)
    # print(recoder.worker_last_timestamp)
    # print(recoder.worker_last_version)
    # print(recoder.worker_upload_times)
    # print(len(history))

    while True:
        try:
            cmd, id_worker, _, data = com.recv_msg(wsocket)
        except Exception as e:
            traceback.print_exc()
            break

        if cmd == "GET_W":
            # Encode parameter
            if lock.acquire():
                try:
                    parameters = com.encode_variables(sess, "W_global", iteration, compression=1)
                except Exception as e:
                    traceback.print_exc()
                    lock.release()
                    return

                command = 0
                if id_worker >= 0:
                    min_changed = recoder.leave(id_worker, iteration)
                    history = history[min_changed:]
                    command = recoder.get_worker_last_command(id_worker)

                lock.release()
                com.send_msg(wsocket, parameters, "PARAM", -1, command) # 这里 -1 表示是server
                # 给工作节点发送完参数后，本次链接断开
                break
        elif cmd == "PUSH":
            if lock.acquire():
                try:
                    recoder.arrive(id_worker)
                except:
                    traceback.print_exc()
                finally:
                    lock.release()

            if FLAGS.strategy == "MY":
                command = recoder.get_command(id_worker)
                print("Workerid: {}, Command: {}".format(id_worker, command))
                if command == -1:
                    # 丢弃本次更新
                    continue
                if command > 0:
                    time.sleep(command)

            old_iter, gradients, indices = com.decode_variables(data, is_grad=True)
            delay = iteration - old_iter
            # for each trainable variable of the model
            for k in gradients.keys():
                # Compute staleness for each parameter
                staleness = count_pred(history[-delay:], k, indices[k])
                # gradients[k] = [learning_rate_decay(iteration) * gradients[k][i] / max(1, staleness[i]) for i in
                #                 range(len(gradients[k]))]
                gradients[k] = [learning_rate_decay(iteration) * gradients[k][i] for i in
                                range(len(gradients[k]))]
            # Update parameters
            feed_dict = {}
            for k in gradients.keys():
                feed_dict[k[:-5] + "_delta:0"] = gradients[k]
                feed_dict[k[:-5] + "_delta_indices:0"] = indices[k]

            if lock.acquire():
                try:
                    sess.run(update_op, feed_dict)
                    history.append({key: set(indices) for key, indices in indices.items()})
                    iteration += 1
                except:
                    traceback.print_exc()
                finally:
                    lock.release()

            # record history and delete unnecessary value
            # last_min = recoder.get_oldest_version()
            # last_min = min(worker_latest_record.values())
            # worker_latest_record[id_worker] = iteration
            # now_min = min(worker_latest_record.values())
            # history = history[(now_min - last_min):]
            # Add update to history

            # cmd, _, data = com.recv_msg(wsocket)
            # if cmd == "GET_W":
            #     # Encode parameter
            #     parameters = com.encode_variables(sess, "W_global", iteration, compression=1)
            #     # print("Parameter Size: {:.2f} KB".format(
            #     #     sys.getsizeof(parameters) / 1024))
            #     com.send_msg(wsocket, parameters, "PARAM", -1)
            #     wsocket.close()
            # else:
            #     wsocket.close()

    wsocket.close()

if __name__ == '__main__':
    print(learning_rate_decay(10000))

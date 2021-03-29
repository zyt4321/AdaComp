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

import socket
import struct
import pickle as pck
import heapq
from Compress import *
import numpy as np
import tensorflow as tf
import sys
FLAGS = tf.app.flags.FLAGS

topk_plain = TopkPlainCompressor()
topk = TopkCompressor()
kus = KurtosisCompressor()
adat = AdaTreshCompressor()
onebit = OneBitCompressor()
onebit01 = OneBitCompressor01()
my_comp = MyCompressor()

compressor = my_comp

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    # try:
    #     data = data.decode()
    # except Exception as e:
    #     data = int.from_bytes(data, byteorder='big',signed=False)
    # print(data)
    return data


def send_msg(sock, msg, cmd, id_worker, command):
    # Prefix each message with a 4-byte length (network byte order)
    cmd = cmd[0]
    assert (isinstance(cmd, str) and len(cmd) == 1)

    if type(msg) == type("str"):
        msg = msg.encode()
    msg = struct.pack('>I', len(msg)) + struct.pack('>c', cmd.encode()) + \
          struct.pack('>i', id_worker) + struct.pack('>f', command) + msg
    sock.sendall(msg)


def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        raise Exception("Recv Error")
    raw_cmd = recvall(sock, 1)
    if not raw_cmd:
        raise Exception("Recv Error")
    raw_id_worker = recvall(sock, 4)
    if not raw_id_worker:
        raise Exception("Recv Error")
    raw_command = recvall(sock, 4)
    if not raw_command:
        raise Exception("Recv Error")

    msglen = struct.unpack('>I', raw_msglen)[0]
    cmd = struct.unpack('>c', raw_cmd)[0].decode()
    id_worker = struct.unpack('>i', raw_id_worker)[0]
    command = struct.unpack('>f', raw_command)[0]
    if cmd == 'P':
        cmd = "PUSH"
    elif cmd == 'G':
        cmd = "GET_W"
    # Read the message data
    return cmd, id_worker, command, recvall(sock, msglen)


def encode_variables(sess, collection, iteration, compression=1, uncompress=False):
    updates = {}
    merged = {}
    # Compression of variables
    if uncompress:
        indices_dict = {}
        for var in sess.graph.get_collection_ref(collection):
            value = sess.run(var).flatten()
            updates[var.op.name] = value
            indices_dict[var.op.name] = list(range(len(value)))
        print("Parameter Size: {:.2f} KB".format(sys.getsizeof(pck.dumps(updates, protocol=-1)) / 1024))

        return pck.dumps((iteration, updates, indices_dict), protocol=-1)

    if compression < 1:
        indices_dict = {}

        # # save
        # dicts = {}

        avg_dict = {}
        total_compress_time = 0
        mask_dict, means_0_dict, means_1_dict = {}, {}, {}
        for var in sess.graph.get_collection_ref(collection):
            value = sess.run(var).flatten()
            # # save
            # dicts[var.op.name] = value

            # 1. topk and others basic methods
            # values, indices = fixed_compress_rate_without_residual(value, compression)
            # merged[var.op.name] = topk.compress(value, var.op.name, compression)
            # values, indices = kus.compress(value, var.op.name, compression)
            # values, indices = adat.compress(value, var.op.name, compression)
            # values, indices = compressor.compress(value, var.op.name, compression)
            #
            # updates[var.op.name] = values
            # indices_dict[var.op.name] = indices


            # 2. my compress
            # cluster_res, group_avg, indices = compressor.compress(value, var.op.name, compression)
            t1 = time.time()
            merged[var.op.name] = compressor.compress(value, var.op.name, compression)
            t2 = time.time()
            total_compress_time += t2 - t1

            # updates[var.op.name] = cluster_res
            # avg_dict[var.op.name] = group_avg
            # indices_dict[var.op.name] = indices

            # 3. 1 bit

            # mask0, mean0, mean1 = compressor.compress(value, var.op.name)

            # mask_dict[var.op.name] = mask0
            # means_0_dict[var.op.name] = mean0
            # means_1_dict[var.op.name] = mean1

        # 1. topk and others basic methods
        # packed = (iteration, updates, indices_dict)
        packed = (iteration, merged)
        # 2. my compress
        # packed = (iteration, updates, avg_dict, indices_dict)
        # 3. 1 bit
        # print("Parameter Size: {:.2f} KB".format(sys.getsizeof(pck.dumps(mask_dict, protocol=-1)) / 1024))
        # packed = (iteration, mask_dict, means_0_dict, means_1_dict)

        # # save
        # np.savez("grads_save/save.npz", **dicts)
        print("Compress: {:.3f}".format(total_compress_time))
        return pck.dumps(packed, protocol=-1)

    else:
        # Full matrix communications
        for var in sess.graph.get_collection_ref(collection):
            updates[var.op.name] = sess.run(var).flatten().tolist()
        return pck.dumps([iteration, updates], protocol=-1)


def decode_variables(message, is_grad=False):
    if not is_grad or FLAGS.uncompress:
        return pck.loads(message)

    packed = pck.loads(message)
    after_decompress = compressor.decompress(*packed)
    return after_decompress

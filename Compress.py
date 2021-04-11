#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @Date: 2021/3/1 上午10:44
# @Filename: Compress
# @Author：zyt
import numpy as np
import tensorflow as tf
import exSTING
import heapq
import bitarray
import math
import sys
import pickle as pck
import time

# Float class with inverse ordering
class backwards(float):
    def __lt__(self, other):
        return float.__lt__(abs(self), abs(other))

    def __le__(self, other):
        return float.__le__(abs(self), abs(other))

    def __gt__(self, other):
        return float.__gt__(abs(self), abs(other))

    def __ge__(self, other):
        return float.__ge__(abs(self), abs(other))

class OneBitCompressor():
    def __init__(self):
        self.err = {}

    def compress(self, tensor, name, compression_rate):
        if name in self.err:
            tensor = tensor + self.err[name]
        numel = tensor.size

        mask0 = tensor < 0
        sum0 = np.sum(tensor[mask0])
        num0 = np.sum(mask0)
        mean0 = sum0 / num0 if num0 > 0 else sum0

        mask1 = ~mask0
        sum1 = np.sum(tensor[mask1])
        num1 = numel - num0
        mean1 = sum1 / num1 if num1 > 0 else sum1

        tensor_decompressed = mask0 * mean0 + ~mask0 * mean1
        temp = tensor - tensor_decompressed
        self.err[name] = temp
        # bit_array = bitarray.bitarray()
        # bit_array.extend(mask0.tolist())

        bit_array = bitarray.bitarray()
        for i in range(len(tensor)):
            if tensor[i] < 0:
                bit_array.append(1)
            else:
                bit_array.append(0)
        # print(bit_array)
        # print(bit_array_1)
        return (bit_array, mean0, mean1)
        # return (mask0, mean0, mean1)

    def decompress(self, *args):
        # mask0, mean0, mean1 = tensor_compressed
        # iteration, tensor_compressed_dict, mean0_dict, mean1_dict = args
        iteration, packed = args

        # tensor_compressed_dict, mean0_dict, mean1_dict = packed
        updates = {}
        indices_dict = {}
        # for name in tensor_compressed_dict:
        for name in packed:
            bit_array, mean0, mean1 = packed[name]
            # mask0, mean0, mean1 = packed[name]
            # mask0 = tensor_compressed_dict[name]
            mask0 = bit_array.tolist()
            mask0 = np.array(list(mask0)).astype(bool)
            # mean0 = mean0_dict[name]
            # mean1 = mean1_dict[name]
            tensor_decompressed = mask0 * mean0 + ~mask0 * mean1

            updates[name] = tensor_decompressed.tolist()

            indices_dict[name] = list(range(len(updates[name])))
        return (iteration, updates, indices_dict)


class OneBitCompressor01():
    def __init__(self):
        self.err = {}

    def compress(self, tensor, name):
        if name in self.err:
            tensor = tensor + self.err[name]
        numel = tensor.size

        mask0 = tensor < 0
        num0 = np.sum(mask0)

        mask1 = ~mask0

        # tensor_compressed = mask0.type(torch.uint8), mean0, mean1

        tensor_decompressed = mask0  + ~mask0
        temp = tensor - tensor_decompressed
        self.err[name] = temp

        # return tensor_compressed, shape
        return mask0.tolist()

    def decompress(self, *args):
        # mask0, mean0, mean1 = tensor_compressed
        iteration, tensor_compressed_dict = args

        updates = {}
        indices_dict = {}
        for name in tensor_compressed_dict:
            mask0 = tensor_compressed_dict[name]
            mask0 = np.array(mask0).astype(bool)

            tensor_decompressed = mask0 + ~mask0

            updates[name] = tensor_decompressed.tolist()

            indices_dict[name] = list(range(len(updates[name])))
        return (iteration, updates, indices_dict)


class TopkPlainCompressor():
    def __init__(self):
        pass
    def compress(self, tensor, name, compression_rate):
        nb_samples = int(max(1, len(tensor) * compression_rate))
        heapqueue = []
        for i in range(nb_samples):
            heapq.heappush(heapqueue, (backwards(tensor[i]), i))
        for i in range(nb_samples, len(tensor)):
            heapq.heappushpop(heapqueue, (backwards(tensor[i]), i))

        values, indices = [], []
        # for j in range(nb_samples,len(value)):
        for j in range(nb_samples):
            v, ind = heapq.heappop(heapqueue)
            values.append(float(v))  # convert backwards to float
            indices.append(ind)
        return values, indices

    def decompress(self, *args):
        return args

def topk_val_indices(tensor, nb_samples):
    # heapqueue = []
    # for i in range(nb_samples):
    #     heapq.heappush(heapqueue, (tensor[i], i))
    # for i in range(nb_samples, len(tensor)):
    #     if tensor[i] < heapqueue[0][0]:
    #         continue
    #     heapq.heappushpop(heapqueue, (tensor[i], i))
    #
    # values, indices = [], []
    # # for j in range(nb_samples,len(value)):
    # for j in range(nb_samples):
    #     v, ind = heapq.heappop(heapqueue)
    #     # values.append(v)  # convert backwards to float
    #     indices.append(ind)
    # return np.array(indices)
    #
    # if k == 0:
    #     return list()

    hp = [(tensor[i], i) for i in range(nb_samples)]
    heapq.heapify(hp)
    for i in range(nb_samples, len(tensor)):
        if hp[0][0] < tensor[i]:
            heapq.heappop(hp)
            heapq.heappush(hp, (tensor[i], i))
    ans = [x[1] for x in hp]
    return np.array(ans)


class TopkCompressor():
    def __init__(self):
        self.err = {}
    def compress(self, tensor, name, rate):
        # 补上残差
        if name in self.err:
            tensor = tensor + self.err[name]
        total_size = len(tensor)
        topk_size = max(int(total_size * rate), 1)

        # indices = np.argpartition(np.abs(tensor), total_size - topk_size)
        # topk_indices = indices[-topk_size:]
        # topk_value = tensor[topk_indices]

        topk_indices = topk_val_indices(np.abs(tensor), topk_size)
        topk_value = tensor[topk_indices]

        # assert sorted(topk_value) == sorted(topk_value_1)
        # assert sorted(topk_indices) == sorted(topk_indices_1)

        # 没有传输的梯度，进入残差（即传输了的梯度，残差清零）
        for v, i in zip(topk_value, topk_indices):
            tensor[i] = 0
        self.err[name] = tensor

        bit_num = math.ceil(math.log(total_size, 2))
        # bit_arr = [bitarray.bitarray(bin(i)[2:].zfill(bit_num)) for i in topk_indices.tolist()]
        bit_total = bitarray.bitarray()
        for i in topk_indices.tolist():
            bit_total.extend(bin(i)[2:].zfill(bit_num))
        # return topk_value.tolist(), topk_indices.tolist()
        # print(total_size, bit_num, len(topk_indices))

        # print("1Parameter Size: {:.2f} B".format(sys.getsizeof(pck.dumps(topk_indices.tolist(), protocol=-1))))
        # print("1Parameter Size: {:.2f} B".format(sys.getsizeof(pck.dumps([int(i) for i in topk_indices.tolist()], protocol=-1))))
        # print("2Parameter Size: {:.2f} B".format(sys.getsizeof(pck.dumps(bit_total, protocol=-1))))
        # print("3Parameter Size: {:.2f} B".format(sys.getsizeof(pck.dumps(bit_total.tobytes(), protocol=-1))))
        # print("4Parameter Size: {:.2f} B".format(sys.getsizeof(pck.dumps(tensor, protocol=-1))))
        # print("5Parameter Size: {:.2f} B".format(sys.getsizeof(pck.dumps(topk_value, protocol=-1))))

        return (topk_value, bit_num, bit_total)

    def decompress(self, *args):
        iteration, packed_dict = args
        updates_dict = {}
        indices_dict = {}
        for name in packed_dict:
            updates, bit_num, bit_total = packed_dict[name]

            bit_arr = []
            bit_str = bit_total.to01()
            for i in range(0, len(bit_str), bit_num):
                bit_arr.append(int(bit_str[i:i+bit_num], 2))
            updates_dict[name] = updates
            indices_dict[name] = bit_arr

        return (iteration, updates_dict, indices_dict)


class KurtosisCompressor():
    def __init__(self):
        self.err = {}
    def compress(self, tensor, name, rate_base):
        # 计算Kurtosis
        mean = np.mean(tensor)
        variance = np.var(tensor)
        R_ku = np.mean((tensor - mean) ** 4 / (variance ** 2))
        rate = rate_base * (10 ** (3 - R_ku))

        # 补上残差
        if name in self.err:
            tensor = tensor + self.err[name]
        total_size = len(tensor)
        topk_size = max(int(total_size * rate), 1)
        indices = np.argpartition(np.abs(tensor), total_size - topk_size)
        topk_indices = indices[-topk_size:]
        topk_value = tensor[topk_indices]

        # 没有传输的梯度，进入残差（即传输了的梯度，残差清零）
        for v, i in zip(topk_value, topk_indices):
            tensor[i] = 0
        self.err[name] = tensor

        return topk_value.tolist(), topk_indices.tolist()

    def decompress(self, *args):
        return args


class AdaTreshCompressor():
    def __init__(self):
        self.r = {}
    def compress(self, g, name, rate_base):
        # 补上残差
        if name in self.r:
            s = g + self.r[name]
        else:
            s = g
        temp = np.abs(g) / (np.max(np.abs(s)) - np.abs(s))
        alpha = 1 - np.exp(- temp)
        indices = np.argwhere(alpha > 0.9)[:, 0]
        values = g[indices]
        print(len(g), len(indices), np.max(np.abs(s)))

        # 没有传输的梯度，进入残差（即传输了的梯度，残差清零）
        self.r[name] = g.copy()
        for v, i in zip(values, indices):
            self.r[name][i] = 0

        return values.tolist(), indices.tolist()

    def decompress(self, *args):
        return args


class MyCompressor():
    def __init__(self):
        self.err = {}
    def compress(self, tensor, name, rate):
        # 补上残差
        if name in self.err:
            tensor = tensor + self.err[name]

        # 取topk个值（待改进）
        total_size = len(tensor)
        topk_size = max(int(total_size * rate), 1)
        indices = np.argpartition(np.abs(tensor), total_size - topk_size)
        # topk_indices = indices[-topk_size:]
        # topk_value = tensor[topk_indices]

        topk_indices = topk_val_indices(np.abs(tensor), topk_size)
        topk_value = tensor[topk_indices]

        # 将topk value进行聚类
        cluster_res, group_avg = exSTING.agglo(topk_value, 32)
        recover = [group_avg[i] for i in cluster_res]
        residual = topk_value - recover
        # 没有传输的梯度，完全计入残差；传输了的梯度，残差设为（原值-均值）
        self.err[name] = tensor.copy()
        for v, i in zip(residual, topk_indices):
            self.err[name][i] = v
        # 全局index编码
        bit_num = math.ceil(max(1, math.log(total_size, 2)))
        bit_total = bitarray.bitarray()
        for i in topk_indices.tolist():
            bit_total.extend(bin(i)[2:].zfill(bit_num))

        # cluster index编码
        avg_num = math.ceil(max(1, math.log(len(group_avg), 2)))
        bit_cluster = bitarray.bitarray()
        for i in cluster_res:
            bit_cluster.extend(bin(i)[2:].zfill(avg_num))
        # print(len(group_avg), topk_size)
        # print("4Parameter Size: {:.2f} B".format(sys.getsizeof(pck.dumps(bit_cluster, protocol=-1))))
        # print("4Parameter Size: {:.2f} B".format(sys.getsizeof(pck.dumps(bit_cluster.tobytes(), protocol=-1))))
        # print("5Parameter Size: {:.2f} B".format(sys.getsizeof(pck.dumps(group_avg, protocol=-1))))
        # print("5Parameter Size: {:.2f} B".format(sys.getsizeof(pck.dumps(np.array(group_avg), protocol=-1))))


        # return (cluster_res, group_avg, topk_indices.tolist())
        return (bit_cluster, group_avg, bit_num, bit_total)

    def decompress_old(self, *args):
        iteration, packed_dict = args

        # iteration, cluster_res_dict, group_avg_dict, topk_indices = args
        # iteration, bit_cluster_dict, group_avg_dict, bit_num_dict, bit_total_dict = args
        # 对每个tensor，还原其传输过来的值
        updates = {}
        topk_indices_dict = {}
        for name in packed_dict:
            cluster_res, group_avg, topk_indices = packed_dict[name]
            recover = [group_avg[i] for i in cluster_res]

            updates[name] = recover
            topk_indices_dict[name] = topk_indices
        return (iteration, updates, topk_indices_dict)

    def decompress(self, *args):
        iteration, packed_dict = args

        # iteration, cluster_res_dict, group_avg_dict, topk_indices = args
        # iteration, bit_cluster_dict, group_avg_dict, bit_num_dict, bit_total_dict = args
        # 对每个tensor，还原其传输过来的值
        updates = {}
        topk_indices = {}
        for name in packed_dict:
            bit_cluster, group_avg, bit_num, bit_total = packed_dict[name]

            index_arr_total = []
            bit_str = bit_total.to01()
            for i in range(0, len(bit_str), bit_num):
                index_arr_total.append(int(bit_str[i:i + bit_num], 2))

            avg_num = math.ceil(max(1, math.log(len(group_avg), 2)))
            index_arr_cluster = []
            bit_str = bit_cluster.to01()

            for i in range(0, len(bit_str), avg_num):
                index_arr_cluster.append(int(bit_str[i:i + avg_num], 2))

            recover = [group_avg[i] for i in index_arr_cluster]

            updates[name] = recover
            topk_indices[name] = index_arr_total
        return (iteration, updates, topk_indices)


if __name__ == '__main__':
    # topk = TopkCompressor()
    # val, index = topk.compress(np.array([-1, -2, -0.13, 0, 1, -6, 5, -3, 3, 0.1]), "", 0.1)
    # print(val)
    # print(index)
    #
    # tensor = np.array([0]*40 + [0.5]*35 + [1]*25 + [1.5]*13 + [2]*5 + [-0.5]*35 + [-1]*25 + [-1.5]*13 + [-2]*5)
    # mean = np.mean(tensor)
    # variance = np.var(tensor)
    # R_ku = np.mean((tensor - mean) ** 4 / (variance ** 2))
    # print(R_ku)

    # adat = AdaTreshCompressor()
    # tensor = np.random.random((20))
    # print(tensor)
    # for i in range(2):
    #     val, ind = adat.compress(tensor, "1", 0.01)

    my = MyCompressor()
    tensor = np.random.random((200)) - 0.5
    c, o = my.compress(tensor, "1", 0.001)
    print(c, o)

    tensor2 = np.random.random((200)) - 0.5
    c2, o2 = my.compress(tensor2, "2", 0.001)
    print(c2, o2)

    res = my.decompress(1, {"1":c, "2": c2})
    res_o = my.decompress_old(1, {"1":o, "2": o2})
    print(res)
    print(res_o)
#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @Date: 2021/3/9 下午4:13
# @Filename: exSTING
# @Author：zyt
import numpy as np
# from scipy import stats
import time

def distance(bin_1, bin_2):
    d1 = bin_1["variance"]
    n1 = bin_1["num"]
    a1 = bin_1["mean"]
    max1 = bin_1["max"]
    min1 = bin_1["min"]
    d2 = bin_2["variance"]
    n2 = bin_2["num"]
    a2 = bin_2["mean"]
    max2 = bin_2["max"]
    min2 = bin_2["min"]
    d_merge = (n1 * d1 + n2 * d2 + (n1 * n2 * ((a1 - a2) ** 2)) / (n1 + n2)) / (n1 + n2)

    # return (d_merge - d1 - d2) * max(abs(max1), abs(max2), abs(min1), abs(min2))
    # return (d_merge - d1 - d2) * (abs(max1) + abs(max2) + abs(min1) + abs(min2))
    # return (d_merge) * max(abs(a1), abs(a2))
    return (d_merge - d1 - d2) * max(abs(max1), abs(max2), abs(min1), abs(min2))

def merge(merge_info_1, merge_info_2):
    ori_idx1 = merge_info_1["ori_idx"]
    idx1 = merge_info_1["idx"]
    d1 = merge_info_1["variance"]
    n1 = merge_info_1["num"]
    a1 = merge_info_1["mean"]
    ori_idx2 = merge_info_2["ori_idx"]
    idx2 = merge_info_2["idx"]
    d2 = merge_info_2["variance"]
    n2 = merge_info_2["num"]
    a2 = merge_info_2["mean"]
    new_bin = {}
    new_bin["ori_idx"] = ori_idx1 + ori_idx2
    new_bin["idx"] = idx1 + idx2
    new_bin["num"] = n1 + n2
    new_bin["mean"] = (n1 * a1 + n2 * a2) / (n1 + n2)
    new_bin["variance"] = (n1 * d1 + n2 * d2 + (n1 * n2 * ((a1 - a2) ** 2)) / (n1 + n2)) / (n1 + n2)
    new_bin["min"] = min(merge_info_1["min"], merge_info_2["min"])
    new_bin["max"] = max(merge_info_1["max"], merge_info_2["max"])
    return  new_bin

def agglo(x2, cluster=2):
    # x_mean = np.mean(x2)
    # x_std = np.std(x2)
    # t1 = time.time()
    x_max = np.max(x2)
    x_min = np.min(x2)
    # print("mean:", x_mean)
    # print("std:", x_std)
    # print("max:", x_max)
    # print("min:", x_min)
    interval = (x_max - x_min) / 1000 if x_min != x_max else 1

    # BIN = int(x_max // interval + abs(x_min) // interval) + 2
    BIN = 1001
    # print("Num of bin: ", BIN)

    bin_list = [[] for _ in range(BIN)]
    for i, val in enumerate(x2):
        bin_index = int((val - x_min) // interval)
        bin_list[bin_index].append((i, val))
    # t2 = time.time()
    # for i in bin_list:
    #     print(len(i))

    # 计算每个bin的统计指标
    bin_statistic = []
    cnt = 0
    for idx, bin in enumerate(bin_list):
        bin_val = [i[1] for i in bin]
        if len(bin_val) == 0:
            bin_statistic.append({})
            continue
        s = {}
        # desc = stats.describe(bin_val, ddof=0)
        s["ori_idx"] = [idx]
        s["idx"] = [cnt]
        # s["num"] = desc.nobs
        # s["mean"] = desc.mean
        # s["variance"] = desc.variance
        # s["min"] = desc.minmax[0]
        # s["max"] = desc.minmax[1]
        s["num"] = len(bin_val)
        s["mean"] = np.mean(bin_val)
        s["variance"] = np.var(bin_val)
        s["min"] = np.min(bin_val)
        s["max"] = np.max(bin_val)
        bin_statistic.append(s)
        cnt += 1
    # t3 = time.time()
    # 去除所有num == 0的bin
    while [] in bin_list:
        bin_list.remove([])
    while {} in bin_statistic:
        bin_statistic.remove({})
    # t4 = time.time()
    # print("Num of bin: ", len(bin_list))

    # 计算两两距离
    distance_list = []
    for idx in range(len(bin_list) - 1):
        curr = bin_statistic[idx]
        next = bin_statistic[idx + 1]

        dist = distance(curr, next)
        distance_list.append(dist)
    # t5 = time.time()
    # 选择距离最近的两个bin，进行合并
    while len(bin_statistic) > cluster:
        min_dis_index = distance_list.index(min(distance_list))
        bin_merge_1 = bin_statistic[min_dis_index]
        bin_merge_2 = bin_statistic[min_dis_index + 1]
        merged = merge(bin_merge_1, bin_merge_2)
        # 删除已合并bin，注意删除顺序
        bin_statistic.pop(min_dis_index + 1)
        bin_statistic.pop(min_dis_index)
        distance_list.pop(min_dis_index)
        # 添加合并的bin
        bin_statistic.insert(min_dis_index, merged)
        # 计算新的距离
        for idx in [min_dis_index - 1, min_dis_index]:
            if idx < 0 or idx + 1 >= len(bin_statistic):
                continue
            curr = bin_statistic[idx]
            next = bin_statistic[idx + 1]

            dist = distance(curr, next)
            distance_list[idx] = dist

        # temp = []
        # for i in bin_statistic:
        #     temp.append(i["num"])
        # print(temp)
    # t6 = time.time()
    # bin的分组情况, 每组的数据数量, 每组均值
    groups = []
    groups_num = []
    groups_avg = []
    for i in bin_statistic:
        groups.append(i["idx"])
        groups_num.append(i["num"])
        groups_avg.append(i["mean"])
    # print(groups_num)

    # 每组的范围
    # groups_range = []
    # for i in bin_statistic:
    #     l = min(i["ori_idx"])
    #     r = max(i["ori_idx"])
    #     left = l * interval + x_min
    #     right = (r + 1) * interval + x_min
    #     groups_range.append((left, right))

    # 对每个数，找到其所在分组
    # 遍历每个group下的所有bin
    # 每个bin将其包含的数的所在位置，设为group_id
    cluster_res = [-1 for i in range(len(x2))]
    for gid, group in enumerate(groups):
        for bin_id in group:
            for val_idx, val in bin_list[bin_id]:
                cluster_res[val_idx] = gid
    # t7 = time.time()
    # print("T2-T1: {:.3f}, T3-T2: {:.3f}, T4-T3: {:.3f}, T5-T4: {:.3f}, T6-T5: {:.3f}, T7-T6: {:.3f}".format(
    #     t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 -t6
    # ))
    return cluster_res, groups_avg


    # plot 相关
    # hist, bins = np.histogram(x2, bins=100,  density=True)
    # start = 0.0
    # stop = 1.0
    # number_of_lines= cluster + 1
    # cm_subsection = np.linspace(start, stop, number_of_lines)
    # colors = [ cm.jet(x) for x in cm_subsection ]
    #
    # color_list = []
    # for i in bins:
    #     is_in_range = False
    #     for r_idx, r in enumerate(groups_range):
    #         if r[0] <= i <= r[1]:
    #             color_list.append(colors[r_idx + 1])
    #             is_in_range = True
    #             break
    #     if not is_in_range:
    #         color_list.append(colors[0])
    #
    #
    # plt.bar(bins[:-1], hist, color=color_list, width=bins[1] - bins[0], edgecolor=None)
    # X = x2.reshape(-1, 1)
    # labels = cluster_res
    # # plt.scatter(X, [-1] * len(X), c=labels, s=2, cmap='coolwarm')
    # plt.show()

if __name__ == '__main__':
    # x = [1,2,3,4,5,6,7,8,9,0]
    x = [0.69959329,0.48048923,0.32015305,0.76785673,0.9278056, 0.54826804,
 0.52022435,0.08020569, 0.57799004,0.25622789,0.38879256,0.84993924,
 0.13713626,0.49336196, 0.95963372,0.51279234,0.06915951,0.2950675,
 0.37195582,0.96056861]
 #    x = [3.01, 3.02]
 #    x = [0, 0]
    cluster_res, group_avg = agglo(x, 32)

    recover = [group_avg[i] for i in cluster_res]

    print(cluster_res)
    print(group_avg)
    print(recover)

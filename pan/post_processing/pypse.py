# -*- coding: utf-8 -*-
# @Time    : 2019/9/9 17:23
# @Author  : zhoujun

import numpy as np
from queue import Queue


def get_dis(sv1, sv2):
    return np.linalg.norm(sv1 - sv2)


def pse_py(text, similarity_vectors, label, label_values, dis_threshold=0.8):
    pred = np.zeros(text.shape)
    queue = Queue(maxsize=0)
    # print(666666666666, label)
    points = np.array(np.where(label > 0)).transpose((1, 0))
    # print(888888888888, points)
    for point_idx in range(points.shape[0]):
        y, x = points[point_idx, 0], points[point_idx, 1]
        label_value = label[y, x]
        queue.put((y, x, label_value))
        pred[y, x] = label_value
    # 计算kernel的值
    d = {}
    # print(77777777777777, label_values)
    for i in label_values:
        kernel_idx = label == i
        # print(8888888888, similarity_vectors[kernel_idx])
        kernel_similarity_vector = similarity_vectors[kernel_idx].mean(0)  # 4
        d[i] = kernel_similarity_vector

    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    kernal = text.copy()
    # (y, x, label_value) = queue.get()
    # cur_kernel_sv = d[label_value]
    # for j in range(4):
    #     tmpx = x + dx[j]
    #     tmpy = y + dy[j]
    #     print(kernal)
    #     if tmpx < 0 or tmpy >= kernal.shape[0] or tmpy < 0 or tmpx >= kernal.shape[1]:
    #         continue
    #     if kernal[tmpy, tmpx] == 0 or pred[tmpy, tmpx] > 0:
    #         continue
    #     # print(111111111111, similarity_vectors[tmpy, tmpx])
    #     # print(2222222222222, cur_kernel_sv)
    #     # print(333333333333333, np.linalg.norm(similarity_vectors[tmpy, tmpx] - cur_kernel_sv))
    #     if np.linalg.norm(similarity_vectors[tmpy, tmpx] - cur_kernel_sv) >= dis_threshold:
    #         continue
    #     queue.put((tmpy, tmpx, label_value))
    #     pred[tmpy, tmpx] = label_value
    # print('ssssssssssssssssssssssssssss', y, x, label_value, cur_kernel_sv, d)
    while not queue.empty():
        (y, x, label_value) = queue.get()
        # print('ssssssssssssssssssssssssssss', y, x, label)
        cur_kernel_sv = d[label_value]
        for j in range(4):
            tmpx = x + dx[j]
            tmpy = y + dy[j]
            if tmpx < 0 or tmpy >= kernal.shape[0] or tmpy < 0 or tmpx >= kernal.shape[1]:
                continue
            if kernal[tmpy, tmpx] == 0 or pred[tmpy, tmpx] > 0:
                continue
            # print(111111111111, similarity_vectors[tmpy, tmpx])
            # print(2222222222222, cur_kernel_sv)
            # print(333333333333333, np.linalg.norm(similarity_vectors[tmpy, tmpx] - cur_kernel_sv))
            if np.linalg.norm(similarity_vectors[tmpy, tmpx] - cur_kernel_sv) >= dis_threshold:
                continue
            queue.put((tmpy, tmpx, label_value))
            pred[tmpy, tmpx] = label_value
    return pred

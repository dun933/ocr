# -*- coding: utf-8 -*-
# @Time    : 2019/8/24 12:06
# @Author  : zhoujun

import torch
from torchvision import transforms
import os
import cv2
import math
from math import *
import numpy as np
import time
from PIL import Image

from pan.models import get_model
from pan.post_processing import decode_np as decode


def rotate(
        img,  # 图片
        pt1, pt2, pt3, pt4
):
    # print(pt1, pt2, pt3, pt4)
    withRect = math.sqrt((pt4[0] - pt1[0]) ** 2 + (pt4[1] - pt1[1]) ** 2)  # 矩形框的宽度
    heightRect = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    # print(withRect, heightRect)
    angle = acos((pt4[0] - pt1[0]) / withRect) * (180 / math.pi)  # 矩形框旋转角度
    # print(angle)

    if pt4[1] > pt1[1]:
        angle = angle
        # print("顺时针旋转")
    else:
        # print("逆时针旋转")
        angle = -angle

    height = img.shape[0]  # 原始图像高度
    width = img.shape[1]  # 原始图像宽度
    rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)  # 按angle角度旋转图像
    heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
    widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))

    rotateMat[0, 2] += (widthNew - width) / 2
    rotateMat[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, rotateMat, (widthNew, heightNew + 50), borderValue=(255, 255, 255))
    # cv2.imshow('rotateImg2',  imgRotation)
    # cv2.waitKey(0)

    # 旋转后图像的四点坐标
    [[pt1[0]], [pt1[1]]] = np.dot(rotateMat, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(rotateMat, np.array([[pt3[0]], [pt3[1]], [1]]))
    [[pt2[0]], [pt2[1]]] = np.dot(rotateMat, np.array([[pt2[0]], [pt2[1]], [1]]))
    [[pt4[0]], [pt4[1]]] = np.dot(rotateMat, np.array([[pt4[0]], [pt4[1]], [1]]))

    # 处理反转的情况
    if pt2[1] > pt4[1]:
        pt2[1], pt4[1] = pt4[1], pt2[1]
    if pt1[0] > pt3[0]:
        pt1[0], pt3[0] = pt3[0], pt1[0]

    imgOut = imgRotation[int(pt2[1]):int(pt4[1]), int(pt1[0]):int(pt3[0])]
    # cv2.imshow("imgOut", imgOut)  # 裁减得到的旋转矩形框
    # cv2.waitKey(0)
    return imgOut


from shapely.geometry import Polygon


def count_area(a, b, index_a, index_b):
    a = Polygon([(a[0], a[1]), (a[2], a[3]), (a[4], a[5]), (a[6], a[7])]).convex_hull
    b = Polygon([(b[0], b[1]), (b[2], b[3]), (b[4], b[5]), (b[6], b[7])]).convex_hull
    try:
        cut_area = a.intersection(b).area
    except:
        return 0, 0
    min_area = [b, index_b] if a.area >= b.area else [a, index_a]
    percent_area = cut_area / min_area[0].area
    return percent_area, min_area[1]


class Pytorch_model:
    def __init__(self, model_path, gpu_id=None):
        '''
        初始化pytorch模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param gpu_id: 在哪一块gpu上运行
        '''
        self.gpu_id = gpu_id
        # checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        checkpoint = torch.load(model_path)

        # if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
        self.device = torch.device("cuda:%s" % self.gpu_id)
        # else:
        # self.device = torch.device("cpu")
        print('device:', self.device)

        config = checkpoint['config']
        config['arch']['args']['pretrained'] = False
        self.net = get_model(config)

        self.img_channel = config['data_loader']['args']['dataset']['img_channel']
        self.net.load_state_dict(checkpoint['state_dict'])
        self.net.to(self.device)
        self.net.eval()

    def predict(self, img: str, scale_w, scale_h, ori_img):
        '''
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param img: 图像地址
        :param is_numpy:
        :return:
        '''
        # assert os.path.exists(img), 'file is not exists'
        # img = cv2.imread(img)
        # if self.img_channel == 3:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # h, w = img.shape[:2]
        # short_edge = min(h, w)
        # if short_edge < short_size:
        #     # 保证短边 >= inputsize
        #     scale = short_size / short_edge
        #     if scale > 1:
        #         img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = transforms.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            # torch.cuda.synchronize(self.device)
            start = time.time()
            preds = self.net(tensor)[0]
            # torch.cuda.synchronize(self.device)
            preds, boxes_list = decode(preds, threshold=0.7)
            if boxes_list == '':
                return ''
            # print('boxes_list', boxes_list)
            # scale = (preds.shape[1] / w, preds.shape[0] / h)
            # print(33333333333333, scale)
            # if len(boxes_list):
            #     boxes_list = boxes_list / scale
            t = time.time() - start
        images = []
        boxes_list = boxes_list.astype('int32')
        boxes_list = boxes_list.reshape(-1, 8)
        ori_img = np.array(ori_img)
        img_copy = ori_img.copy()
        for index, i in enumerate(boxes_list):
            i = [int(i[0] * scale_w), int(i[1] * scale_h), int(i[2] * scale_w), int(i[3] * scale_h),
                 int(i[4] * scale_w), int(i[5] * scale_h), int(i[6] * scale_w), int(i[7] * scale_h)]
            i = [j if j > 0 else 0 for j in i]
            if abs(i[1] - i[7]) < abs(i[0] - i[6]):
                i = [i[2], i[3], i[4], i[5], i[0], i[1], i[6], i[7]]
                x0, y0 = min(i[0], i[2], i[4], i[6]), min(i[1], i[3], i[5], i[7])
                new_im = ori_img[min(i[1], i[3], i[5], i[7]):max(i[1], i[3], i[5], i[7]),
                         min(i[0], i[2], i[4], i[6]):max(i[0], i[2], i[4], i[6])]
                nei_i = [ii - x0 if index in [0, 2, 4, 6] else ii - y0 for index, ii in enumerate(i)]
                try:
                    crop_img = rotate(new_im, [nei_i[0], nei_i[1]], [nei_i[4], nei_i[5]], [nei_i[6], nei_i[7]],
                                      [nei_i[2], nei_i[3]])
                except Exception as e:
                    print(1111111111, e)
                    print(i)
                    # cv2.imshow('rotateImg2', new_im)
                    continue
            else:
                i = [i[0], i[1], i[2], i[3], i[6], i[7], i[4], i[5]]
                x0, y0 = min(i[0], i[2], i[4], i[6]), min(i[1], i[3], i[5], i[7])
                new_im = ori_img[min(i[1], i[3], i[5], i[7]):max(i[1], i[3], i[5], i[7]),
                         min(i[0], i[2], i[4], i[6]):max(i[0], i[2], i[4], i[6])]
                nei_i = [ii - x0 if index in [0, 2, 4, 6] else ii - y0 for index, ii in enumerate(i)]
                try:
                    crop_img = rotate(new_im, [nei_i[0], nei_i[1]], [nei_i[4], nei_i[5]], [nei_i[6], nei_i[7]],
                                      [nei_i[2], nei_i[3]])
                except Exception as e:
                    print(22222222222, e)
                    continue
            # cv2.line(img_copy, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), 1)
            # cv2.line(img_copy, (i[0], i[1]), (i[4], i[5]), (0, 255, 0), 1)
            # cv2.line(img_copy, (i[6], i[7]), (i[2], i[3]), (0, 255, 0), 1)
            # cv2.line(img_copy, (i[4], i[5]), (i[6], i[7]), (0, 255, 0), 1)
            # Image.fromarray(crop_img).save('fast/{}.jpg'.format(index))

            images.append([i, crop_img])
        over_index = set()
        for indexi, i in enumerate(images):
            for indexj, j in enumerate(images):
                percent_area, index = count_area(i[0], j[0], indexi, indexj)
                if percent_area >= 0.6 and indexi != indexj:
                    over_index.add(index)
        new_images = []
        for index, i in enumerate(images):
            if index not in over_index:
                new_images.append(i)
        images = new_images
        # Image.fromarray(img_copy).save('1111.jpg')
        return images


model_path = 'pan/checkpoints/model_best.pth'
model = Pytorch_model(model_path, gpu_id=0)


def text_predict(img, scale_w, scale_h, ori_img):
    images = model.predict(img, scale_w, scale_h, ori_img)
    return images

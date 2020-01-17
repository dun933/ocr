# -*- coding: utf-8 -*-
import os
import cv2
from PIL import Image

from crnn.crnn_torch import crnnOcr as crnnOcr


def rec_txt(crop_img):
        H, W = crop_img.shape[:2]
        H2 = 32
        W2 = float(W * H2) / H
        W2 = int(W2)
        size = (W2, H2)
        cv2.resize(crop_img,size,interpolation=cv2.INTER_LINEAR)
        crop_img = Image.fromarray(crop_img)
        text = crnnOcr(crop_img.convert('L'))
        return text


if __name__ == "__main__":
   pass

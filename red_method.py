import cv2
import numpy as np


def seal_eliminate(image):
    hue_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    low_range = np.array([0, 0, 0])
    high_range = np.array([180, 255, 200])
    mask_bg = cv2.inRange(hue_image, low_range, high_range)

    index = mask_bg == 255

    img = np.zeros(image.shape, np.uint8)
    img[:, :] = (255, 255, 255)
    img[index] = image[index]  # (0,0,255)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    low_range_red1 = np.array([0, 93, 96])
    high_range_red1 = np.array([10, 255, 255])
    mask_red1 = cv2.inRange(img, low_range_red1, high_range_red1)
    res_red1 = cv2.inpaint(img, mask_red1, 1, flags=cv2.INPAINT_TELEA)

    low_range_red2 = np.array([156, 93, 96])
    high_range_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(img, low_range_red2, high_range_red2)
    res_red2 = cv2.inpaint(res_red1, mask_red2, 1, flags=cv2.INPAINT_TELEA)

    res = cv2.cvtColor(res_red2, cv2.COLOR_HSV2RGB)
    return res


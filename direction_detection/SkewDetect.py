""" Calculates skew angle """
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time

import cv2
from math import fabs, sin, cos, radians
import numpy as np
import scipy
from PIL import Image
from skimage.transform import hough_line, hough_line_peaks


class SkewDetect:

    piby4 = np.pi / 4

    def __init__(
        self,
        input_file=None,
        batch_path=None,
        output_file=None,
        sigma=3.0,
        display_output=None,
        num_peaks=20,
        plot_hough=None
    ):

        self.sigma = sigma
        self.input_file = input_file
        self.batch_path = batch_path
        self.output_file = output_file
        self.display_output = display_output
        self.num_peaks = num_peaks
        self.plot_hough = plot_hough
        self.angleNet = cv2.dnn.readNetFromTensorflow(os.path.join(os.getcwd(), "models", "Angle-model.pb"),
                                                 os.path.join(os.getcwd(), "models", "Angle-model.pbtxt"))

    def rotate_img(self, image, degree):
        img1 = np.array(image.convert('RGB'))
        height, width = img1.shape[:2]

        # 旋转后的尺寸
        heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))  # 这个公式参考之前内容
        widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

        matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

        matRotation[0, 2] += (widthNew - width) / 2  # 因为旋转之后,坐标系原点是新图像的左上角,所以需要根据原图做转化
        matRotation[1, 2] += (heightNew - height) / 2

        imgRotation = cv2.warpAffine(img1, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
        return Image.fromarray(imgRotation)

    def get_rotated_img(self, origin_img):
        # 定义文本旋转处理类对象
        # origin_img = Image.open(jpg_path)  # 读取图像数据
        res = self.determine_skew(origin_img)
        angle = res['Estimated Angle']

        if (angle >= 0) and (angle <= 90):
            rot_angle = angle - 90
        if (angle >= -45) and (angle < 0):
            rot_angle = angle - 90
        if (angle >= -90) and (angle < -45):
            rot_angle = 90 + angle
        # print(rot_angle)
        # 根据检测出来的旋转角度进行旋转操作
        rotated = self.rotate_img(origin_img, rot_angle)
        angle = self.detect_angle_90(rotated)
        final_img = self.rotate_img(origin_img, rot_angle+angle)
        return final_img

    def get_max_freq_elem(self, arr):
        max_arr = []
        freqs = {}
        for i in arr:
            if i in freqs:
                freqs[i] += 1
            else:
                freqs[i] = 1

        sorted_keys = sorted(freqs, key=freqs.get, reverse=True)
        max_freq = freqs[sorted_keys[0]]

        for k in sorted_keys:
            if freqs[k] == max_freq:
                max_arr.append(k)

        return max_arr

    def compare_sum(self, value):
        if 43 <= value <= 47:
            return True
        else:
            return False

    def calculate_deviation(self, angle):

        angle_in_degrees = np.abs(angle)
        deviation = np.abs(SkewDetect.piby4 - angle_in_degrees)

        return deviation

    def determine_skew(self, origin_img):
        # img = io.imread(img_file, as_gray=True)
        # edges = canny(img, sigma=self.sigma)
        img = origin_img.convert('RGB')
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        B_channel, G_channel, R_channel = cv2.split(img)
        _, img = cv2.threshold(R_channel, 160, 255, cv2.THRESH_BINARY)
        img_h, img_w = img.shape
        img = img[int(img_h*0.2):int(img_h*0.8), int(img_w*0.2):int(img_w*0.8)]
        kernel = np.ones((15, 15), np.uint8)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.erode(img, kernel)
        # img = cv2.dilate(img, kernel)
        # Image.fromarray(img).show()
        edges = cv2.Canny(img, 40, 255)
        Image.fromarray(edges).show()
        edges = edges.astype(np.float32)
        edges /= 255

        h, a, d = hough_line(edges)
        _, ap, _ = hough_line_peaks(h, a, d, num_peaks=self.num_peaks)
        if len(ap) == 0:
            return {"Estimated Angle": 0}

        absolute_deviations = [self.calculate_deviation(k) for k in ap]
        average_deviation = np.mean(np.rad2deg(absolute_deviations))
        ap_deg = [np.rad2deg(x) for x in ap]

        bin_0_45 = []
        bin_45_90 = []
        bin_0_45n = []
        bin_45_90n = []

        for ang in ap_deg:

            deviation_sum = int(90 - ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_45_90.append(ang)
                continue

            deviation_sum = int(ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_0_45.append(ang)
                continue

            deviation_sum = int(-ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_0_45n.append(ang)
                continue

            deviation_sum = int(90 + ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_45_90n.append(ang)

        angles = [bin_0_45, bin_45_90, bin_0_45n, bin_45_90n]
        # print(angles)
        maxi = np.argmax([len(i) for i in angles])

        # print(angles, maxi)
        if len(angles[maxi]) > 0:
            if len(angles[maxi]) >= 5:
                angles = sorted(angles[maxi])
                angles = angles[1:-1]
            elif len(angles[maxi]) <= 2:
                angles = [0]
            else:
                angles = angles[maxi]
            # print(angles)
            ans_arr = self.get_max_freq_elem(angles)
            ans_res = np.mean(ans_arr)

        else:
            try:
                ap_deg_5 = [int(i//5) for i in ap_deg]
                mode = scipy.stats.mode(ap_deg_5)[0][0]*5
                ap_deg = [i for i in ap_deg if abs(i - mode) < 10]
                ans_arr = self.get_max_freq_elem(ap_deg)
                ans_res = np.mean(ans_arr)
                # print(11111111111111111111111111111)
            except:
                ans_res = 0

        data = {
            # "Image File": img_file,
            "Average Deviation from pi/4": average_deviation,
            "Estimated Angle": ans_res,
            "Angle bins": angles}
        return data

    def detect_angle_90(self, img, adjust=True):
        """
        文字方向检测
        """
        img = np.array(img)
        h, w = img.shape[:2]
        ROTATE = [0, 90, 180, 270]
        if adjust:
            thesh = 0.05
            xmin, ymin, xmax, ymax = int(thesh * w), int(thesh * h), w - int(thesh * w), h - int(thesh * h)
            img = img[ymin:ymax, xmin:xmax]  # # 剪切图片边缘

        inputBlob = cv2.dnn.blobFromImage(img,
                                          scalefactor=1.0,
                                          size=(224, 224),
                                          swapRB=True,
                                          mean=[103.939, 116.779, 123.68], crop=False)
        self.angleNet.setInput(inputBlob)
        pred = self.angleNet.forward()
        index = np.argmax(pred, axis=1)[0]
        return ROTATE[index]


skew_detect = SkewDetect()


# main函数
if __name__ == '__main__':
    skew_detect = SkewDetect()
    for name in os.listdir('images'):
        if '2010 股权转让协议-领庆创投-领锐创投-松芝投资-祥禾投资_117.jpg' not in name:
            continue
        st_time = time.time()
        img = skew_detect.get_rotated_img(f'images/{name}')
        img.save(f'rotated_imgs1/{name}')
        print(f'{name} use time:', time.time() - st_time)
        img.show()

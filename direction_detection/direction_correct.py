import os
import tensorflow as tf
import cv2
import numpy as np
import math
from math import fabs, sin, radians, cos
from PIL import Image


pwd = os.getcwd()

# print(os.getcwd())
##vgg文字方向检测模型
AngleModelPb = os.path.join('direction_detection', "models", "Angle-model.pb")
AngleModelPbtxt = os.path.join('direction_detection', "models", "Angle-model.pbtxt")
# print(111111, AngleModelPb)
# 转换为tf模型，以便GPU调用
# from tensorflow.python.platform import gfile

# config = tf.ConfigProto()
# sess = tf.Session()

# with gfile.FastGFile(AngleModelPb, 'rb') as f:
#     graph_def = tf.compat.v1.GraphDef()
#     graph_def.ParseFromString(f.read())
#     sess.graph.as_default()
#     tf.import_graph_def(graph_def, name='')
#     print('load model...')
# inputImg = sess.graph.get_tensor_by_name('input_1:0')
# predictions = sess.graph.get_tensor_by_name('predictions/Softmax:0')
# keep_prob = tf.compat.v1.placeholder(tf.float32)
angleNet = cv2.dnn.readNetFromTensorflow(AngleModelPb,AngleModelPbtxt)


def angle_detect_tf(img, angle, adjust=True):
    """
    文字方向检测
    """
    img = np.array(img)
    h, w = img.shape[:2]
    ROTATE = [0, 90, 180, 270]
    if adjust:
        thesh = 0.05
        xmin, ymin, xmax, ymax = int(thesh * w), int(thesh * h), w - int(thesh * w), h - int(thesh * h)
        img = img[ymin:ymax, xmin:xmax]  ##剪切图片边缘

    inputBlob = cv2.dnn.blobFromImage(img,
                                      scalefactor=1.0,
                                      size=(224, 224),
                                      swapRB=True,
                                      mean=[103.939, 116.779, 123.68], crop=False);
    angleNet.setInput(inputBlob)
    pred = angleNet.forward()
    index = np.argmax(pred, axis=1)[0]
    return -ROTATE[index]


# def angle_detect_tf(img, angle, adjust=True):
#     """
#     文字方向检测
#     """
#     # print('load_img...')
#     img = np.array(img)
#     h, w = img.shape[:2]
#     ROTATE = [0, 90, 180, 270]
#     if adjust:
#         thesh = 0.05
#         xmin, ymin, xmax, ymax = int(thesh * w), int(thesh * h), w - int(thesh * w), h - int(thesh * h)
#         img = img[ymin:ymax, xmin:xmax]  ##剪切图片边缘
#     img = cv2.resize(img, (224, 224))
#     img = np.array(rotate_img(Image.fromarray(img), angle))
#     # Image.fromarray(img).show()
#     img = img[..., ::-1].astype(np.float32)
#
#     img[..., 0] -= 103.939
#     img[..., 1] -= 116.779
#     img[..., 2] -= 123.68
#     # Image.fromarray(img.astype(np.uint8)).show()
#     img = np.array([img])
#     # print('predict_img')
#     out = sess.run(predictions, feed_dict={inputImg: img,
#                                            keep_prob: 0.
#                                            })
#     # print(1111111, out)
#     index = np.argmax(out, axis=1)[0]
#     # print(222222222, index)
#     return -ROTATE[index]


def rotate_img(image, degree):
    degree = -degree
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


def fourier_demo(image1, ft):
    a_img = image1.copy()
    image1.thumbnail((1000, 1000), Image.ANTIALIAS)
    ft_list = ['FT007006003001', 'FT007006005001', 'FT007006007001', 'FT007006008001', 'FT007006009001',
               'FT007006010001', 'FT007006016001', 'FT007006022001', 'FT007008002001']
    if ft not in ft_list:
        # angle_2 = rotate_classification(image1, 0)
        angle_2 = angle_detect_tf(image1, 0)

        rotated = rotate_img(a_img, angle_2)
        return rotated
    else:
        # 1、读取文件，灰度化
        # image1 = Image.open(image_path)
        img = np.array(image1.convert('RGB'))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2、图像延扩
        h, w = img.shape[:2]
        new_h = cv2.getOptimalDFTSize(h)
        new_w = cv2.getOptimalDFTSize(w)
        right = new_w - w
        bottom = new_h - h
        nimg = cv2.copyMakeBorder(gray, 0, bottom, 0, right, borderType=cv2.BORDER_CONSTANT, value=0)

        # 3、执行傅里叶变换，并过得频域图像
        f = np.fft.fft2(nimg)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift))

        # 二值化
        magnitude_uint = magnitude.astype(np.uint8)
        ret, thresh = cv2.threshold(magnitude_uint, 11, 255, cv.THRESH_BINARY)

        # 霍夫直线变换
        lines = cv2.HoughLinesP(thresh, 2, np.pi / 180, 30, minLineLength=40, maxLineGap=100)

        # # 创建一个新图像，标注直线
        # lineimg = np.ones(nimg.shape, dtype=np.uint8)
        # lineimg = lineimg * 255

        # new = np.ones(nimg.shape, dtype=np.uint8)
        # new = new * 255

        max_len = 0
        index = 0
        if len(lines) > 0:
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                dis = (x1 - x2) ** 2 + (y1 - y2) ** 2
                if x2 - x1 != 0 and y2 - y1 != 0:
                    if dis > max_len:
                        max_len = dis
                        index = i
            #         cv.line(new, (x1, y1), (x2, y2), (0, 255, 0), 2)

            x1, y1, x2, y2 = lines[index][0]

            # # show霍夫直線
            # cv.line(lineimg, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Image.fromarray(lineimg).show()

            theta = (x2 - x1) / (y2 - y1)
            angle_1 = math.atan(theta)
            angle_1 = angle_1 * (180 / math.pi) / (w / h)
            if abs(1 / theta) < 0.1:
                angle_1 = 0
        else:
            angle_1 = 0
        # angle_2 = rotate_classification(image1, angle_1)
        angle_2 = angle_detect_tf(image1, angle_1)
        rotated = rotate_img(a_img, angle_1 + angle_2)
    return rotated


img = np.zeros((224, 224, 3), dtype=np.uint8)
img = 255 - img
print(angle_detect_tf(Image.fromarray(img), 0))


if __name__ == '__main__':
    p = r'171212 思华科技营业执照（正本）.pdf'
    # print(11111111)
    # img = Image.open(r'C:\Users\Admin\Desktop\1.png').convert('RGB')
    import fitz
    pdf = fitz.open(p)
    page = pdf[0]
    trans = fitz.Matrix(3, 3).preRotate(0)
    pm = page.getPixmap(matrix=trans, alpha=False)
    img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
    print(angle_detect_tf(img, 0))
    fourier_demo(img, '001').save('11.jpg')

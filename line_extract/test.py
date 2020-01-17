import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.01
# set_session(tf.Sexssion(config=config))
#
# KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'cpu'})))


import warnings
warnings.filterwarnings("ignore")

from line_extract.model_v3 import unet
from line_extract.data import data_preprocess
import numpy as np
import cv2
from PIL import Image, ImageEnhance


def image_normalized(img):
    '''
    tif，size:512*512，gray
    :param dir_path: path to your images directory
    :return:
    '''
    MAX_LEN = 1200
    # img = Image.open(file_path).convert('RGB')
    # img = ImageEnhance.Contrast(img).enhance(3)

    # img = np.array(img)
    img_shape = img.shape

    image_size = (img_shape[1], img_shape[0])
    h, w = img_shape[1], img_shape[0]
    if (w < h):
        if (h > MAX_LEN):
            scale = 1.0 * MAX_LEN / h
            w = w * scale
            h = MAX_LEN
    elif (h <= w):
        if (w > MAX_LEN):
            scale = 1.0 * MAX_LEN / w
            h = scale * h
            w = MAX_LEN

    w = int(w // 16 * 16)
    h = int(h // 16 * 16)
    # h, w = 512, 512

    img_standard = cv2.resize(img, (h, w), interpolation=cv2.INTER_AREA)
    # cv2.imshow('resize', img_standard)
    # cv2.waitKey(0)
    print(img_standard.shape)
    img_new = img_standard
    img_new = np.asarray([img_new / 255.])
    return img_new, image_size


def line_detect(img):
    session_config = tf.ConfigProto(
        log_device_placement=True,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=False)

    with tf.Session(config=session_config) as sess:
        with tf.device('/cpu:0'):
            model = unet()
            model.load_weights('line_extract/model.weights')
            new_img = np.zeros((img.shape[0]+30, img.shape[1]+30, 3), dtype=np.uint8)
            new_img += 255
            # print(111111111, img.shape)
            # print(222222222, new_img.shape)
            new_img[15:-15, 15:-15] = img
            x, img_size = image_normalized(new_img)
            results = model.predict(x)
            img = dp.saveResult([results[0]], img_size, '')
            img = img[15:-15, 15:-15]
    return img

test_path = 'test'
save_path = 'predict'
dp = data_preprocess(flag_multi_class=False, num_classes=2)

# model = load_model('./model/model_4_0.9957594732443492.hdf5',custom_objects={'loss':loss,'binary_PTA':binary_PTA, 'binary_PFA':binary_PFA})
# with tf.device("/cpu:0"):
# model = load_model('./line_extract/model_17_0.9887387033492799.hdf5')
# with tf.device("/cpu:0"):
# session_config = tf.ConfigProto(
#     log_device_placement=True,
#     inter_op_parallelism_threads=0,
#     intra_op_parallelism_threads=0,
#     allow_soft_placement=False)
#
# with tf.Session(config=session_config) as sess:
#     with tf.device('/cpu:0'):
#         model = unet()
# model.load_weights('line_extract/model.weights')
img = line_detect(np.zeros((100, 100, 3), dtype=np.uint8))
Image.fromarray(img).save('line_extract/1.jpg')
print(11111111111111111111)


if __name__ == '__main__':

    test_path = 'test'
    save_path = 'predict'

    dp = data_preprocess(test_path=test_path,save_path=save_path,flag_multi_class=False, num_classes=2)

    # model = load_model('./model/model_4_0.9957594732443492.hdf5',custom_objects={'loss':loss,'binary_PTA':binary_PTA, 'binary_PFA':binary_PFA})
    model = load_model('./model/model_17_0.9887387033492799.hdf5')

    for name in os.listdir(test_path):
        print(name)
        image_path = os.path.join(test_path,name)
        x, img_size = image_normalized(image_path)
        results = model.predict(x)
        # print(results.shape)
        # cv2.imshow('result', results[0])
        # cv2.waitKey(2000)
        dp.saveResult([results[0]], img_size, os.path.splitext(name)[0])

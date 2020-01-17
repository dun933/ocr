import cv2
import numpy as np
from PIL import Image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tensorflow.python.platform import gfile


def image_normalized(img):
    '''
    tif，size:512*512，gray
    :param dir_path: path to your images directory
    :return:
    '''
    MAX_LEN = 600
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


def saveResult(npyfile, size):
    for i, item in enumerate(npyfile):
        img = item
        img_std = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        # img_std += 1
        img = np.squeeze(img, axis=-1)
        coor = np.argwhere(img < 0.2)
        for i in coor:
            img_std[i[0]][i[1]] = 255
            # img_std
        img_std = cv2.resize(img_std, size, interpolation=cv2.INTER_CUBIC)
        # print(img_std)
        # cv2.imwrite(os.path.join(self.save_path, ("%s." + self.img_type) % (name)), img_std)
    return img_std



sess = tf.Session()
with tf.device('/cpu:0'):
    with gfile.FastGFile('line_extract/model.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')  # 导入计算图

        # 需要有一个初始化的过程
        sess.run(tf.global_variables_initializer())

        # for tensor_name in tf.contrib.graph_editor.get_tensors(tf.get_default_graph()):
        #     print(tensor_name)
        input_x = sess.graph.get_tensor_by_name("input_1:0")
        out_softmax = sess.graph.get_tensor_by_name("conv2d_18/Sigmoid:0")
        # out_label = sess.graph.get_tensor_by_name("output:0")


        def line_detect(img):
            new_img = np.zeros((img.shape[0] + 60, img.shape[1] + 60, 3), dtype=np.uint8)
            new_img += 255
            # print(111111111, img.shape)
            # print(222222222, new_img.shape)
            new_img[30:-30, 30:-30] = img
            img, img_size = image_normalized(new_img)
            print(img.shape)
            img_out_softmax = sess.run(out_softmax, feed_dict={input_x: img})
            print(img_out_softmax.shape)
            img = saveResult(img_out_softmax, img_size)
            Image.fromarray(img).save('line_extract/1.jpg')
            img = img[30:-30, 30:-30]
            return img

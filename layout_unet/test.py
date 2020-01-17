import os
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))

from model_v3 import *
from data import *
import numpy as np
import cv2
from PIL import Image



def image_normalized(file_path):
    '''
    tif，size:512*512，gray
    :param dir_path: path to your images directory
    :return:
    '''
    img = cv2.imread(file_path)
    img_shape = img.shape
    image_size = (img_shape[1],img_shape[0])

    # img_standard = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    img = Image.fromarray(img)
    # img.thumbnail((1024, 1024), Image.ANTIALIAS)
    img.thumbnail((448, 448), Image.ANTIALIAS)
    new_shape = np.array(img).shape
    img = np.array(img)
    img = cv2.resize(img, (new_shape[1]-new_shape[1]%16+16, new_shape[0]-new_shape[0]%16+16),
     interpolation=cv2.INTER_CUBIC)
    img_standard = img

    print(img_standard.shape)
    img_new = img_standard
    img_new = np.asarray([img_new / 255.])
    return img_new,image_size


if __name__ == '__main__':

    #path to images which aring wating for predicting
    test_path = "CamVid/test"

    # save the predict images
    save_path = "CamVid/predict"

    test_path = 'test'
    # test_path = 'data/publaynet/val_img'
    save_path = 'predict'

    dp = data_preprocess(test_path=test_path,save_path=save_path,flag_multi_class=True,num_classes=6)

    #load model
    # model = load_model('./model/model_26_0.9567285537719726.hdf5')
    # model_path = './model/model_49_0.9783685684204102.hdf5'
    model_path = 'model/model_71_0.9756122589111328.hdf5'
    print(model_path)
    model = unet(pretrained_weights=model_path, input_size=(None, None, 3), num_class=5)


    import time
    for name in os.listdir(test_path):
        st_time = time.time()

        image_path = os.path.join(test_path,name)
        x,img_size = image_normalized(image_path)
        results = model.predict(x)
        print(name, 'time:', time.time() - st_time)
        dp.saveResult([results[0]],img_size,name.split('.')[0])


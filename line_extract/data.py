from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from PIL import ImageEnhance, Image
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import cv2
import warnings
import random

warnings.filterwarnings("ignore")




BackGround = [255, 255, 255]
Line = [0, 0, 0]

COLOR_DICT = np.array([Line, BackGround])


class data_preprocess:
    def __init__(self, train_path=None, image_folder=None, label_folder=None,
                 valid_path=None,valid_image_folder =None,valid_label_folder = None,
                 test_path=None, save_path=None,
                 img_rows=512, img_cols=512,
                 flag_multi_class=False,
                 num_classes = 2):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.train_path = train_path
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.valid_path = valid_path
        self.valid_image_folder = valid_image_folder
        self.valid_label_folder = valid_label_folder
        self.test_path = test_path
        self.save_path = save_path
        self.data_gen_args = dict(rotation_range=5,
                                  width_shift_range=0.5,
                                  height_shift_range=0.5,
                                  shear_range=0.5,
                                  zoom_range=0.5,
                                  vertical_flip=True,
                                  horizontal_flip=True,
                                  fill_mode='nearest')
        self.image_color_mode = "rgb"
        self.label_color_mode = 'grayscale'

        self.flag_multi_class = flag_multi_class
        self.num_class = num_classes
        self.target_size = (800, 800)
        self.img_type = 'jpg'

    def adjustData(self, img, label):
        if (self.flag_multi_class):
            img = img / 255.
            label = label[:, :, :, 0] if (len(label.shape) == 4) else label[:, :, 0]
            new_label = np.zeros(label.shape + (self.num_class,))
            for i in range(self.num_class):
                new_label[label == i, i] = 1
            label = new_label
        elif (np.max(img) > 1):
            img = img / 255.
            # label = label[:, :, :, 1] if (len(label.shape) == 4) else label[:, :, 1]
            label = label / 255.
            label = 1.0 - label
            label[label > 0.5] = 1
            label[label <= 0.5] = 0
            # print(label.shape)
            # print(len(np.argwhere(label==0)))
        return (img, label)

    def trainGenerator(self, batch_size, image_save_prefix="image", label_save_prefix="label",
                       save_to_dir=None, seed=7):
        '''
        can generate image and label at the same time
        use the same seed for image_datagen and label_datagen to ensure the transformation for image and label is the same
        if you want to visualize the results of generator, set save_to_dir = "your path"
        '''
        image_datagen = ImageDataGenerator(**self.data_gen_args)
        label_datagen = ImageDataGenerator(**self.data_gen_args)
        image_generator = image_datagen.flow_from_directory(
            self.train_path,
            classes=[self.image_folder],
            class_mode=None,
            color_mode=self.image_color_mode,
            target_size=self.target_size,
            batch_size=batch_size,
            save_to_dir=save_to_dir,
            save_prefix=image_save_prefix,
            seed=seed)
        label_generator = label_datagen.flow_from_directory(
            self.train_path,
            classes=[self.label_folder],
            class_mode=None,
            color_mode=self.label_color_mode,
            target_size=self.target_size,
            batch_size=batch_size,
            save_to_dir=save_to_dir,
            save_prefix=label_save_prefix,
            seed=seed)
        train_generator = zip(image_generator, label_generator)
        for (img, label) in train_generator:

            img = img.astype(np.uint8)
            img = Image.fromarray(img[0]).convert('RGB')
            img = ImageEnhance.Contrast(img).enhance(random.randint(1, 4)*0.5)

            # img = np.array(img)
            # cv2.imshow('1', img)
            # cv2.waitKey(0)

            img = np.array(img, dtype=np.float32)
            img = np.expand_dims(img, axis=0)
            img, label = self.adjustData(img, label)
            yield (img, label)

    def testGenerator(self):
        filenames = os.listdir(self.test_path)
        for filename in filenames:
            img = io.imread(os.path.join(self.test_path, filename), as_gray=False)
            img = img / 255.
            img = trans.resize(img, self.target_size, mode='constant')
            img = np.reshape(img, img.shape + (1,)) if (not self.flag_multi_class) else img
            img = np.reshape(img, (1,) + img.shape)
            yield img

    def validLoad(self, batch_size,seed=7):
        image_datagen = ImageDataGenerator(**self.data_gen_args)
        label_datagen = ImageDataGenerator(**self.data_gen_args)
        image_generator = image_datagen.flow_from_directory(
            self.valid_path,
            classes=[self.valid_image_folder],
            class_mode=None,
            color_mode=self.image_color_mode,
            target_size=self.target_size,
            batch_size=batch_size,
            seed=seed)
        label_generator = label_datagen.flow_from_directory(
            self.valid_path,
            classes=[self.valid_label_folder],
            class_mode=None,
            color_mode=self.label_color_mode,
            target_size=self.target_size,
            batch_size=batch_size,
            seed=seed)
        train_generator = zip(image_generator, label_generator)
        for (img, label) in train_generator:
            img, label = self.adjustData(img, label)
            yield (img, label)
        # return imgs,labels

    def saveResult(self, npyfile, size, name, threshold=127):
        for i, item in enumerate(npyfile):
            img = item
            img_std = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            # img_std += 1
            if self.flag_multi_class:
                for row in range(len(img)):
                    for col in range(len(img[row])):
                        num = np.argmax(img[row][col])
                        img_std[row][col] = COLOR_DICT[num]
            else:
                # for k in range(len(img)):
                #     for j in range(len(img[k])):
                #         num = img[k][j]
                #         if num < (threshold/255.0):
                #             img_std[k][j] = [255, 255, 255]
                #         else:
                #             img_std[k][j] = [0, 0, 0]
                img = np.squeeze(img, axis=-1)
                coor = np.argwhere(img < 0.5)
                for i in coor:
                    img_std[i[0]][i[1]] = 255
                # img_std
            img_std = cv2.resize(img_std, size, interpolation=cv2.INTER_CUBIC)
            # print(img_std)
            # cv2.imwrite(os.path.join(self.save_path, ("%s." + self.img_type) % (name)), img_std)
            return img_std
        return False

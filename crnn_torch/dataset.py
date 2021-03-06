#!/usr/bin/python
# -*- encoding:utf-8 -*-
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import six
import sys
import os
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import numpy as np
import utils.util as util
import utils.keys as keys


class imageDataset(Dataset):
    def __init__(self, root, mapping, transform=None, target_transform=None):
        """Initialization for image Dataset.
        args
        root (string): directory of images
        mapping (string): file of mapping filename and its labels
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.images = list()
        self.labels = list()
        self.name = list()

        with open(mapping, encoding='utf-8-sig') as f:
            pair_list = f.readlines()
            self.nSample = len(pair_list)

        for pair in pair_list:
            items = pair.strip().split()
            # print(items)
            img = items[0]
            # label = items[1] # No blank in the middle of the label string.
            label = ' '.join(items[1:])
            self.images.append(img)
            self.labels.append(keyFilte(label, keys.alphabet))
            self.name.append(img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        try:
            img = Image.open(os.path.join(self.root, self.images[index])).convert('L')
        # print(os.path.join(self.root, self.images[index]))
        # print(img)
        except Exception:
            print('image load error')
            # return self[index + 1]
        if self.transform is not None:
            img = self.transform(img)
        return (img, self.labels[index], self.name[index])


class synthDataset(Dataset):
    def __init__(self, fontpath, fontsize_range='32-36', text_generator=None, transform=None, target_transform=None):
        '''Initialization of synthDataset
        args
            fonts (string): font file path
            fontsize_range (int): font sizes
            text_generator: class of text generator
        '''
        self.fontsize_range = fontsize_range.strip().split('-')
        font_list = os.listdir(fontpath)
        self.fonts = [os.path.join(fontpath, font) for font in font_list]
        self.alphabet = keys.alphabet
        self.text_generator = text_generator
        self.gen_len = self.text_generator.__len__()
        self.len_thr = self.text_generator.__len_thr__()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        if self.gen_len < 128000:
            length = 128000
        else:
            length = self.gen_len
        return length

    def __getitem__(self, index):
        cur = index % self.gen_len
        text = self.text_generator.__getitem__(cur)
        text = keyFilte(text, self.alphabet)

        # Random foreground and background gray.
        # Notice: background is close to 0, and text is close to 255.
        bg_gray = util.random_scale(0, 10)
        tx_gray = util.random_scale(bg_gray + 100, 255)
        interval = random.randint(0, 5)

        # Add multiple fonts
        font_choice = random.choice(self.fonts)
        fontsize_choice = random.randint(int(self.fontsize_range[0]), int(self.fontsize_range[1]))
        font = ImageFont.truetype(font_choice, fontsize_choice)

        # Text width
        a_r = len(text)
        imgH = fontsize_choice + 4
        imgW = 2 + imgH * a_r + interval * (a_r - 1)
        bg = Image.new('L', (imgW, imgH), color=bg_gray)
        offset = 1
        draw = ImageDraw.Draw(bg)
        for i in range(len(text)):
            # If not in keys, randomly replace
            o_char = text[i]
            if self.alphabet.find(o_char) == -1:
                o_char = self.alphabet[random.randint(0, len(self.alphabet) - 1)]
            draw.text((offset, 1), o_char, tx_gray, font=font)
            offset += fontsize_choice + interval

        # Add rotation
        rotate_angle = random.gauss(0, 0.05)
        bg = bg.rotate(rotate_angle, expand=1)

        # Add noise
        nptxt = np.array(bg)
        nptxt = util.Addblur(nptxt, 3)
        nptxt = util.motionBlur(nptxt, 2)
        nptxt = util.AffineTransform(nptxt, 0.02)
        scale = random.randint(1, 4)
        nptxt = cv2.resize(nptxt, (0, 0), fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
        nptxt = cv2.resize(nptxt, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(nptxt)

        if self.transform is not None:
            img = self.transform(img)
        return (img, text)


class hyperDataset(Dataset):
    def __init__(self, root, mapping, font, fontsize=32, text_generator=None, transform=None, target_transform=None):
        '''Initialization of hyperDataset. The training dataset is composed of 
        generated and annotated images.
        args
            root (string): annotated image root path
            mapping (string): annotated and image mapping
            font (string): font file path
            fontsize (int): font size 
            text_generator: class of text generator
        '''
        self.synth_dataset = synthDataset(font, fontsize, text_generator, transform, target_transform)
        self.image_dataset = imageDataset(root, mapping, transform, target_transform)

    def __len__(self):
        return self.synth_dataset.__len__() + self.image_dataset.__len__()

    def __getitem__(self, index):
        switch = random.randint(0, 1)
        if switch:
            index = index % self.synth_dataset.__len__()
            return self.synth_dataset.__getitem__(index)
        else:
            index = index % self.image_dataset.__len__()
            return self.image_dataset.__getitem__(index)


class resizeNormalize(object):

    def __init__(self, maxW, imgH, interpolation=Image.BILINEAR):
        self.imgH = imgH
        self.maxW = maxW
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        ratio = img.size[0] / img.size[1]
        imgW = int(self.imgH * ratio)
        img = img.resize((imgW, self.imgH), self.interpolation)
        padding = (0, 0, self.maxW - imgW, 0)
        img = ImageOps.expand(img, border=padding, fill='black')
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class Normalize(object):

    def __init__(self, maxW):
        self.maxW = maxW
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        imgW = img.size[0]
        padding = (0, 0, self.maxW - imgW, 0)
        img = ImageOps.expand(img, border=padding, fill='black')
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class graybackNormalize(object):

    def __init__(self):
        return

    def __call__(self, img):
        return img.point(lambda i: 255 - i)


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, trans="normal", keep_ratio=True, min_ratio=1):
        """
        args:
            imgH: can be divided by 32
            maxW: the maximum width of the collection
            keep_ratio:
            min_ratio:
        """
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.trans = trans

    def __call__(self, batch):
        images, labels, name = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        maxW = 280
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                if w > maxW: maxW = w
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(self.min_ratio * imgH, imgW)  # assure imgW >= imgH

        transform = Normalize(imgW)
        if self.trans == "resize":
            transform = resizeNormalize(imgW, imgH)
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels, name


def keyFilte(text, alphabet):
    valid_char = []
    for char in text:
        if alphabet.find(char) != -1:
            valid_char.append(char)
    if len(valid_char) == 0:
        for i in range(5):
            valid_char.append(random.choice(alphabet))

    return ''.join(valid_char)

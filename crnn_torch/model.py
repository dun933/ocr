
import argparse
import torch
from PIL import Image, ImageOps
import numpy as np

import crnn_torch.utils.keys as keys


# import yaml
import crnn_torch.models.crann as crann
from torch.autograd import Variable
from torch.nn import CTCLoss
import crnn_torch.utils.util as util
from crnn_torch.config import crann as opt
import torchvision.transforms as transforms


parser = argparse.ArgumentParser()
# parser.add_argument('--yaml', default='crnn_torch/config/crann.yml', help='path to config yaml')
alphabet = keys.alphabet
nClass = len(alphabet) + 1
# opt = parser.parse_args()
# f = open(opt.yaml)
# opt = yaml.load(f)
if opt.N_GPU > 1:
    opt.RNN['multi_gpu'] = True
else:
    opt.RNN['multi_gpu'] = False

criterion = CTCLoss()
converter = util.strLabelConverter(alphabet)
model = crann.CRANN(opt, nClass)
device = torch.device("cuda:0")
model.load_state_dict(torch.load("crnn_torch/CHECKPOINTS_SAVE_PATH/best_model.pth")["state_dict"])
model.to(device)
model.eval()
print(222222222222222222222222222)
transform = transforms.Compose([transforms.ToTensor()])


def predict(img):
    copy_img = img.copy()
    img = img.point(lambda i: 255 - i)
    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / 32
    width = int(width / scale)

    img = img.resize([width, 32], Image.BILINEAR)
    print('img width', width)
    # if width > 280:
    #     padding = (0, 0, 0, 0)
    # else:
    #     padding = (0, 0, 280 - width, 0)
    #
    # img = ImageOps.expand(img, border=padding, fill='black')
    # print(8888888888888888, img.size)

    # image = torch.tensor(img)
    # img = np.array(img)
    # image = img.astype(np.float32)
    # image = torch.from_numpy(image)
    image = transform(img)

    # image = transforms.ToTensor()
    image.sub_(0.5).div_(0.5)

    # bsz = image.size(0)
    # if torch.cuda.is_available():
    #     image = image.cuda()
    # else:
    #     image = image.cpu()+
    # image = image.cuda()
    # image = image.cpu()
    image = image.cuda()
    # image = image.view(1, 1, *image.size())
    image = image.unsqueeze(0)
    predict = model(image)
    predict_len = Variable(torch.IntTensor([predict.size(0)] * 1))
    # d, c = predict.topk(3, 2)
    # print(c.shape)
    # print(2222222222222222222, predict.topk(3, 2))
    # _, acc = predict.max(2)
    acc = predict.softmax(2).topk(5)
    # _, acc = predict.max(2)
    # acc = acc.transpose(1, 0).contiguous().view(-1)
    # print(acc.shape)
    # acc = acc.transpose(1, 0).contiguous().view(-1)
    # acc = c.transpose(1, 0).contiguous().view(-1)
    # d = torch.exp(d)
    # print(d, acc)
    sim_preds = converter.decode(acc, predict_len.data, copy_img, scale, raw=False)
    return sim_preds


def predict1(img):

    img = img.point(lambda i: 255 - i)
    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / 32
    width = int(width / scale)

    img = img.resize([width, 32], Image.ANTIALIAS)
    if width > 280:
        padding = (0, 0, 0, 0)
    else:
        padding = (0, 0, 280 - width, 0)

    img = ImageOps.expand(img, border=padding, fill='black')
    # print(8888888888888888, img.size)

    # transform = transforms.Compose([transforms.ToTensor()])
    # image = torch.tensor(img)
    # img = np.array(img)
    # image = img.astype(np.float32)
    # image = torch.from_numpy(image)
    image = transform(img)

    # image = transforms.ToTensor()
    image.sub_(0.5).div_(0.5)

    # bsz = image.size(0)
    # if torch.cuda.is_available():
    #     image = image.cuda()
    # else:
    #     image = image.cpu()+
    # image = image.cuda()
    # image = image.cpu()
    image = image.cuda()
    # image = image.view(1, 1, *image.size())
    image = image.unsqueeze(0)
    predict = model(image)
    predict_len = Variable(torch.IntTensor([predict.size(0)] * 1))
    # d, c = predict.topk(3, 2)
    # print(c.shape)
    # print(2222222222222222222, predict.topk(3, 2))
    # _, acc = predict.max(2)
    # acc = predict.softmax(2).topk(3)
    _, acc = predict.max(2)
    acc = acc.transpose(1, 0).contiguous().view(-1)
    # print(acc.shape)
    # acc = acc.transpose(1, 0).contiguous().view(-1)
    # acc = c.transpose(1, 0).contiguous().view(-1)
    # d = torch.exp(d)
    # print(d, acc)
    sim_preds = converter.decode1(acc.data, predict_len.data, raw=False)
    return sim_preds
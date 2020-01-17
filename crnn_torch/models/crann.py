#!/usr/bin/python
# encoding: utf-8

import crnn_torch.models.convnet as ConvNets
import crnn_torch.models.recurrent as SeqNets
import torch.nn as nn
import torch.nn.parallel


class CRANN(nn.Module):
    def __init__(self, crann_config, n_class):
        super(CRANN, self).__init__()
        self.ngpu = crann_config.N_GPU
        cnn_conf = crann_config.CNN
        print('Constructing {}'.format(cnn_conf['MODEL']))
        self.cnn = ConvNets.__dict__[cnn_conf['MODEL']]()

        rnn_conf = crann_config.RNN
        print('Constructing {}'.format(rnn_conf['MODEL']))
        self.rnn = SeqNets.__dict__[rnn_conf['MODEL']](rnn_conf, n_class)

        # self.classifier = nn.Sequential(
        #     # nn.Dropout(),
        #     nn.Linear(512, 1024),
        #     nn.Tanh(),
        #     # nn.Dropout(0.1),
        #     nn.Linear(1024, n_class),
        # )

    def forward(self, input):
        c_feat = data_parallel(self.cnn, input, self.ngpu)

        # logger.info(list(self.cnn))
        # params = list(self.cnn.named_parameters())
        # for (name,param) in params:
        #     logger.info(name)
        #     if not param.grad is None:
        #         logger.info(torch.sum(param.grad))
        #     logger.info("----------------")
        #_N, _C, _H, _W = input.size() 
        #print("input size, N:{0}, C{1}, H{2}, W{3}".format(_N, _C, _H, _W))
        # logger.info(c_feat.size())
        b, c, h, w = c_feat.size()
        assert h == 1, "the height of the conv must be 1"

        c_feat = c_feat.squeeze(2)

        c_feat = c_feat.permute(2, 0, 1) # [w, b, c]
        w, b, c = c_feat.size()

        # t_rec = c_feat.contiguous().view(w*b, c)
        # output = self.classifier(t_rec)
        # output = output.view(w, b, -1)

        output = data_parallel(self.rnn, c_feat, self.ngpu, dim=1)
        # print(output[0])
        return output

def data_parallel(model, input, ngpu, dim=0):
    if isinstance(input.data, torch.cuda.FloatTensor) and ngpu > 1:
        output = nn.parallel.data_parallel(model, input, range(ngpu), dim=dim)
    else:
        output = model(input)
    return output

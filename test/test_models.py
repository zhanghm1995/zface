'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-08-01 17:28:39
Email: haimingzhang@link.cuhk.edu.cn
Description: Testing the models
'''

import sys
sys.path.append('./')
sys.path.append('../')
import torch


from models.generator import HififaceGenerator


def test_HififaceGenerator():
    net = HififaceGenerator(size=512)
    net = net.cuda()

    Xs = torch.randn(1, 3, 512, 512).cuda()
    Xt = torch.randn(1, 3, 512, 512).cuda()
    print(Xs.dtype, Xt.dtype)

    out = net(Xs, Xt)[0]
    print(out.shape)


def test_gen2_HifiFaceGenerator():
    from models.gen2 import HififaceGenerator
    net = HififaceGenerator(activation="mish")
    print(net)

    Xs = torch.randn(1, 3, 256, 256)
    Xt = torch.randn(1, 3, 256, 256)

    target_mask = torch.randn(1, 1, 256, 256)

    output = net(Xs, Xt, target_mask)[0]
    print(output.shape)




if __name__ == "__main__":
    test_gen2_HifiFaceGenerator()



import sys

# sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from .core.raft import RAFT
from .core.utils import flow_viz
from .core.utils.utils import InputPadder

DEVICE = 'cuda'


# parser = argparse.ArgumentParser()
# # parser.add_argument('--model', help="restore checkpoint")
# # parser.add_argument('--path', help="dataset for evaluation")
# parser.add_argument('--small', action='store_true', help='use small model')
# parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
# parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
# args = parser.parse_args()


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
    cv2.waitKey()

def demo(args, imfile1, imfile2):
    model =RAFT(args)
    # model = torch.nn.DataParallel(RAFT(args))
    #加载模型时去掉module.前缀
    state_dict = torch.load('/home/lf/diff/Code/thirdparty/RAFT/checkpoint/raft-things.pth')
    new_state_dict = {}
    for k,v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    # model.load_state_dict(torch.load('/home/lf/diff/Code/thirdparty/RAFT/checkpoint/raft-things.pth'))

    # model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        padder = InputPadder(imfile1.shape)
        image1, image2 = padder.pad(imfile1, imfile2)

        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

        return flow_up


def demo_vis(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up)
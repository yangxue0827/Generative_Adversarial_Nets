# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import numpy as np
from PIL import Image, ImageOps
import tools


def _image_preprocessing(filename, xsize, ysize):
    im = Image.open(filename)

    if filename.endswith('.png'):
        im = im.convert('RGB')
    downsampled_im = ImageOps.fit(im, (xsize, ysize), method=Image.LANCZOS)
    norm_im = np.array(downsampled_im, dtype=np.float32) / 255.

    downsampled_im.close()
    im.close()
    return norm_im

if __name__ == '__main__':
    names = []
    print(sys.argv[1], type(sys.argv[1]))
    for name in os.listdir(sys.argv[1]):
        if name.endswith('.jpg'):
            names.append(name[:-4])

    dataset_X = np.zeros((len(names), 256, 256, 3))
    dataset_Y = np.zeros((len(names), 256, 256, 3))

    for i in range(len(names)):
        tools.view_bar('read data: ', i + 1, len(names))
        dataset_X[i] = _image_preprocessing(os.path.join(sys.argv[1], names[i] + '.jpg'), 256, 256)
        dataset_Y[i] = _image_preprocessing(os.path.join(sys.argv[1], names[i] + '.png'), 256, 256)

    np.save(sys.argv[1].split('/')[-1] + '_x.npy', dataset_X)
    np.save(sys.argv[1].split('/')[-1] + '_y.npy', dataset_Y)
    print('\ndone!')

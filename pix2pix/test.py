# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pix2pix import Pix2pix
import tools

A = np.load('CMP_facade_DB_test_y.npy')  # Architectural labels
B = np.load('CMP_facade_DB_test_x.npy')  # Photo

# with tf.device('/gpu:0'):
#     model = Pix2pix(256, 256, ichan=3, ochan=3)
model = Pix2pix(256, 256, ichan=3, ochan=3)

model_path = "models/model.ckpt"
images_path = 'images'
tools.mkdir(model_path.split('/')[0])
tools.mkdir(images_path)

saver = tf.train.Saver()

# with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if os.path.exists(model_path + '.index'):
        saver.restore(sess, model_path)
        print('restore model...')

    for i in range(len(A)):
        image = np.zeros([B.shape[1], B.shape[1] * 3, 3])
        image[:, 0:B.shape[1], :] = A[i]
        image[:, B.shape[1]: 2*B.shape[1], :] = \
            (model.sample_generator(sess, np.expand_dims(A[i], axis=0), is_training=False)[0] + 1.) / 2.
        image[:, 2*B.shape[1]:3*B.shape[1], :] = B[i]
        plt.imsave('images/test_%d.jpg' % i, image)


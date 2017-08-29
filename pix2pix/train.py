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

# 200 epoch
iters = 200*588  # taken from pix2pix paper §5.2
batch_size = 1  # taken from pix2pix paper §5.2

A = np.load('CMP_facade_DB_train_y.npy')  # Architectural labels
B = np.load('CMP_facade_DB_train_x.npy')  # Photo

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

    for step in range(iters):
        a = np.expand_dims(A[step % A.shape[0]], axis=0)
        b = 2. * np.expand_dims(B[step % B.shape[0]], axis=0) - 1.  # normalize because generator use tanh activation in its output layer

        gloss_curr, dloss_curr = model.train_step(sess, a, a, b)

        if step % 200 == 0:
            print('Step %d: G loss: %f | D loss: %f' % (step, gloss_curr, dloss_curr))

        if step % 1000 == 0:
            # fig = plt.figure()
            # fig.set_size_inches(10, 10)
            # fig.subplots_adjust(left=0, bottom=0,
            #                        right=1, top=1, wspace=0, hspace=0.1)
            # p = np.random.permutation(B.shape[0])

            # for i in range(0, 81, 3):
                # Plot 3 images: First is the architectural label, second the generator output, third the ground truth
                # fig.add_subplot(9, 9, i + 1)
                # plt.imshow(A[p[i // 3]])
                # plt.axis('off')
                # # fig.add_subplot(9, 9, i + 2)
                # plt.imshow((model.sample_generator(sess, np.expand_dims(A[p[i // 3]], axis=0), is_training=True)[0] + 1.) / 2.)
                # plt.axis('off')
                # fig.add_subplot(9, 9, i +3)
                # plt.imshow(B[p[i // 3]])
                # plt.axis('off')
            # plt.savefig('images/iter_%d.jpg' % step)
            # plt.close()
            
            p = np.random.permutation(B.shape[0])
            image = np.zeros([B.shape[1] * 9, B.shape[1] * 9, 3])
            for i in range(0, 9):
                for j in range(0, 9, 3):
                    image[(i*B.shape[1]):((i+1)*B.shape[1]), (j*B.shape[1]):((j+1)*B.shape[1]), :] \
                        = A[p[i * 9 + j]]

                    image[(i*B.shape[1]):((i+1)*B.shape[1]), ((j+1)*B.shape[1]):((j+2)*B.shape[1]), :] \
                        = (model.sample_generator(sess, np.expand_dims(A[p[i * 9 + j]], axis=0), is_training=False)[0] + 1.) / 2.

                    image[(i*B.shape[1]):((i+1)*B.shape[1]), ((j+2)*B.shape[1]):((j+3)*B.shape[1]), :] \
                        = B[p[i * 9 + j]]
            plt.imsave('images/iter_%d.jpg' % step, image)
            
        if step % 1000 == 0:
            # Save the model
            save_path = saver.save(sess, model_path)
            print("Model saved in file: %s" % save_path)

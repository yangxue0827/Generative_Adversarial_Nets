# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import cv2
import random

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

flags = tf.app.flags
flags.DEFINE_integer("iter", 1000000, "Iteration to train [1000000]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_string("model_path", './model/cgan.model', "Save model path ['./model/cgan.model']")
flags.DEFINE_boolean("is_train", False, "Train or test [False]")
flags.DEFINE_integer("test_number", None, "The number that want to generate, if None, generate randomly [None]")
FLAGS = flags.FLAGS

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
Z_dim = 100
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


""" Discriminator Net model """
X = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, y_dim])

D_W1 = tf.Variable(xavier_init([X_dim + y_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


def discriminator(x, y):
    inputs = tf.concat(axis=1, values=[x, y])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


""" Generator Net model """
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

G_W1 = tf.Variable(xavier_init([Z_dim + y_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def generator(z, y):
    inputs = tf.concat(axis=1, values=[z, y])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return fig


G_sample = generator(Z, y)
D_real, D_logit_real = discriminator(X, y)
D_fake, D_logit_fake = discriminator(G_sample, y)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if os.path.exists(os.path.join(FLAGS.model_path + '.index')):
    saver.restore(sess, FLAGS.model_path)
    print('restore model...')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

mkdir('train/')
mkdir('test/')
mkdir('model/')


def test(path, index):

    n_sample = 16

    Z_sample = sample_Z(n_sample, Z_dim)
    y_sample = np.zeros(shape=[n_sample, y_dim])
    if FLAGS.test_number:
        y_sample[:, FLAGS.test_number] = 1
    else:
        for i in range(n_sample):
            y_sample[i][random.randint(0, y_dim - 1)] = 1

    samples = sess.run(G_sample, feed_dict={Z: Z_sample, y: y_sample})

    fig = plot(samples)
    image_path = '{}/{}.png'.format(path, str(index).zfill(3))
    plt.savefig(image_path, bbox_inches='tight')
    plt.close(fig)

    return image_path


def main(_):
    i = 0

    if FLAGS.is_train:
        for it in range(FLAGS.iter):

            X_mb, y_mb = mnist.train.next_batch(FLAGS.batch_size)

            Z_sample = sample_Z(FLAGS.batch_size, Z_dim)
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: Z_sample, y: y_mb})
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_sample, y: y_mb})

            if it % 1000 == 0:
                test('./train', i)
                i += 1
                print('Iter: {}'.format(it))
                print('D loss: {:.4}'.format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                saver.save(sess, FLAGS.model_path)

    else:
        img_path = test('./test', FLAGS.test_number)
        img = cv2.imread(img_path)
        cv2.imshow('image', img)
        cv2.waitKey(0)
    sess.close()


if __name__ == '__main__':
    tf.app.run()
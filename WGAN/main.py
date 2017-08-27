# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import scipy.misc
import numpy as np

from model import WGAN
from utils import pp, visualize, to_json, show_all_variables

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 28, "The size of image to use (will be center cropped). [28]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 28, "The size of the output images to produce [28]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "mnist", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for center cropping first, then resize the images [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("critic_num", 5, "The number of times the discriminator is updated [5]")
flags.DEFINE_float("clip_up", 0.01, "Upper bounds of the truncation [0.01]")
flags.DEFINE_float("clip_down", -0.01, "Lower bounds of the truncation [-0.01]")
flags.DEFINE_string("mode", "gp", "The mode of wgan, regular or gp [gp]")
flags.DEFINE_float("LAMBDA", 10., "Parameters of gp mode [10.]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True

    with tf.Session(config=run_config) as sess:
        if FLAGS.dataset == 'mnist':
            wgan = WGAN(
                sess,
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                output_width=FLAGS.output_width,
                output_height=FLAGS.output_height,
                batch_size=FLAGS.batch_size,
                sample_num=FLAGS.batch_size,
                y_dim=10,
                dataset=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_fname_pattern,
                crop=FLAGS.crop,
                checkpoint_dir=FLAGS.checkpoint_dir,
                sample_dir=FLAGS.sample_dir,
                clip_up=FLAGS.clip_up,
                clip_down=FLAGS.clip_down,
                critic_num=FLAGS.critic_num,
                mode=FLAGS.mode,
                LAMBDA=FLAGS.LAMBDA)
        else:
            wgan = WGAN(
                sess,
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                output_width=FLAGS.output_width,
                output_height=FLAGS.output_height,
                batch_size=FLAGS.batch_size,
                sample_num=FLAGS.batch_size,
                dataset=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_fname_pattern,
                crop=FLAGS.crop,
                checkpoint_dir=FLAGS.checkpoint_dir,
                sample_dir=FLAGS.sample_dir,
                clip_up=FLAGS.clip_up,
                clip_down=FLAGS.clip_down,
                critic_num=FLAGS.critic_num,
                mode=FLAGS.mode,
                LAMBDA=FLAGS.LAMBDA)

        show_all_variables()

        if FLAGS.train:
            wgan.train(FLAGS)
        else:
            if not wgan.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")

        # to_json("./web/js/layers.js", [wgan.h0_w, wgan.h0_b, wgan.g_bn0],
        #                 [wgan.h1_w, wgan.h1_b, wgan.g_bn1],
        #                 [wgan.h2_w, wgan.h2_b, wgan.g_bn2],
        #                 [wgan.h3_w, wgan.h3_b, wgan.g_bn3],
        #                 [wgan.h4_w, wgan.h4_b, None])

        # Below is codes for visualization
        OPTION = 1
        visualize(sess, wgan, FLAGS, OPTION)


if __name__ == '__main__':
    tf.app.run()

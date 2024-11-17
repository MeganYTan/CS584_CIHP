from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
from utils import *

N_CLASSES = 20
DATA_DIR = './datasets/CIHP'
LIST_PATH = './datasets/CIHP/list/val.txt'
DATA_ID_LIST = './datasets/CIHP/list/val_id.txt'
RESTORE_FROM = './checkpoint/CIHP_pgn'

def main():
    """Run inference to get parsing output."""
    # Create TensorFlow session
    coord = tf.train.Coordinator()
    with tf.name_scope("create_inputs"):
        reader = ImageReader(DATA_DIR, LIST_PATH, DATA_ID_LIST, None, False, False, False, coord)
        image = reader.image
        image_rev = tf.reverse(image, tf.stack([1]))
        image_batch = tf.stack([image, image_rev])

    # Original size of the image
    h_orig, w_orig = tf.to_float(tf.shape(image_batch)[1]), tf.to_float(tf.shape(image_batch)[2])

    # Create network
    with tf.variable_scope('', reuse=False):
        net = PGNModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)

    # Parsing output
    parsing_out1 = net.layers['parsing_fc']
    parsing_out2 = net.layers['parsing_rf_fc']
    parsing_out1 = tf.image.resize_images(parsing_out1, tf.shape(image_batch)[1:3])
    parsing_out2 = tf.image.resize_images(parsing_out2, tf.shape(image_batch)[1:3])
    raw_output = tf.reduce_mean(tf.stack([parsing_out1, parsing_out2]), axis=0)
    head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)
    raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output]), axis=0)
    raw_output_all = tf.expand_dims(raw_output_all, dim=0)
    pred_all = tf.argmax(raw_output_all, axis=3)
    pred_all = tf.expand_dims(pred_all, dim=3)  # Create 4D tensor

    # Initialize variables and load the model
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Load model weights
    restore_var = tf.global_variables()
    loader = tf.compat.v1.train.Saver(var_list=restore_var)
    if RESTORE_FROM is not None:
        if load(loader, sess, RESTORE_FROM):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

    parsing_output = sess.run(pred_all)
    print("Shape of parsing output:", parsing_output.shape)
    print("Unique values in array: ", np.unique(parsing_output))

if __name__ == '__main__':
    main()

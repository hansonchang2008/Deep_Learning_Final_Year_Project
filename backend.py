import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
current_dir = r'%s' % os.getcwd().replace('\\','/')

def get_one_image(img_dir):
     image = Image.open(img_dir)
     image_arr = np.array(image)
     
     print(image_arr)
     return image_arr

def inference(images, batch_size, n_classes, keep_prob):
    # conv1, shape = [kernel_size, kernel_size, channels, kernel_numbers]
    with tf.variable_scope("conv1", reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 3, 50],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[50],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name="conv1")

    # pool1 && norm1
    with tf.variable_scope("pooling1_lrn") as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling1")
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope("conv2", reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 50, 50],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[50],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name="conv2")

    # pool2 && norm2
    with tf.variable_scope("pooling2_lrn") as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling2")
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm2')

#Two new conv layers are added
    # conv3
    with tf.variable_scope("conv3", reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 50, 50],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[50],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(1))
        conv = tf.nn.conv2d(norm2, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name="conv3")

    # pool3 && norm3
    with tf.variable_scope("pooling3_lrn") as scope:
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling3")
        norm3 = tf.nn.lrn(pool3, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm3')

    # conv4
    with tf.variable_scope("conv4", reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 50, 50],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[50],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(1))
        conv = tf.nn.conv2d(norm3, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name="conv4")

    # pool4 && norm4
    with tf.variable_scope("pooling4_lrn") as scope:
        pool4 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling4")
        norm4 = tf.nn.lrn(pool4, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm4')
###New layers ended

    # full-connect1
    with tf.variable_scope("fc1", reuse=tf.AUTO_REUSE) as scope:
        reshape = tf.reshape(norm4, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable("weights",
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(1))
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name="fc1")
        #dropout
        fc1_drop = tf.nn.dropout(fc1, keep_prob=keep_prob) #keep_prob=0.9
    # full_connect2
    with tf.variable_scope("fc2", reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable("weights",
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(1))
        fc2 = tf.nn.relu(tf.matmul(fc1_drop, weights) + biases, name="fc2")
        #dropout
        fc2_drop = tf.nn.dropout(fc2, keep_prob=keep_prob) #keep_prob=0.9

    # softmax
    with tf.variable_scope("softmax_linear", reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable("weights",
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(1))
        softmax_linear = tf.add(tf.matmul(fc2_drop, weights), biases, name="softmax_linear")
    return softmax_linear



def test(test_file):
    log_dir = current_dir
    image_arr = get_one_image(test_file) 

    with tf.Graph().as_default():
        image = tf.cast(image_arr, tf.float32)
        image = tf.reshape(image, [1, 192, 256, 3])
         
        print(image.shape)
        p = inference(image,1,3,1) 
        logits = tf.nn.softmax(p)
        x = tf.placeholder(tf.float32,shape = [192, 256, 3]) 
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                #Use saver.restore to load trained model
                print('Loading success')
            else:
                print('No checkpoint')
            prediction = sess.run(logits, feed_dict={x: image_arr})
            max_index = np.argmax(prediction) 
            print('Predicted label:')
            print(max_index)
            print('Predicted result:')
            print(prediction)
            print('This is a Health with possibility %.6f' % prediction[:, 0])
            print('This is a Diabetes with possibility %.6f' % prediction[:, 1])
            print('This is a Heart Disease with possibility %.6f' % prediction[:, 2])

            if max_index==0:
                print('This is a Health with possibility %.6f' % prediction[:, 0])
                pos=prediction[:,0]
            elif max_index == 1:
                print('This is a Diabetes with possibility %.6f' % prediction[:, 1])
                pos = prediction[:, 1]
            elif max_index == 2:
                print('This is a Heart Disease with possibility %.6f' % prediction[:, 2])
                pos = prediction[:, 2]
            else:
                print('ERROR')

    return max_index, pos


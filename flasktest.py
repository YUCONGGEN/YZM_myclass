from flask import Flask,Response,jsonify
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import cv2
import os
import random
import time
import csv
import sys
app = Flask(__name__)
#from  write import write
class JSONResponse(Response):

    @classmethod
    def force_type(cls, response, environ=None):
        '''
        这个方法只有视图函数返回非字符、非元祖、非Response对象才会调用
        :param response:是视图函数的返回值
        :param environ:
        :return:
        '''
        print(response)
        print(type(response))
        if isinstance(response,(list,dict)):

            #jsonify除了将字典转换成json对象，还将对象包装成了一个Response对象
            response = jsonify(response)

        return super(JSONResponse,cls).force_type(response,environ) #python 面向对象的一个知识点 super

@app.route("/<name>")
def yzm(name):
    print(name)
    ##############################################################################
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z']
    ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z']
    number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    char_set = number + alphabet + ALPHABET
    # 图像大小
    IMAGE_HEIGHT = 40
    IMAGE_WIDTH = 120
    MAX_CAPTCHA = 4

    CHAR_SET_LEN = len(char_set)
    model_path = './model_test3/'
    image_path = './test1/'

    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])

    Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
    keep_prob = tf.placeholder(tf.float32)  # dropout

    def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
        x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        # 随机初始化权重
        w_c1 = tf.get_variable(name='w_c1', shape=[3, 3, 1, 64], dtype=tf.float32,
                               initializer=tf.contrib.layers.xavier_initializer())
        # 偏置
        b_c1 = tf.Variable(b_alpha * tf.random.normal([64]))
        conv1 = tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        # 批标准化
        batch_mean, batch_var = tf.nn.moments(conv1, [0, 1, 2], keep_dims=True)
        shift = tf.Variable(tf.zeros([64]))
        scale = tf.Variable(tf.ones([64]))
        epsilon = 1e-3
        conv1 = tf.nn.batch_normalization(conv1, batch_mean, batch_var, shift, scale, epsilon)

        # 放在最大池化之后的relu
        conv1 = tf.nn.elu(conv1)
        conv1 = tf.nn.dropout(conv1, keep_prob)

        w_c2 = tf.get_variable(name='w_c2', shape=[3, 3, 64, 128], dtype=tf.float32,
                               initializer=tf.contrib.layers.xavier_initializer())
        b_c2 = tf.Variable(b_alpha * tf.random_normal([128]))
        conv2 = tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        # 批标准化
        batch_mean, batch_var = tf.nn.moments(conv2, [0, 1, 2], keep_dims=True)
        shift = tf.Variable(tf.zeros([128]))
        scale = tf.Variable(tf.ones([128]))
        epsilon = 1e-3
        conv2 = tf.nn.batch_normalization(conv2, batch_mean, batch_var, shift, scale, epsilon)

        conv2 = tf.nn.elu(conv2)
        conv2 = tf.nn.dropout(conv2, keep_prob)

        w_c3 = tf.get_variable(name='w_c3', shape=[3, 3, 128, 256], dtype=tf.float32,
                               initializer=tf.contrib.layers.xavier_initializer())
        b_c3 = tf.Variable(b_alpha * tf.random_normal([256]))
        conv3 = tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3)
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        # 批标准化 通过相当于一个图
        batch_mean, batch_var = tf.nn.moments(conv3, [0, 1, 2], keep_dims=True)
        shift = tf.Variable(tf.zeros([256]))
        scale = tf.Variable(tf.ones([256]))
        epsilon = 1e-3
        conv3 = tf.nn.batch_normalization(conv3, batch_mean, batch_var, shift, scale, epsilon)

        conv3 = tf.nn.elu(conv3)
        conv3 = tf.nn.dropout(conv3, keep_prob)

        w_c4 = tf.get_variable(name='w_c4', shape=[3, 3, 256, 256], dtype=tf.float32,
                               initializer=tf.contrib.layers.xavier_initializer())
        b_c4 = tf.Variable(b_alpha * tf.random_normal([256]))
        conv4 = tf.nn.bias_add(tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4)
        conv4 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        # 批标准化 通过相当于一个图
        batch_mean, batch_var = tf.nn.moments(conv3, [0, 1, 2], keep_dims=True)
        shift = tf.Variable(tf.zeros([256]))
        scale = tf.Variable(tf.ones([256]))
        epsilon = 1e-3
        conv4 = tf.nn.batch_normalization(conv4, batch_mean, batch_var, shift, scale, epsilon)

        conv4 = tf.nn.elu(conv4)
        conv4 = tf.nn.dropout(conv4, keep_prob)

        # 全连接层w=(w-f+p)/2+1算出来3x8x128=
        w_d = tf.Variable(w_alpha * tf.random_normal([3 * 8 * 256, 1024]))  # 4*10*64
        b_d = tf.Variable(w_alpha * tf.random_normal([1024]))
        dense = tf.reshape(conv4, [-1, w_d.get_shape().as_list()[0]])  # conv4最后链接
        dense = tf.nn.elu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, keep_prob)

        w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
        b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
        out = tf.add(tf.matmul(dense, w_out), b_out)
        return out

    # 向量转回文本
    def vec2text(vec):
        char_pos = vec.nonzero()[0]
        text = []
        for i, c in enumerate(char_pos):
            char_at_pos = i  # c/63
            char_idx = c % CHAR_SET_LEN
            if char_idx < 10:
                char_code = char_idx + ord('0')
            elif char_idx < 36:
                char_code = char_idx - 10 + ord('A')
            elif char_idx < 62:
                char_code = char_idx - 36 + ord('a')
            elif char_idx == 62:
                char_code = ord('_')
            else:
                raise ValueError('error')
            text.append(chr(char_code))
        return "".join(text)

    # def predict_captcha(captcha_image):
    #     output = crack_captcha_cnn()
    #
    #     saver = tf.train.Saver()
    #     with tf.Session() as sess:
    #         saver.restore(sess, tf.train.latest_checkpoint(model_path))
    #
    #         predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    #         text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})
    #
    #         text = text_list[0].tolist()
    #         vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    #         i = 0
    #         for n in text:
    #             vector[i * CHAR_SET_LEN + n] = 1
    #             i += 1
    #         return vec2text(vector)
    #
    # if not os.path.exists(image_path):
    #     print('Image does not exist, please check!, path:"{}"'.format(os.path.abspath(image_path)))
    #     sys.exit()
    image_list = os.listdir(image_path)

    output = crack_captcha_cnn()
    print(output)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        print(model_path)
        saver.restore(sess, tf.train.latest_checkpoint(model_path))

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        data = {}
        #print(image_list)62x4
        for image_ in image_list:
            image_=name+".jpg"

            #print()
            text_ = name
            #print(text_)
            image_p = os.path.join(image_path, image_)
            # 单张图片预测
            image = np.float32(cv2.imread(image_p, 0))
            image = image.flatten() / 255

            text_list = sess.run(predict, feed_dict={X: [image], keep_prob: 1})

            text = text_list[0].tolist()
            vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
            i = 0
            for n in text:
                vector[i * CHAR_SET_LEN + n] = 1
                i += 1
            predict_text = vec2text(vector)

            data['{}'.format(text_)] = "{}".format(predict_text)

            print("验证码: {0}.jpg  预测值: {1}".format(text_, predict_text))
            time.sleep(1)
            break
    return "验证码: {0}.jpg  预测值: {1}".format(text_, predict_text)
    #return name

if __name__ == '__main__':

    app.response_class = JSONResponse
    app.run(debug=True)

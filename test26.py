
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import cv2
import os
import random
from csv import reader
import time
yu={}
strp = list(reader(open('train_label.csv', encoding='UTF-8-sig')))
#total1=1
for k in strp:
      yu[ '{}'.format(k[0].split('.')[0])]="{}".format(k[0]+","+k[1] )
      #total1=total1+1
#print("字典",yu)
# number

alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
char_set=number+alphabet+ALPHABET
# 图像大小
IMAGE_HEIGHT = 40  # 80
IMAGE_WIDTH = 120  # 160
MAX_CAPTCHA = 4

mark1=mark2=mark3=mark4=mark5=0
CHAR_SET_LEN = len(char_set)  #字符总长度

image_filename_list = []#图片文档名字
total = 0

train_path = './train1/'

valid_path = './test1/'#有效的字符路径


def get_image_file_name(imgFilePath):#获得图片文件的名字
    fileName = []
    total = 0
    for filePath in os.listdir(imgFilePath):
        captcha_name = filePath.split('/')[-1]#以/分割保留后面
        fileName.append(captcha_name)
        total += 1
    #random.seed(time.time())
    # 打乱顺序
    #random.shuffle(fileName)
    #print(fileName, total)
    return fileName, total



# 获取训练数据的名称列表
image_filename_list, total = get_image_file_name(train_path)
# 获取测试数据的名称列表
image_filename_list_valid, total = get_image_file_name(valid_path)

num=0
# 读取图片和标签
def gen_captcha_text_and_image(imageFilePath, image_filename_list, imageAmount):
    #num = random.randint(0, imageAmount - 1)
    global num
    num1=num % imageAmount
    img = cv2.imread(os.path.join(imageFilePath, image_filename_list[num1]), 0)
   # img = cv2.resize(img, (160, 60))
    img = np.float32(img)
    text = image_filename_list[num1].split('.')[0]
    #print("修改前", text)

    text1 = yu['{}'.format(int(image_filename_list[num1].split('.')[0]))]
    text=text1.split(',')[-1]
    num=num+1
    #print(num1,imageAmount,image_filename_list,text, img)
    return text, img


# 文本转向量
# 例如，如果验证码是 ‘0296’ ，则对应的标签是
# [1 0 0 0 0 0 0 0 0 0
#  0 0 1 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 1
#  0 0 0 0 0 0 1 0 0 0]
def name2label(name):
    label = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    for i, c in enumerate(name):
        idx = i * CHAR_SET_LEN + ord(c) - ord('0')
        label[idx] = 1
    return label


# label to name
def label2name(digitalStr):
    digitalList = []
    for c in digitalStr:
        digitalList.append(ord(c) - ord('0'))
    return np.array(digitalList)


# 文本转向量
def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    def char2pos(c):
        # print(ord("0"))  # 48
        # print(ord("9"))  # 57
        # print(ord("A"))  # 65
        # print(ord("Z"))  # 90
        # print(ord("a"))  # 97
        # print(ord("z"))  # 122

        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    #print(vector)
    return vector


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


# 生成一个训练batch
def get_next_batch(imageFilePath, image_filename_list=None, batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    def wrap_gen_captcha_text_and_image(imageFilePath, imageAmount):
        while True:
            text, image = gen_captcha_text_and_image(imageFilePath, image_filename_list, imageAmount)
            if image.shape == (40, 120):
                return text, image

    for listNum in os.walk(imageFilePath):
        pass
    imageAmount = len(listNum[2])


    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image(imageFilePath, imageAmount)

        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)
        print(batch_x, batch_y)

    return batch_x, batch_y


####################################################################
# 占位符，X和Y分别是输入训练数据和其标签，标签转换成8*10的向量
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)

model = tf.keras.Sequential([   #顺序模型
        tf.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 1), activation='elu'),
        tf.keras.layers.Conv2D(64, (3, 3),  activation='elu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3),  activation='elu'),
        tf.keras.layers.Conv2D(64, (3, 3),  activation='elu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3),  activation='elu'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='elu'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='elu'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='elu'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='elu'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='elu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(512, (3, 3), activation='elu'),
        tf.keras.layers.Conv2D(512, (3, 3), activation='elu'),
        tf.keras.layers.Conv2D(512, (3, 3), activation='elu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='elu'),
        tf.keras.layers.Dense(248, activation='softmax'),
        tf.keras.layers.Dropout(rate=0.6)],name="ycg15")
print(model.summary())

def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    #随机初始化权重
    w_c1 = tf.get_variable(name='w_c1', shape=[3, 3, 1, 64], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
    #偏置
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

    w_c2 = tf.get_variable(name='w_c2', shape=[3,3, 64, 128], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
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

    w_c3 = tf.get_variable(name='w_c3', shape=[3, 3, 128, 256], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
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
    dense = tf.reshape(conv4,[-1, w_d.get_shape().as_list()[0]])  # conv4最后链接
    dense = tf.nn.elu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out


# 训练
def train_crack_captcha_cnn():
    global mark1
    global mark2
    global mark3
    global mark4

    mark1 = 1
    mark2 =1
    mark3 = 1
    mark4 = 1
    output = crack_captcha_cnn()
    # loss
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    # tf.nn.sigmoid_cross_entropy_with_logits()函数计算交叉熵,输出的是一个向量而不是数;
    # 交叉熵刻画的是实际输出（概率）与期望输出（概率）的距离，也就是交叉熵的值越小，两个概率分布就越接近
    # tf.reduce_mean()函数求矩阵的均值
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    val_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    tf.summary.scalar('loss', loss)  # 1
    tf.summary.scalar('loss', val_loss)  # 1
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢减小
    # tf.train.AdamOptimizer（）函数实现了Adam算法的优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    val_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('val_accuracy', val_accuracy)  # 4
    tf.summary.scalar('accuracy', accuracy)  # 2
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # batch_x, batch_y = get_next_batch(train_path, image_filename_list, 16)
        # _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
        # acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
        #
        # batch_x_test, batch_y_test = get_next_batch(valid_path, image_filename_list_valid, 128)
        # val_acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
        # _, val_loss = sess.run([optimizer, loss], feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})

        # global loss_
        # loss_ = 1
        # global acc
        # acc=0
        # global val_loss
        # val_loss=1
        # global val_acc
        # val_acc=0

        #
        # tf.summary.scalar('val_loss', val_loss)  # 3
        # tf.summary.scalar('val_accuracy', val_acc)  # 4

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('logs3/', tf.get_default_graph())
        step = 0
        while True :
            batch_x, batch_y = get_next_batch(train_path, image_filename_list, 16)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob:1})
            acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, keep_prob: 1})
            #print(step,"训练集loss:"+str(loss_),"训练集acc:"+str(acc))

            if step % 1 == 0  :



                #验证
                batch_x_test, batch_y_test = get_next_batch(valid_path, image_filename_list_valid, 128)
                val_acc = sess.run(val_accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                _, val_loss_ = sess.run([optimizer, val_loss], feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})

                merged = tf.summary.merge_all()  # 5
                #tensor_step=tf.Variable(step)
                summary, _ = sess.run([merged, optimizer], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.})#7
                writer.add_summary(summary, step)#8
                print(step,"训练集loss:"+str(loss_),"训练集acc:"+str(acc),"验证集acc"+str(val_acc),"验证集loss"+str(val_loss_))

                if  val_acc >0.90  and mark1==1:
                      mark1=0
                      saver.save(sess, "./model_test3/8crack_capcha.model", global_step=step)
                if  val_acc >0.95 and mark2==1 :
                      mark2 = 0
                      saver.save(sess, "./model_test3/8crack_capcha.model", global_step=step)
                if  val_acc >0.96 and mark3==1 :
                      mark3 = 0
                      saver.save(sess, "./model_test3/8crack_capcha.model", global_step=step)
                if  val_acc >0.97 and mark4==1:
                      mark4 = 0
                      saver.save(sess, "./model_test3/8crack_capcha.model", global_step=step)
                if  val_acc >0.98 or step > 1000:
                              saver.save(sess, "./model_test3/8ucrack_capcha.model", global_step=step)
                              writer.close()
                              break
            step += 1


def predict_captcha(captcha_image):
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})

        text = text_list[0].tolist()
        vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
        i = 0
        for n in text:
            vector[i * CHAR_SET_LEN + n] = 1
            i += 1
        return vec2text(vector)


# 执行训练
train_crack_captcha_cnn()
#opt=crack_captcha_cnn()
#print(opt)
print("训练完成，请开始测试…")

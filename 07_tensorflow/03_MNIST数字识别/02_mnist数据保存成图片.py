# coding:utf8

from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import os

# 读取数据集
mnist = input_data.read_data_sets("./data/mnist", one_hot=True)

# 保存图片路径， 如果没有就自动创建
save_dir = "./data/raw/"
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

# 保存前20张图片
for i in range(20):
    image_array = mnist.train.images[i, :]
    # mnist中图片是784维度， 把他改成28*28维度
    image_array = image_array.reshape(28, 28)
    # 保存文件
    filename = save_dir + "mnist_train_%d.jpg" % i
    scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)

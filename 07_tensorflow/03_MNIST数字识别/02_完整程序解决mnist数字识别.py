import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST相关常数
INPUT_NODE = 784    # 输入层节点数, 等于图片的像素数
OUTPUT_NODE = 10    # 输出层节点书, 也就是类别数量

# 配置神经网络参数
LAYER1_NODE = 500   # 隐藏层节点个数, 这里只有一个隐藏层的神经网络作为样例
BATCH_SIZE = 100    # 一个训练batch中数据个数,越小越接近随即梯度下降,越大越接近梯度下降

# 随机梯度下降
LEARNING_RATE_BASE = 0.8    # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习衰减率

REGULARIZATION_RATE = 0.0001 # 正则化在损失函数中的系数
TRAINING_STEPS = 30000      # 训练轮数
MOVING_AVERAGE_DECAY = 0.99 # 滑动平均衰减率


# 一个辅助函数 用于计算当前神经网络的y前向传播的过程
# 使用reLu激活的三层全链接网络 有一个隐藏层
def inference(input_tensor, avg_class, weights1, bias1, weights2, bias2):
    # 没有滑动平均类的时候,不使用滑动平均
    if avg_class is None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + bias1)
        return tf.matmul(layer1, weights2) + bias2
    else:
        # 首先计算滑动平均,然后再计算前向传播
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(bias1))
        return tf.matmul(layer1, avg_class.average(weights2) + avg_class.average(bias2))


# 训练模型的过程
def train(mnist):
    # 输入的特征和真实值 利用占位符
    x = tf.placeholder(tf.float32, shape=[None, INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_NODE], name="y-input")

    # 生成隐藏层的参数   标准差0.1
    weights1 = tf.Variable(tf.truncated_normal(shape=[INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal(shape=[LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算当前参数神经网络的前向传播 这里用于计算滑动平均的类为None
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义训练轮数变量,这个变量不能被训练
    global_step = tf.Variable(0, trainable=False)

    # 定义滑动平均类 给定平滑衰减和训练轮数,达到下降速度逐渐减慢的效果
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 将神经元上的所有变量使用滑动平均
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # 滑动平均不会改变原变量,会维护一个新的影子变量
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵 刻画预测值和真实值的差距
    # 多个分类有一个是正确值的时候,采用这个函数计算
    # argmax函数用于得到正确答案对应类别的编号
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 计算当前batch中所有样例的交叉熵的平均
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算l2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE,)
    # 计算正则化后的损失
    regularization = regularizer(weights1) + regularizer(weights2)
    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean = regularization
    # 设置指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, # 基础学习率
        global_step,    # 当前迭代的轮数
        mnist.train.num_examples / BATCH_SIZE,  # 过完所有训练数据需要的迭代次数
        LEARNING_RATE_DECAY,    # 学习率衰减速度
    )

    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 每一步训练,都要反向传播更新参数 和 更新滑动平均值
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

    # average_y 是一个batch_size*10的二位数组 每一行是一个结果
    # argmax 是在第一个维度选取最大值的下标
    # equal 比较是否相同 返回bool
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    # 平均值计算在这个BATCH上的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 开启会话训练
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels,
        }
        test_feed = {
            x: mnist.test.images,
            y_: mnist.validation.labels
        }

        # 迭代训练
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("经过%i迭代后 准确率:%s" % (i, validate_acc))
            # 产生这一轮使用的batch数据 运行训练过程
            xs, ys = mnist.train.naxt_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 训练后的最终结果
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("训练的最终结果:%s" % test_acc)




# 程序的入口
def main(argv=None):
    # 此处声明MNIST数据集 如果有就读取否则下载
    mnist = input_data.read_data_sets("../../data/mnist")
    train(mnist)

if __name__ == '__main__':
    tf.app.run()

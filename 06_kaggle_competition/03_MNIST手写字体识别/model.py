import pandas as pd
# import tensorflow as tf
import skflow

# 读取训练数据
train = pd.read_csv("./data/train.csv")
# print(train.shape)  # (42000, 785)
test = pd.read_csv("./data/test.csv")
# print(test.shape)   # (28000, 784)

y_train = train["label"]
x_train = train.drop("label", 1)
x_test = test

# skflow中封装好的tensorflow分类器 进行学习
classifier_linear = skflow.TensorFlowLinearClassifier(n_classes=10,
                                               batch_size=100,
                                               steps=1000,
                                               learning_rate=0.01)
classifier_linear.fit(x_train, y_train)
linear_y_predict = classifier_linear.predict(x_test)

linear_submission = pd.DataFrame({
    "ImageId": range(1, 28001),
    "Label": linear_y_predict
})
linear_submission.to_csv("./predict/linear_submission.csv", index=False)

# 使用skflow封装好的tensorflow的深度神经网络模型进行学习预测
classifier_DNN = skflow.TensorFlowDNNClassifier(hidden_units=[200, 50, 10],
                                                n_classes=10,
                                                steps=5000,
                                                learning_rate=0.01,
                                                batch_size=50)
classifier_DNN.fit(x_train, y_train)
dnn_y_predict = classifier_DNN.predict(x_test)
dnn_submission = pd.DataFrame({
    "ImageId": range(1, 28001),
    "Label": dnn_y_predict
})
dnn_submission.to_csv("./predict/dnn_submission.csv", index=False)
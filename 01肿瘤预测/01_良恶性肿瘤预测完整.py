import numpy as np
import pandas as pd

# 数据预处理部分

# 创建特征列表
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
                'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell size',
                'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
# 使用pandas.read_csv从互联网读取数据集
data = pd.read_csv('./data/breast-cancer-wisconsin.data', names=column_names)
# 将?替换为标准缺失值表示
data = data.replace(to_replace='?', value=np.nan)
# 丢失带有缺失值的数据 只要有一个维度有缺失就丢弃
data = data.dropna(how='any')
# 输出data数据的数量和维度
# print(data.shape)



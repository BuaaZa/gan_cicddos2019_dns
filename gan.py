import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# import matplotlib
# import seaborn as sns
# import sklearn
# import imblearn
# import matplotlib.pyplot as plt
# import time
# import sklearn.metrics as m
import gc
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pyarrow as pa
import pyarrow.parquet as pq

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import csv
import random

# Ignore warnings
warnings.filterwarnings('ignore')

parquet_file_path = 'data/DrDoS_DNS.parquet'

def load_data():
    df_from_parquet = pq.read_table(parquet_file_path).to_pandas()
    print(df_from_parquet.shape)
    print(df_from_parquet.head())
    return df_from_parquet


print("---Start loading data---")
df = load_data()
print("---End loading data---")
print(df.iloc[:, -1:].value_counts())  # 统计最后列Label各个不同值的出现次数

train, test = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)  # 打乱后分割训练集和测试集 train 0.7 test 0.3

train_dns_cpy = (train[train['Label'] == 'DrDoS_DNS']).copy()
test_dns_cpy = (test[test['Label'] == 'DrDoS_DNS']).copy()
df_dns_train = train_dns_cpy
df_dns_test = test_dns_cpy
print("---shape of train and test data---")
print(df_dns_train.shape)
print(df_dns_test.shape)

# Saving .csv files for samples
train_csv_path = 'result/DrDoS_DNS_Train.csv'
test_csv_path = 'result/DrDoS_DNS_Test.csv'
df_dns_train = df_dns_train.iloc[:, :-1].dropna(axis=1, how='all')  # 选取除最后一列外的所有列, 删除所有完全由NA/NaN值组成的列
df_dns_test = df_dns_test.iloc[:, :-1].dropna(axis=1, how='all')
df_dns_train.to_csv(train_csv_path, index=False)  # 默认使用 UTF-8 编码格式来保存CSV文件
df_dns_test.to_csv(test_csv_path, index=False)  # 默认使用 UTF-8 编码格式来保存CSV文件
print(df_dns_train.shape)
print(df_dns_test.shape)

# GAN Model
print("---Start training GAN model---")
header = pd.read_csv(train_csv_path, header=None, encoding='utf-8')  # 不将CSV文件的第一行用作列名，生成默认的整数索引作为列名
# 深拷贝
df = header.copy() # 列名是第一行 without label
df_gen = df.iloc[0:1]
print("---show train data---")
print(df.shape)
print(df.head())
print(df_gen.shape)
print(df_gen.head())

col_count = df.shape[1]

def save_data2csv(output_data):
    # add row to end of DataFrame
    df.loc[len(df.index)] = output_data.numpy()[0]
    df_gen.loc[len(df_gen.index)] = output_data.numpy()[0]

data = []
with open(train_csv_path, 'r') as f:
    reader = csv.reader(f)
    data = list(reader) # 二维列表，第一行是列名

# 从row1开始读，去掉header
def get_dataset():
    row_index = random.randint(1, len(data) - 1)

    re_float = [float(i) for i in data[row_index][:]]
    return torch.Tensor([re_float])


# generating noise for generator 生成一个形状为(1, col_num)的全零PyTorch张量
def make_noise(col_num):
    # return torch.zeros((1, 68))
    return torch.Tensor(np.random.uniform(0, 0, (1, col_num)))

class generator(nn.Module):
    def __init__(self, inp, out):
        super(generator, self).__init__()
        # 3层感知器
        self.net = nn.Sequential(nn.Linear(inp, 300),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(300, 300),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(300, out)
                                 )

    def forward(self, x):
        x = self.net(x)
        return x

class discriminator(nn.Module):
    def __init__(self, inp, out):
        super(discriminator, self).__init__()
        self.net = nn.Sequential(nn.Linear(inp, 300),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(300, 300),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(300, out),
                                 nn.Sigmoid()
                                 )

    def forward(self, x):
        x = self.net(x)
        return x

# 全零张量初始化模型
# Generator
gen = generator(col_count, col_count)
x1 = gen(make_noise(col_count))
# Discriminator
discrim = discriminator(col_count, 1)
x2 = discrim(x1)

step_num = 20
epochs = 4
gen_num = step_num * epochs
# d_step = 10
# g_step = 8
d_step = 2
g_step = 2

# 二元交叉熵损失Binary Cross Entropy Loss
criteriond1 = nn.BCELoss()
optimizerd1 = optim.SGD(discrim.parameters(), lr=0.001, momentum=0.9)

criteriond2 = nn.BCELoss()
# optimizerd2 = optim.SGD(gen.parameters(), lr=0.001, momentum=0.9)
# 使用 Adam 优化器替换 SGD 优化器
optimizerd2 = optim.Adam(gen.parameters(), lr=0.001, betas=(0.5, 0.999))

# 训练 GAN
for nsteps in range(step_num):
    if (nsteps % 100 == 0):
        print("Step:", nsteps)
    for epoch in range(epochs):
        # training discriminator
        # Train D on real + fake
        for d_i in range(d_step):
            #         print("--------------discriminator-----------------")
            discrim.zero_grad()

            # real
            data_d_real = Variable(get_dataset())  # real data
            data_d_real_pred = discrim(data_d_real)  # real decision
            data_d_real_loss = criteriond1(data_d_real_pred, Variable(torch.ones(1, 1)))  # ones = true
            data_d_real_loss.backward()  # compute/store gradients（梯度）, but don't change params

            print("data_d_real_decision: ", data_d_real_pred)
            # print("data_d_real_loss: ", data_d_real_loss)

            # fake
            data_d_noise = Variable(make_noise(col_count))
            data_d_gen_out = gen(data_d_noise).detach()  # fake data, detach to avoid training G on these labels
            data_fake_dicrim_out = discrim(data_d_gen_out)  # fake decision
            data_fake_d_loss = criteriond1(data_fake_dicrim_out, Variable(torch.zeros(1, 1)))  # zeros = fake
            data_fake_d_loss.backward()  # Only optimizes D's parameters; changes based on stored gradients from backward()
            print("data_d_fake_decision: ", data_fake_dicrim_out)
            # print("data_fake_d_loss: ", data_fake_d_loss)

            # 执行优化器的step方法，利用之前累积的梯度（包括真实数据和假数据的损失贡献）来更新判别器的权重
            optimizerd1.step()

        # training generator
        # Train G on D's response (but DO NOT train D on these labels)
        for g_i in range(g_step):
            #         print("--------------generator-----------------")
            gen.zero_grad()

            data_noise_gen = Variable(make_noise(col_count))  # gen input
            data_g_gen_out = gen(data_noise_gen)  # fake data
            data_g_dis_out = discrim(data_g_gen_out)  # fake decision
            data_g_loss = criteriond2(data_g_dis_out,
                                      Variable(torch.ones(1, 1)))  # we want to fool, so pretend it's all genuine
            data_g_loss.backward()

            #         print("data_noise_gen: ", data_noise_gen)
            #         print("data_g_gen_out: ", data_g_gen_out)
            print("data_g_fake_decision: ", data_g_dis_out)
            # print("data_g_loss: ", data_g_loss)
            #         print("\n")

            optimizerd2.step()  # Only optimizes G's parameters

        save_data2csv(data_d_gen_out)

# 删除第一行表头
df_dns_mix = df.drop(index=0).dropna() # train + gen without label
df_dns_gen = df_gen.drop(index=0).dropna() # only gen without label
print("---mix data---")
print(df_dns_mix.shape)
print(df_dns_mix.head())
print("---show gen data---")
print(df_dns_gen.shape)
print(df_dns_gen.head())
print(df_dns_gen.dtypes)
# label
df_dns_label = train_dns_cpy.iloc[1, -1:]
df_dns_mix_labels = pd.concat([df_dns_label] * (gen_num + len(train_dns_cpy)))
df_dns_gen_labels = pd.concat([df_dns_label] * gen_num)

# 去除行索引
df_dns_mix_labels = df_dns_mix_labels.reset_index(drop=True)
df_dns_gen_labels = df_dns_gen_labels.reset_index(drop=True)
# df补充label
df_mix_dns_label = pd.concat([df_dns_mix, df_dns_mix_labels], axis=1)
df_gen_dns_label = pd.concat([df_dns_gen, df_dns_gen_labels], axis=1)


gen_weight_path = 'result/generator_weights.pth'
discrim_weight_path = 'result/discriminator_weights.pth'
# 保存模型权重
torch.save(gen.state_dict(), gen_weight_path)
torch.save(discrim.state_dict(), discrim_weight_path)

# 保存生成数据
gen_csv_path = 'result/DrDoS_DNS_Gen.csv'
df_dns_gen.to_csv(gen_csv_path, index=False)

# 不知道为什么要分类？？？

# from sklearn import metrics
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
#
# labels = ['BENIGN', 'DrDoS_DNS']
# avg_accuracy = 0.0
#
# def labeller(x, match):
#     if x == match:
#         return 1
#     else:
#         return 0
#
# tmp_df = df.copy()
# for i in labels:
#     tmp_df['Label'] = df['Label'].apply(lambda x: labeller(x, i))
#     train, test = train_test_split(tmp_df, test_size=0.2, random_state=101)
#
#     # 数值型特征进行标准化
#     scaler = StandardScaler()
#     cols = train.select_dtypes(include=['float64', 'int64']).columns
#     sc_train = scaler.fit_transform(train.select_dtypes(include=['float64', 'int64']))
#     sc_test = scaler.fit_transform(test.select_dtypes(include=['float64', 'int64']))
#
#     sc_train_df = pd.DataFrame(sc_train, columns=cols)
#     sc_test_df = pd.DataFrame(sc_test, columns=cols)
#
#     train_X = sc_train_df
#     train_y = train['Label']
#
#     test_X = sc_test_df
#     test_y = test['Label']
#
#     X_train, X_test, Y_train, Y_test = train_test_split(train_X, train_y, train_size=0.80, random_state=101)
#
#     KNN_Classifier = KNeighborsClassifier(n_jobs=-1)
#     KNN_Classifier.fit(X_train, Y_train)
#     accuracy = metrics.accuracy_score(Y_test, KNN_Classifier.predict(X_test))
#     avg_accuracy += accuracy
#
# avg_accuracy /= len(labels)
# print("Avg. accuracy", avg_accuracy)


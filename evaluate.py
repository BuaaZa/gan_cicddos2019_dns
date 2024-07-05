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
from sklearn.metrics.pairwise import rbf_kernel
import pyarrow as pa
import pyarrow.parquet as pq

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
import torch.optim as optim
from torch.autograd import Variable
import csv
import random

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

train_csv_path = 'result/DrDoS_DNS_Train.csv'
test_csv_path = 'result/DrDoS_DNS_Test.csv'
gen_csv_path = 'result/DrDoS_DNS_Gen.csv'

gen_weight_path = 'result/generator_weights.pth'
discrim_weight_path = 'result/discriminator_weights.pth'

# 测试集上 测试 GAN 判别器的准确性
train_data = pd.read_csv(train_csv_path, encoding='utf-8')
real_data_df = pd.read_csv(test_csv_path, encoding='utf-8') # 测试集的真实数据 without label
generated_data_df = pd.read_csv(gen_csv_path, encoding='utf-8') # 生成器生成的数据 without label

print(train_data.shape)
print(real_data_df.shape)
print(real_data_df.dtypes)
print(real_data_df.head())
print(generated_data_df.shape)
print(generated_data_df.dtypes)
print(generated_data_df.head())

input_dim = real_data_df.shape[1]
output_dim = 1

generator = generator(input_dim, input_dim)
discrim = discriminator(input_dim, output_dim)
generator.load_state_dict(torch.load(gen_weight_path))
discrim.load_state_dict(torch.load(discrim_weight_path))

def predict(discriminator, data):
    # 确保数据为浮点类型
    data = data.astype(np.float32)

    discriminator.eval()  # 切换到评估模式
    with torch.no_grad():  # 禁用梯度计算
        inputs = torch.tensor(data.values, dtype=torch.float32)
        outputs = discriminator(inputs)
        predictions = (outputs > 0.5).numpy().astype(int)  # 将预测值转换为二进制分类结果
    return predictions

real_predictions = predict(discrim, real_data_df)
generated_predictions = predict(discrim, generated_data_df)

real_labels = np.ones(len(real_predictions))
generated_labels = np.zeros(len(generated_predictions))

real_accuracy = accuracy_score(real_labels, real_predictions)
generated_accuracy = accuracy_score(generated_labels, generated_predictions)

# 打印结果
print(f"判别器在真实数据上的准确性: {real_accuracy:.2f}")
print(f"判别器在生成数据上的准确性: {generated_accuracy:.2f}")

# 分析生成数据的质量

def dataframe_to_tensor(dataframe):
    # 确保数据为浮点类型
    dataframe = dataframe.astype(np.float32)
    return torch.tensor(dataframe.values, dtype=torch.float32)

def calculate_wasserstein_distance(real_scores, generated_scores):
    real_scores_np = real_scores.cpu().detach().numpy().flatten()
    generated_scores_np = generated_scores.cpu().detach().numpy().flatten()
    return wasserstein_distance(real_scores_np, generated_scores_np)

real_data = dataframe_to_tensor(real_data_df)
generated_data = dataframe_to_tensor(generated_data_df)

real_discrims = discrim(real_data)
generated_discrims = discrim(generated_data)

wasserstein_dist = calculate_wasserstein_distance(real_discrims, generated_discrims)
print(f"Wasserstein Distance: {wasserstein_dist}")

def gaussian_kernel_matrix(x, y, sigmas):
    beta = 1.0 / (2.0 * (torch.unsqueeze(sigmas, 1)))
    dist = torch.cdist(x, y, p=2.0)
    s = torch.matmul(beta, dist.view(1, -1))
    return torch.sum(torch.exp(-s), 0)

def mmd(x, y, batch_size=1000, kernel=rbf_kernel):
    n_x = len(x)
    n_y = len(y)
    mmd_value = 0.0

    x_batches = [x[i:i + batch_size] for i in range(0, n_x, batch_size)]
    y_batches = [y[i:i + batch_size] for i in range(0, n_y, batch_size)]

    for x_batch in x_batches:
        for y_batch in y_batches:
            K_XX = kernel(x_batch, x_batch)
            K_YY = kernel(y_batch, y_batch)
            K_XY = kernel(x_batch, y_batch)

            # 求均值并加权
            mmd_value += (np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY))

    mmd_value /= (len(x_batches) * len(y_batches))
    mmd_value /= (n_x * n_y)

    return mmd_value

real_data_np = real_data.numpy()
generated_data_np = generated_data.numpy()
# 计算MMD
mmd_result = mmd(real_data_np, generated_data_np, batch_size=1000)
print("MMD:", mmd_result)
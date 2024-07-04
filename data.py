import warnings

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Ignore warnings
warnings.filterwarnings('ignore')

csv_file_path = 'data/DrDoS_DNS.csv'

def load_data():
    # Settings
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    # Load Data
    df = pd.read_csv(csv_file_path, index_col=0, usecols=lambda x: x != 'Unnamed: 0')
    print(df.shape)
    print(df.head())
    return df

print("---Start loading data---")
df = load_data()

# 打乱DataFrame
shuffled_df = df.sample(frac=1).reset_index(drop=True)
# 计算前十分之一的行数
n = len(shuffled_df) // 10

# 获取前十分之一的行
df = shuffled_df.head(n)
print(df.shape)

# 删除列, axis=1指定要沿列方向操作,inplace=True表示修改应直接应用于原DataFrame而不是返回一个新的
drop_columns = [ # this list includes all spellings across CIC NIDS datasets
    "Flow ID",
    'Fwd Header Length.1',
    "Source IP", "Src IP",
    "Source Port", "Src Port",
    "Destination IP", "Dst IP",
    "Destination Port", "Dst Port",
    "Timestamp",
    "Unnamed: 0", "Inbound", "SimillarHTTP" # CIC-DDoS other undocumented columns
]
df.columns = df.columns.str.strip() # sometimes there's leading / trailing whitespace
df.drop(columns=drop_columns, inplace=True, errors='ignore')

print(df.dtypes)
print(df.shape)

# 删除数据错误的行
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

df[['Flow Bytes/s', 'Flow Packets/s']] = df[['Flow Bytes/s', 'Flow Packets/s']].apply(pd.to_numeric)
# 取df中的数值列
df_num = df.select_dtypes(include='number')
# df=(df.iloc[:,:-1]-df.iloc[:,:-1].min())/(df.iloc[:,:-1].max()-df.iloc[:,:-1].min())
# 对df_num中的每一列进行了z - score标准化，计算了每列的平均值和范围,应用了标准化公式
df_norm = (df_num - df_num.mean()) / (df_num.max() - df_num.min())
# 赋值回原始df
df[df_norm.columns] = df_norm

print(df.shape)

parquet_file_path = 'data/DrDoS_DNS.parquet'
df.to_parquet(parquet_file_path, engine='pyarrow')
print("---End processing data---")

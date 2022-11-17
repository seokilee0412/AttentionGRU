from marcap import marcap_data
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

#초기 모델이 lookback은 논문에서의 20일을 따름, 종가(Close)를 비교하여 라벨링
def peak_detection(dataset): #Peak Detection

  max = - 100000000
  min =   100000000
  peak_list = set()
  trough_list = set()
  cr = -1
  pr = -1

  for i in range(len(dataset)):
    if dataset.loc[i,'Close'] > max:
      max = dataset.loc[i,'Close']
      max_idx = i
    if dataset.loc[i,'Close'] < min:
      min = dataset.loc[i,'Close']
      min_idx = i

    if dataset.loc[i,'Close'] <= max:
      count = 1
      for j in range(1,20):
        if i+j >= len(dataset):
          break
        if dataset.loc[i+j,'Close'] <= max:
          count +=1
      if count == 20:
        peak_list.add(max_idx)
        cr = max_idx + 20
    if i == cr:
      # print(i)
      max = - 100000000000

    elif dataset.loc[i,'Close'] >= min:
      count = 1
      for j in range(1,20):
        if i+j >= len(dataset):
          break
        if dataset.loc[i+j,'Close'] >= min:
          count += 1
      if count == 20:
        trough_list.add(min_idx)
        pr = min_idx + 20
        # print(pr)
    if i == pr:
      # print(i) 
      min = 100000000000000
  return peak_list, trough_list


   #극점 전후 1일 극점으로 라벨링, 극대 1, 극소 -1, 비극점 0
 #극점에서 6일 이상 떨어진 것들 비극점
def y_labeling(dataset,peak_list,trough_list):
  del_list = set()
  for i in range(len(dataset)):
    if i in peak_list:
      del_list.update([i-2,i-3,i-4,i-5,i-6]) #극점도 비극점에도 쓰이지 않는 데이터
      dataset.loc[i,'label'] = 1
      if i == 0:
        dataset.loc[i+1,'label'] = 1
      elif i == len(dataset):
        dataset.loc[i-1,'label'] = 1
      else:
        dataset.loc[i-1,'label'] = 1
        dataset.loc[i+1,'label'] = 1

    elif i in trough_list:
      del_list.update([i-2,i-3,i-4,i-5,i-6])
      dataset.loc[i,'label'] = -1
      if i == 0:
        dataset.loc[i+1,'label'] = 2
      elif i == len(dataset):
        dataset.loc[i-1,'label'] = 2
      else:
        dataset.loc[i-1,'label'] = 2
        dataset.loc[i+1,'label'] = 2

  del_list = del_list - set(peak_list)
  del_list = del_list - set(trough_list)
  del_list =list(del_list)
  length_data = len(dataset)
  
  # print(del_list)
  true_del_list = []
  for i in range(len(del_list)):
    if del_list[i] >= 0 and del_list[i] <= length_data:
      true_del_list.append(del_list[i])
  # print(true_del_list)
  for i in range(len(dataset)):
    if i not in true_del_list and pd.isna(dataset.loc[i,'label']):
      dataset.loc[i,'label'] = 0
  # print(true_del_list)
  return dataset

#RSI
def cal_RSI(dataset):
  dataset['RSI'] = 0
  for i in range(len(dataset)):
    if i >=13:
      minus_change = 0
      plus_change = 0
      for j in range(0,14):
        if dataset.loc[i-j,"Changes"] < 0:
          minus_change += dataset.loc[i-j,"Changes"]
        elif dataset.loc[i-j,"Changes"] > 0:
          plus_change += dataset.loc[i-j,"Changes"]
      sum = abs(minus_change) + plus_change
      dataset.loc[i,"RSI"] = plus_change * 100 / sum
  return dataset

  #OBV
def cal_OBV(dataset):
  dataset['OBV'] = 0
  for i in range(len(dataset)):
    if i > 0:
      if dataset.loc[i,'Close'] > dataset.loc[i-1,'Close']:
        dataset.loc[i,'OBV'] = dataset.loc[i-1,'OBV'] + dataset.loc[i,'Volume']
      elif dataset.loc[i,'Close'] < dataset.loc[i-1,'Close']:
        dataset.loc[i,'OBV'] = dataset.loc[i-1,'OBV'] - dataset.loc[i,'Volume']
      else:
        dataset.loc[i,'OBV'] = dataset.loc[i-1,'OBV']
  return dataset

def extracting_(dataset,split=False):
  dataset.rename(columns={'Close':'Price'}, inplace=True)
  dataset_x = dataset.loc[:,['Price','RSI','OBV']]
  dataset_y = dataset.loc[:,'label']
  data = pd.concat([dataset_x,dataset_y],axis=1)
  if split:
    return dataset_x, dataset_y
  return data

def dataset_for_model(dataset,x_features=3,batch_size= 16,split=False,test_size=300):
  dataset = extracting_(dataset)
  x = []
  y = []
  for i in range(len(dataset)):
    if not pd.isna(dataset.loc[i,'label']):
      if i >=13:
        data = dataset.iloc[i-4:i+1,:]
        data_x = data.loc[:,data.columns != 'label']
        data_y = data.loc[:,'label']
        x.append(np.array(data_x))
        y.append(np.array(data_y)[4])

  x = np.array(x)
  x = torch.from_numpy(x)
  x_rows = x.shape[0]
  y = np.array(y)
  y = torch.from_numpy(y)

  #scaling x
  scaler = MinMaxScaler()
  x = scaler.fit_transform(x)
  x = torch.from_numpy(x)
  x = torch.reshape(x,(x_rows,-1,x_features))

  #one-hot and argmax
  list_y = []
  for i in range(len(y)):
    z = np.zeros(shape=3,dtype=int)
    if y[i] == 0:
      z[0] = 1
    elif y[i] == 1:
      z[1] = 1
    elif y[i] == -1:
      z[2] = 1
    list_y.append(z)
  list_y = np.array(list_y)
  y = torch.from_numpy(list_y)
  y = torch.argmax(y,dim=1)

  #dataset for training
  if split:
    train_data = TensorDataset(x[:x.size(0)-test_size], y[:x.size(0)-test_size])
    test_data = TensorDataset(x[x.size(0)-test_size:], y[x.size(0)-test_size:])

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True)
    return train_loader, test_loader
  else:
    train_data = TensorDataset(x[:100000], y[:100000])
    train_loader = DataLoader(train_data, shuffle=True, batch_size=1, drop_last=True)
    return train_loader

def make_dataloader(batch_size):
    # 삼성전자 주식은 2018년 5월 3일 이후로 액면분할함
    df_2020_samsung = marcap_data('2000-01-01', '2018-05-03', code='005930')
    df_2020_samsung['label'] = np.nan
    df_2020_samsung.reset_index(drop=True,inplace=True)
    peak_list, trough_list = peak_detection(df_2020_samsung)
    df_2020_samsung = y_labeling(df_2020_samsung,peak_list,trough_list)
    df_2020_samsung = cal_RSI(df_2020_samsung)
    df_2020_samsung = cal_OBV(df_2020_samsung)
    train_loader,test_loader = dataset_for_model(df_2020_samsung, batch_size = batch_size, split=True)
    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = make_dataloader()


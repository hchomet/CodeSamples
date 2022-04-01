#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import plotly.express as px #alternative charting function
import plotly.io as pio #graphs do not work in Spyder
pio.renderers.default='browser'
import openpyxl
#%matplotlib inline
from datetime import date

#Reading dataset
data_set = pd.read_csv('Data/Portfolio_data2.csv',engine = 'python')
data_set=data_set[data_set['GroupByCriteriaValues']=='AMERICAN DOLLAR']
data_set=data_set[data_set['Indicator']=='Asset Fx Delta']

data_set=pd.melt(data_set, id_vars=['Date'], value_vars=['Result'])
#Data frame
df = data_set['value']
#Preprocessing data set
df = np.array(df).reshape(-1,1)

'''df['dt']=pd.to_datetime(data_set['Date'],format='%d/%m/%Y')
df['week'] = data_set['dt'].dt.week 
df['day'] = data_set['dt'].dt.day
df['month'] = data_set['dt'].dt.month
data_set['dayofweek'] = data_set['dt'].dt.dayofweek'''

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df = scaler.fit_transform(df)


#Training and test sets
train = df[:260]
test = df[260:]

print(train.shape)
print(test.shape)

def get_data(data, look_back):
  datax, datay = [],[]
  for i in range(len(data)-look_back-1):
    datax.append(data[i:(i+look_back),0])
    datay.append(data[i+look_back,0])
  return np.array(datax) , np.array(datay)

look_back = 1

x_train , y_train = get_data(train, look_back)
print(x_train.shape)
print(y_train.shape)

x_test , y_test = get_data(test,look_back)
print(x_test.shape)
print(y_test.shape)

#Processing train and test sets for LSTM model
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], 1)
print(x_train.shape)
print(x_test.shape)

#Defining the LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

n_features=x_train.shape[1]
model=Sequential()
model.add(LSTM(100,activation='relu',input_shape=(n_features,1)))
model.add(Dense(1))

print(model.summary())

#Compiling
model.compile(optimizer='adam', loss = 'mse')

#Training
model.fit(x_train,y_train, epochs = 20, batch_size=1)

y_pred = model.predict(x_test)

#Visualizing the results
plt.figure(figsize=(10,5))
plt.title('LSTM',fontsize = 20)
plt.plot(y_test , label = 'Actual', color = "darkblue",linestyle="dotted")
plt.plot(y_pred , label = 'Predicted', color = "red")
plt.rc('legend', fontsize = 15)
plt.xlim(0.0,160)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.grid(True)
plt.legend()
#Importing Libraries
#test
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
from datetime import timedelta
#-----------------------------------------
#folio indicators
#-----------------------------------------

print('Reading data ....')
data_set = pd.read_excel('Data/Portfolio_data3_convert_USD_only.xlsx',na_values='ND')
print('Done')
print(data_set.tail(40))
print(data_set.info()) #now see float 64 (aside from time serie)

data_set=data_set[data_set['More Columns.GroupByLevel1Value']=='JAPANESE YEN']
target_column = 'More Columns.Asset Fx Delta'
df = data_set[target_column]
df_date= np.array(data_set['Date'])
#Preprocessing data set
df = np.array(df).reshape(-1,1)
#df_date=np.array(df_date).reshape(-1,1)
#-----------------------------------------
#preparing training and testing data
#-----------------------------------------


fig, ax = plt.subplots(figsize=(12, 12))
ax.set(xlabel="Date",
       ylabel="Asset Value FX Delta",
       title="Portfolio Convertable - Maerican Dollar")
plt.xticks(rotation=70)
plt.plot(df_date,df)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() #normalises - inverse below
df = scaler.fit_transform(df)

#Training and test sets
train = df[:350]
test = df[350:]
test_date = df_date[350:]

def get_data(data, look_back):
  datax, datay = [],[]
  for i in range(len(data)-look_back-1):
    datax.append(data[i:(i+look_back),0])
    datay.append(data[i+look_back,0])
  return np.array(datax) , np.array(datay)

# Used to predict 1 day in the future:
def get_dataTest(data, look_back):
  datax, datay = [],[]
  for i in range(len(data)-look_back):
    datax.append(data[i:(i+look_back),0])
    datay.append(data[i+look_back,0])
  j = len(data)-look_back
  datax.append(data[j:(j+look_back),0])
  return np.array(datax) , np.array(datay)

look_back = 40

#Create training inputs [samples, timesteps] and outputs [samples]
x_train , y_train = get_data(train, look_back)

#Create testing inputs [samples, timesteps] and outputs [samples]
x_test , y_test = get_dataTest(test,look_back)

n_features = 1
#Processing train and test sets for LSTM model
#Reshape inputs from [samples, timesteps] into [samples, timesteps, features]
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], n_features)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], n_features)
result_date = test_date[look_back:]
assert len(result_date)==len(y_test)

#-----------------------------------------
#LSTM
#-----------------------------------------

#Defining the LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Bidirectional

#Stacked LSTM:
timesteps=x_train.shape[1]
model=Sequential()
model.add(LSTM(100,activation='relu', return_sequences=False, input_shape=(timesteps,n_features)))
#model.add(LSTM(100,activation='relu', return_sequences=True, input_shape=(timesteps,n_features)))
#model.add(LSTM(100,activation='relu'))
model.add(Dense(1))

#Compiling
model.compile(optimizer='adam', loss = 'mse')

#Training
#epoch - number of times learning goes through data - beware over fitting
#batch - keeps n training points the same 
model.fit(x_train,y_train, epochs =100, batch_size=30)

#-----------------------------------------
#LSTM testing
#-----------------------------------------

#Prediction using the trained model
scaler.scale_
y_pred = model.predict(x_test)
#plt.plot(y_pred) broken

#Processing test shape
y_testPlot = np.array(y_test).reshape(-1,1)
y_testPlot = scaler.inverse_transform(y_testPlot)
y_predPlot = scaler.inverse_transform(y_pred)

TestingData = pd.DataFrame(data = y_testPlot,columns = ['Asset Value FX'])
TestingData = pd.concat([TestingData,pd.DataFrame({'Date':result_date})],axis = 1)

LSTM1day = pd.DataFrame(data = y_predPlot,columns = ['Asset Value FX'])
LSTM1day = pd.concat([LSTM1day,pd.DataFrame({'Date':result_date})],axis = 1)
LSTM1day['Date'][len(LSTM1day)-1] = LSTM1day['Date'][len(LSTM1day)-2]+timedelta(days = 1)

#Visualizing the results with dates!
plt.figure(figsize=(10,5))
plt.title('LSTM',fontsize = 20)
plt.plot(TestingData['Date'], TestingData['Asset Value FX'] , label = 'Actual', color = "darkblue",linestyle="dotted")
plt.plot(LSTM1day['Date'], LSTM1day['Asset Value FX'] , label = 'Predicted', color = "red")
plt.xlabel('Date',fontsize = 20)
plt.ylabel('Asset Value FX Delta in %',fontsize = 20)
plt.rc('legend', fontsize = 15)
plt.xlim(TestingData['Date'][0])
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.grid(True)
plt.xticks(rotation=70)
plt.legend()

#Visualizing the results
plt.figure(figsize=(10,5))
plt.title('LSTM',fontsize = 20)
plt.plot(y_testPlot , label = 'Actual', color = "darkblue",linestyle="dotted")
plt.plot(y_predPlot , label = 'Predicted', color = "red")
#plt.xlabel('Date',fontsize = 15)
#plt.ylabel('Asset Value FX Delta in %',fontsize = 15)
plt.rc('legend', fontsize = 15)
plt.legend()
plt.xlim(0.0)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.grid(True)
plt.legend()


#-----------------------------------------
#Extending prediction
#-----------------------------------------

y_predTotal=y_pred
newMem0 = x_test[-1]
#remove first value and add last prediction value
newMem = np.append(newMem0, y_pred[-1])
for i in range(30):
    newMem = np.delete(newMem, 0)
    newMem = newMem.reshape(1,newMem.shape[0], n_features)
    #predict new future value:
    y_stepPred=model.predict(newMem)
    #add new value to final prediction:
    y_predTotal=np.append(y_predTotal, y_stepPred)
    newMem = np.append(newMem, y_stepPred)
y_predTotal=y_predTotal.reshape(y_predTotal.shape[0], n_features)
y_predTotalPlot = scaler.inverse_transform(y_predTotal)

LSTM30days = pd.DataFrame(data = y_predTotalPlot,columns = ['Asset Value FX'])
LSTM30days = pd.concat([LSTM30days,pd.DataFrame({'Date':result_date})],axis = 1)
for i in range(PredictionRange+1):
    LSTM30days['Date'][len(LSTM30days)-(PredictionRange+1-i)] = LSTM30days['Date'][len(LSTM30days)-(PredictionRange+2-i)]+timedelta(days = 1)

#Visualizing the results with dates!
plt.figure(figsize=(10,5))
plt.title('LSTM - 30 days prediction',fontsize = 20)
plt.plot(LSTM30days['Date'], LSTM30days['Asset Value FX'] , label = 'Predicted', color = "red")
plt.plot(TestingData['Date'], TestingData['Asset Value FX'], label = 'Actual', color = "darkblue",linestyle="dotted")
plt.xlabel('Date',fontsize = 20)
plt.ylabel('Asset Value FX Delta in %',fontsize = 20)
plt.rc('legend', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xticks(rotation=70)
plt.xlim(TestingData['Date'][0])
plt.grid(True)
plt.legend()

#Visualizing the results
plt.figure(figsize=(10,5))
plt.title('LSTM - 30 days prediction',fontsize = 20)
plt.plot(y_testPlot , label = 'Actual', color = "darkblue",linestyle="dotted")
plt.plot(y_predTotalPlot , label = 'Predicted', color = "red")
#plt.xlabel('Date',fontsize = 15)
#plt.ylabel('Asset Value FX Delta in %',fontsize = 15)
plt.rc('legend', fontsize = 15)
plt.xlim(0.0)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.grid(True)
plt.legend()

#-----------------------------------------
#Get estimated hedge
#-----------------------------------------
date_df=pd.DataFrame(test_date,columns = ['Date'])
print(date_df.info())
target_date = '2022-01-17 00:00:00'
index = date_df.loc[date_df['Date'] == target_date]
print(index.index)
print("estimating hedge for ",target_date)
print(index.index)
print("Estimated ", y_predTotalPlot[index.index-look_back]) 
if y_predTotalPlot[index.index-look_back] > 10000000 : #fudge to get on specific date but close enough but close enough
  print("Required Hedge ",y_predTotalPlot[index.index-look_back]) 
else:
  print("No hedge required") 


#-----------------------------------------
#Checking the training results
#-----------------------------------------

y_trainPred=model.predict(x_train)
#Visualizing the results
plt.figure(figsize=(10,5))
plt.title('Training data results',fontsize = 20)
plt.plot(y_train , label = 'Actual', color = "darkblue",linestyle="dotted")
plt.plot(y_trainPred , label = 'Predicted', color = "red")
plt.rc('legend', fontsize = 15)
plt.xlim(0.0)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.grid(True)
plt.legend()

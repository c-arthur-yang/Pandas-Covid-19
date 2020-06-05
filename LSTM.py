#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ArthurYang
"""


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, layers, losses, metrics, callbacks

path = "covid19-global-forecasting-week-5"

def load_dataset():
    train_raw = pd.read_csv(path + "/train.csv")
    test_raw = pd.read_csv(path + "/test.csv")

    confirm = train_raw[train_raw['Target'] == 'ConfirmedCases']
    fatality = train_raw[train_raw['Target'] == 'Fatalities']


    confirm_each_day = confirm.groupby(['Date']).agg({'TargetValue':['sum']})
    fatality_each_day =fatality.groupby(['Date']).agg({'TargetValue':['sum']})

    ### reset index
    confirm_each_day = confirm_each_day['TargetValue'].reset_index('Date')
    fatality_each_day = fatality_each_day['TargetValue'].reset_index('Date')
    

    ### Combine 'confirm' and 'fatality' together
    train = pd.DataFrame()
    train[['Date', 'Cases']] = confirm_each_day[['Date', 'sum']]
    train['Deaths'] = fatality_each_day['sum']

    return train


train = load_dataset()
train = train.drop("Date",axis = 1).astype("float32")
WINDOW_SIZE = 14

def batch_dataset(dataset):
    dataset_batched = dataset.batch(WINDOW_SIZE,drop_remainder=True)
    return dataset_batched

ds_data = tf.data.Dataset.from_tensor_slices(tf.constant(train.values,dtype = tf.float32)) \
   .window(WINDOW_SIZE,shift=1).flat_map(batch_dataset)

ds_label = tf.data.Dataset.from_tensor_slices(
    tf.constant(train.values[WINDOW_SIZE:],dtype = tf.float32))

ds_train = tf.data.Dataset.zip((ds_data,ds_label)).batch(50).cache()


class Block(layers.Layer):
    def __init__(self, **kwargs):
        super(Block, self).__init__(**kwargs)
    
    def call(self, x_input,x):
        x_out = tf.maximum((1+x)*x_input[:,-1,:],0.0)
        return x_out
    
    def get_config(self):  
        config = super(Block, self).get_config()
        return config
        
tf.keras.backend.clear_session()
x_input = layers.Input(shape = (None,2),dtype = tf.float32)
x = layers.LSTM(3,return_sequences = True,input_shape=(None,2))(x_input)
x = layers.LSTM(3,return_sequences = True,input_shape=(None,2))(x)
x = layers.LSTM(3,input_shape=(None,2))(x)
x = layers.Dense(2)(x)

x = Block()(x_input,x)
model = models.Model(inputs = [x_input],outputs = [x])
model.summary()

class MSPE(losses.Loss):
    def call(self,y_true,y_pred):
        err_percent = (y_true - y_pred)**2/(tf.maximum(y_true**2,1e-7))
        mean_err_percent = tf.reduce_mean(err_percent)
        return mean_err_percent
    
    def get_config(self):
        config = super(MSPE, self).get_config()
        return config
    
import datetime

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer,loss=MSPE(name = "MSPE"))

history = model.fit(ds_train,epochs=500)

dfresult = train[["Cases","Deaths"]].copy()
dfresult.tail()

for i in range(50):
    arr_predict = model.predict(tf.constant(tf.expand_dims(dfresult.values[-38:,:],axis = 0)))

    dfpredict = pd.DataFrame(tf.cast(tf.floor(arr_predict),tf.float32).numpy(),
                columns = dfresult.columns)
    dfresult = dfresult.append(dfpredict,ignore_index=True)
    
dfresult.query("Cases==0").head()

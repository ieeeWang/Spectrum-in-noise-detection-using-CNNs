# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 22:49:55 2020
write my own MSE loss using tf (instead of np)
@author: lwang
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# print ('tf:', tf.__version__)
import time
from numpy import save, load

from generate_images import prepare_dataset
from helpers import get_IOU

#%%
def printbar():
    today_ts = tf.timestamp()%(24*60*60)

    hour = tf.cast(today_ts//3600+2,tf.int32)%tf.constant(24)
    minite = tf.cast((today_ts%3600)//60,tf.int32)
    second = tf.cast(tf.floor(today_ts%60),tf.int32)
    
    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}",m))==1:
            return(tf.strings.format("0{}",m))
        else:
            return(tf.strings.format("{}",m))
    
    timestring = tf.strings.join([timeformat(hour),timeformat(minite),
                timeformat(second)],separator = ":")
    tf.print("=========="*8+timestring)
    
#%% load data (to use the same dataset for each run)
images_all = load('images_10k.npy')
labels_all = load('labels_10k.npy')

#%% split for train and test
(x_train, x_test) = images_all[:9000],  images_all[9000:]
(y_train, y_test) = labels_all[:9000],  labels_all[9000:]

y_train = y_train.reshape(len(y_train),12)
y_test = y_test.reshape(len(y_test),12)

#%% show some samples
# pick first N samples
num = 10
images_num = x_train[:num]
labels_num = y_train[:num]
labels_num = labels_num.reshape(len(labels_num), 4,3)
num_row = 2
num_col = 5
# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(images_num[i], cmap='gray')
    ax.set_title('f:{}'.format(labels_num[i,:,0]))
plt.tight_layout()
plt.show()

#%% simple data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# keep only x, i.e., loc infor
# y_train = y_train[:,:,0]
# y_test = y_test[:,:,0]

#%% make data pipeline
def data_iter(features, labels, batch_size=8):
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices)  #indices is shuffled
    for i in range(0, num_examples, batch_size):
        #print('i',i)
        indexs = indices[i: min(i + batch_size, num_examples)]
        yield tf.gather(features,indexs), tf.gather(labels,indexs)
        
# test pipeline
batch_size = 8
(features, labels) = next(data_iter(x_train, y_train, batch_size))
print(features.numpy().shape)
print(labels)   


#%% Build a model
tf.keras.backend.clear_session()

# (A) baseline model: MLP
# inputs = keras.Input(shape=(28, 28))
# x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
# x = layers.Flatten()(x)
# x = layers.Dense(128, activation="relu")(x)
# x = layers.Dense(128, activation="relu")(x)
# outputs = layers.Dense(4)(x)
# model = keras.Model(inputs, outputs)
   
# (B) my ramdom-choosen CNN model (~50k para): slightly (deeper) better perf. than (C)
inputs = keras.Input(shape=(28, 28, 1)) # 1 is needed here to keep the same dim with next conv2D layer
x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(x)
x = layers.Dropout(.2)(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(x)
x = layers.Dropout(.2)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x) # <<<<<<<<
x = layers.Dropout(.2)(x)
outputs = layers.Dense(12)(x)
model = keras.Model(inputs, outputs)

# show model
model.summary()

#%%
# samples = x_train[:5]
# samples_labels = y_train[:5]
# 2 lines above are functional equal to as follow
(samples, samples_labels) = next(data_iter(x_train,y_train,batch_size))

predictions = model(samples)
print("Prediction: {}".format(predictions.numpy()))
print("    Labels: {}".format(samples_labels))

#%% Train the model
# @tf.function # will be 2x faster, but debug is not possible anymore
def loss(model, x, y):
    y = tf.dtypes.cast(y, tf.float32) # int (tf or np) -> tf.float
    y_ = model(x)
    squared_loss = tf.square(y-y_)
    batch_losses = tf.reduce_sum(squared_loss, [0,1])/(y.shape[0]*y.shape[1]) 
    return batch_losses


l = loss(model, samples, samples_labels)
print("Loss test: {}".format(l))

# define acc  
# def get_batch_acc(y, y_, dis=1):
#     # dis >=0, it means |predict - y|<=dis is a correct predict
#     y_dis = tf.math.abs(y-y_)
#     y_bool = (y_dis<= dis)
#     acc = np.sum(y_bool)/(y.shape[0]*y.shape[1])
#     return acc

@tf.function
def metric(model, x, y, dis=1.):
    y = tf.dtypes.cast(y, tf.float32) # int (tf or np) -> tf.float
    y_ = model(x)       
    y_dis = tf.math.abs(y-y_)
    y_bool = (y_dis<= dis)
    as_ints = tf.cast(y_bool, tf.int32) #without this line, error for tf.reduce_sum
    acc = tf.reduce_sum(as_ints,[0,1])/(y.shape[0]*y.shape[1])
    return acc

acc = metric(model, samples, samples_labels, dis=1.)
print("acc test: {}".format(acc))


# Use the tf.GradientTape context to calculate the gradients
# @tf.function # will be 2x faster, but debug is not possible anymore
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, 
                                     beta_1=0.9, beta_2=0.999, epsilon=1e-07)

loss_value, grads = grad(model, samples, samples_labels)
print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))
optimizer.apply_gradients(zip(grads, model.trainable_variables))
print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
            loss(model, samples, samples_labels).numpy()))


#%% Training loop <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Keep results for plotting
train_loss_results = []
train_acc_results = []
test_acc_results = []
start_time = time.time()
num_epochs = 201
for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
     
    # compute from the initial state
    acc_train = metric(model, x_train, y_train)
    acc_test = metric(model, x_test, y_test)
    train_acc_results.append(acc_train)
    test_acc_results.append(acc_test)
        
    # Training loop
    for x, y in data_iter(x_train, y_train, batch_size = 128):
        # Optimize the model
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))        
        # Track progress
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss

    # end of epoch
    train_loss_results.append(epoch_loss_avg.result())
    
    if epoch % 10 == 0:# 17 sec / 10 epochs
        printbar()        
        print("Epoch {:03d}: Loss: {:.3f}, Acc-train: {:.3%}, Acc-test: {:.3%}".\
              format(epoch,
              epoch_loss_avg.result(),
              acc_train,
              acc_test))     
                                   
print('elapsed_time:',  time.time() - start_time)
# evaluate whole test set
acc = metric(model, x_test, y_test)
print('Acc-test: {:.3%}'.format(acc.numpy()))

#%% plot loss and acc over epochs
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')
axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)
axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_acc_results, label='train acc',linestyle='--',linewidth=2)
axes[1].plot(test_acc_results, label='test acc',linestyle='--',linewidth=2)
axes[1].legend(loc='best', fontsize='x-large')

#%% (1) check train set
# pick first N samples
num = 10
images = x_train[:num]
labels = y_train[:num]
labels = labels.reshape(len(labels), 4,3)

labels_pre = model(images)
labels_ = np.round(labels_pre.numpy())
labels_s = labels_.reshape(len(labels_), 4,3)

num_row = 2
num_col = 5
# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(images[i,:,:,0], cmap='gray')
    ax.set_title('r:{}'.format(labels[i,:,0]))
    ax.set_xlabel('p:{}'.format(labels_s[i,:,0]))
plt.tight_layout()
plt.show()

print('real:',y_train[:num])
print('pred:',labels_.astype(int))

#%% (2) check test set
# pick first N samples
num = 10
images = x_test[:num]
labels = y_test[:num]
labels = labels.reshape(len(labels), 4,3)

labels_pre = model(images)
labels_ = np.round(labels_pre.numpy())
labels_s = labels_.reshape(len(labels_), 4,3)

num_row = 2
num_col = 5
# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(images[i,:,:,0], cmap='gray')
    ax.set_title('r:{}'.format(labels[i,:,0]))
    ax.set_xlabel('p:{}'.format(labels_s[i,:,0]))
plt.tight_layout()
plt.show()

print('real:',y_test[:num])
print('pred:',labels_.astype(int))

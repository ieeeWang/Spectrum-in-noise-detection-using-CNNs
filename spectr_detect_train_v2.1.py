# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 21:34:40 2020

@author: lwang
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# print ('tf:', tf.__version__)
import time
from generate_images_v2 import prepare_one_dataset, plotbars, xloc2binaryvec
from numpy import save, load

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
# images_org = load('images_10k_v2.npy') #SNR=0.5
# labels_org = load('labels_10k_v2.npy')
# images_org = load('images_10k_v2_snr1.npy')#SNR=1
# labels_org = load('labels_10k_v2_snr1.npy')
# images_org = load('images_10k_v2_snr1_7f.npy')#3 f are ramdom
# labels_org = load('labels_10k_v2_snr1_7f.npy')
images_org = load('images_10k_v2_snr1_7frand.npy')#7 f are ramdom
labels_org = load('labels_10k_v2_snr1_7frand.npy')
 

# plot first N samples
num = 20
images_num = images_org[:num]
labels_num = labels_org[:num]
fig = plt.figure(figsize=(5, 10))
for i in range(num):
    # plot the sample
    plt.subplot(num,1,i+1)
    plt.imshow(images_num[i], cmap='gray')
    plotbars(labels_num[i][0,:])
    if i==0:
        plt.title('train set')
    
    
#%% loc labels -> binary labels
N = len(labels_org)
labels_bi = np.zeros((N,120), dtype ='int32')
for i in range(N):
    temp = labels_org[i,0,:]
    labels_bi[i]=xloc2binaryvec(temp)    
 
   
#%% split for train and test
(x_train, x_test) = images_org[:9000],  images_org[9000:]
(y_train, y_test) = labels_bi[:9000],  labels_bi[9000:]

#%% shuffle [x_train, y_train] along the dimenstiion of '120' 
# so that trained NN does not predict by 'remembering' f order. 
nn, fn = y_train.shape #9000, 120
for i in range(nn):
    indices = np.array(range(fn))
    np.random.shuffle(indices)   
    y_train[i,:] = y_train[i,:][indices]
    x_train[i,:,:] = x_train[i,:,:][:,indices]

# check above code: plot m samples with shuffled labels
m =10
fig = plt.figure(figsize=(5, 10))
for i in range(m):
    image = x_train[i]
    label = np.where(y_train[i] > 0.5)[0] # len() == 6 
    plt.subplot(m,1,i+1)
    plt.imshow(image, cmap='gray')
    if i==0:
        plt.title('check shuffled labels')
    plotbars(label)

#%% reshape for CNN: not neccesary, tf.keras will take care if this 
# x_train = x_train.reshape(x_train.shape[0], 12, 120, 1)
# x_test = x_test.reshape(x_test.shape[0], 12, 120, 1) 
 
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
# print(labels)   

#%% Build a model
tf.keras.backend.clear_session()

# (A) baseline model: MLP
# inputs = keras.Input(shape=(12, 120))
# x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
# x = layers.Flatten()(x)
# x = layers.Dense(1024, activation="relu")(x)
# x = layers.Dropout(.2)(x)
# x = layers.Dense(128, activation="relu")(x)
# outputs = layers.Dense(120, activation= 'sigmoid')(x)
# model = keras.Model(inputs, outputs)   

# (B) CNN model (~1 conv layer): ~ 9m paras
# inputs = keras.Input(shape=(12, 120, 1)) # 1 is needed here to keep the same dim with next conv2D layer
# x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
# x = layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x) # stride =1, not =3 be default
# x = layers.Dropout(.2)(x)
# x = layers.Flatten()(x)
# x = layers.Dense(1024, activation="relu")(x) # this layer further improves perf
# x = layers.Dropout(.2)(x)
# outputs = layers.Dense(120, activation= 'sigmoid')(x)
# model = keras.Model(inputs, outputs)

# (C) CNN model (~2 conv layers): ~ 230k paras
inputs = keras.Input(shape=(12, 120, 1)) # 1 is needed here to keep the same dim with next conv2D layer
x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
x = layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1),padding='same')(x) # stride =1, not =3 be default
x = layers.Dropout(.2)(x)
x = layers.Conv2D(filters=4, kernel_size=(3, 3), padding='same', activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x) # stride =1, not =2 be default
x = layers.Dropout(.2)(x)
x = layers.Flatten()(x)
# x = layers.Dense(1024, activation="relu")(x) # it does not further improve!
outputs = layers.Dense(120, activation= 'sigmoid')(x)
model = keras.Model(inputs, outputs)

# show model
model.summary()

#%% test initial model 
(image_samples, labels_samples) = next(data_iter(x_train,y_train,batch_size))
prediction_samples = model(image_samples)

print("Prediction: {}".format(prediction_samples[0].numpy()))
print("    Labels: {}".format(labels_samples[0]))

# show a sample:
# y_pred1=prediction_samples[0].numpy()
# y_true1=labels_samples[0].numpy()
# loc_pred = np.where(y_pred1>0.5)
# loc_true = np.where(y_true1>0.5)
# print(loc_pred)
# print(loc_true)
# print(np.intersect1d(loc_true,loc_pred))

#%% define the model
#first initial an loss object, error otherwise (put it outside following loss function to save memory)
bce = tf.keras.losses.BinaryCrossentropy()# arguments not allowed here! 

@tf.function # will be 2x faster, but debug is not possible anymore
def loss(model, x, y):
    y_ = model(x)
    batch_loss = bce(y, y_)# arguments here! 
    return batch_loss

l = loss(model, image_samples, labels_samples)
print("Loss test: {}".format(l))

# using np instead of tf
def metric(model, x, y, theta = 0.5):
    e=1e-10
    m = len(y)
    y_ = model(x).numpy()
    Sens = np.zeros(m);  PPV = np.zeros(m)
    for i in range(m):
        loc_pred = np.where(y_[i]>theta)[0]
        loc_true = np.where(y[i] >theta)[0] # len() == 6
        loc_overlap = np.intersect1d(loc_true,loc_pred)
        num_overlap=len(loc_overlap)
        Sens[i] = num_overlap/len(loc_true)
        PPV[i] = num_overlap/(len(loc_pred)+e) #to avoid division by zero
            
    return np.mean(Sens), np.mean(PPV)


Sens, PPV = metric(model, image_samples, labels_samples.numpy())
print("Sens: {}, PPV: {}".format(Sens, PPV))


#%%
# Use the tf.GradientTape context to calculate the gradients
@tf.function # will be 2x faster, but debug is not possible anymore
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, 
                                     beta_1=0.9, beta_2=0.999, epsilon=1e-07)

loss_value, grads = grad(model, image_samples, labels_samples)
print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))
optimizer.apply_gradients(zip(grads, model.trainable_variables))
print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
            loss(model, image_samples, labels_samples).numpy()))

#%% Training loop
start_time = time.time()

# Keep training details
train_Sens_results = []; train_PPV_results = []
test_Sens_results = []; test_PPV_results = []

train_loss_results = []
num_epochs = 101
for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
        
    # Training loop
    for x, y in data_iter(x_train, y_train, batch_size = 128):
        # Optimize the model
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))        
        # Track progress
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss

    # after end of each epoch, update loss, compute metrics and store them
    train_loss_results.append(epoch_loss_avg.result())

    Sens_train, PPV_train = metric(model, x_train, y_train)
    Sens_test, PPV_test = metric(model, x_test, y_test)
    
    train_Sens_results.append(Sens_train)
    train_PPV_results.append(PPV_train)
    test_Sens_results.append(Sens_test)
    test_PPV_results.append(PPV_test)
    

    if epoch % 10 == 0:# 17 sec / 10 epochs
        printbar()        
        print("Epoch {:03d}: Loss: {:.5f}".format(epoch, epoch_loss_avg.result()))  
        print("Sens-train: {:.3%}, PPV-train: {:.3%}".format(Sens_train,PPV_train))
        print("Sens-test: {:.3%}, PPV-test: {:.3%}".format(Sens_test,PPV_test))
                                   
print('elapsed_time:',  time.time() - start_time)


#%% plot loss and metrics over epochs
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')
axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)
axes[1].set_ylabel("Metrics", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_Sens_results, label='train Sens',linestyle='-',linewidth=2)
axes[1].plot(train_PPV_results, label='train PPV',linestyle='-',linewidth=2)
axes[1].plot(test_Sens_results, label='test Sens',marker= 'o', linestyle='--',linewidth=2)
axes[1].plot(test_PPV_results, label='test PPV',marker= '*', linestyle='--',linewidth=2)
axes[1].legend(loc='best', fontsize='x-large')

#%% check test set
test_images = images_org[9000:]
test_labels = labels_org[9000:]

# loc labels -> binary labels
N = len(test_labels)
test_labels2 = np.zeros((N,120), dtype ='int32')
for i in range(N):
    test_labels2[i]=xloc2binaryvec(test_labels[i,0,:])
    
    
# pick N samples
num = 20
start = 100
num_images = test_images[start : start+num]
num_labels_true = test_labels[start : start+num]
num_labels_true2 = test_labels2[start : start+num]
# prediction
labels_p = model(num_images).numpy()
num_labels_pre = []
theta = 0.5
for i in range(num):
    num_labels_pre.append(np.where(labels_p[i]>theta)[0])
    
Sens, PPV = metric(model, num_images, num_labels_true2, theta)
# print("Sens-test: {}, PPV-test: {}".format(Sens, PPV))
print("Sens-test: {:.3%}, PPV-test: {:.3%}".format(Sens, PPV))
    

#%% plot 
# raw images without labels
fig = plt.figure(figsize=(5, 10))
for i in range(num):
    # plot the sample
    plt.subplot(num,1,i+1)
    plt.imshow(num_images[i], cmap='gray')
    if i==0:
        plt.title('samples from test set')

# raw images with ground truth
fig = plt.figure(figsize=(5, 10))
for i in range(num):
    # plot the sample
    plt.subplot(num,1,i+1)
    plt.imshow(num_images[i], cmap='gray')
    if i==0:
        plt.title('ground truth')
    plotbars(num_labels_true[i,0,:], color = 'black') # True

# raw images with prediction
fig = plt.figure(figsize=(5, 10))
for i in range(num):
    # plot the sample
    plt.subplot(num,1,i+1)
    plt.imshow(num_images[i], cmap='gray')
    if i==0:
        plt.title('prediction')
    plotbars(num_labels_pre[i]) # Prediction
    
#%% find samples wrong predicts
def count_wrong_predi(model, x, y, theta = 0.5):
    e=1e-10
    m = len(y)
    y_ = model(x).numpy()
    Sens = np.zeros(m);  PPV = np.zeros(m)
    count=[]
    for i in range(m):
        loc_pred = np.where(y_[i]>theta)[0]
        loc_true = np.where(y[i] >theta)[0] # len() == 6
        loc_overlap = np.intersect1d(loc_true,loc_pred)
        num_overlap=len(loc_overlap)
        Sens[i] = num_overlap/len(loc_true)
        PPV[i] = num_overlap/(len(loc_pred)+e) #to avoid division by zero
        if (Sens[i]<0.99 or PPV[i]<0.99):
            count.append(i)
            
    return count, Sens, PPV

    
count, Sens, PPV = count_wrong_predi(model, test_images, test_labels2)
print('Numboer of samples from test set with wrong prediction:', len(count))

# select the wrong samples
loc_wrong = np.array(count)
wrong_images = test_images[loc_wrong] 
wrong_labels = test_labels[loc_wrong] 
wrong_labels2 = test_labels2[loc_wrong] 
# prediction
labels_tmp = model(wrong_images).numpy()
wrong_labels_pre = []
theta = 0.5
for i in range(len(count)):
    wrong_labels_pre.append(np.where(labels_tmp[i]>theta)[0])
 
#%% plot a P-R curve on one wrong samle
mw = 11
demo_image = wrong_images[mw]
demo_labels_true = wrong_labels2[mw]
demo_labels_pred = labels_tmp[mw]

# Create PR
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

pos_label= 1
# P-R 
prec, recall, thresholds = precision_recall_curve(demo_labels_true, demo_labels_pred,
                                         pos_label=pos_label)
# alternative AUCpr (~AP), with a different computing method
AP = average_precision_score(demo_labels_true, demo_labels_pred, pos_label=pos_label)
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall, 
                                    average_precision=AP,
                                    estimator_name='demo').plot()

# ROC
fpr, tpr, _ = roc_curve(demo_labels_true, demo_labels_pred, pos_label=pos_label)
AUC = auc(fpr, tpr)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=AUC, estimator_name='demo').plot()



#%% plot images with wrong predictions
num = 20
start = 30
# wrong images without labels
fig = plt.figure(figsize=(5, 10))
for i in range(num):
    # plot the sample
    plt.subplot(num,1,i+1)
    plt.imshow(wrong_images[start+i], cmap='gray')
    if i==0:
        plt.title('wrong samples from test set')

# wrong images with ground truth
fig = plt.figure(figsize=(5, 10))
for i in range(num):
    # plot the sample
    plt.subplot(num,1,i+1)
    plt.imshow(wrong_images[start+i], cmap='gray')
    if i==0:
        plt.title('ground truth')
    plotbars(wrong_labels[start+i,0,:]) # , color = 'black'
    
# wrong images with prediction
fig = plt.figure(figsize=(5, 10))
for i in range(num):
    # plot the sample
    plt.subplot(num,1,i+1)
    plt.imshow(wrong_images[start+i], cmap='gray')
    if i==0:
        plt.title('prediction')
    plotbars(wrong_labels_pre[start+i]) # Prediction











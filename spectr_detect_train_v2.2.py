# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 14:22:36 2020
isntead of defining a for-loop train in v2.1, here use high-level APIs: model.fit, 
METRICS refer: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
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
from generate_images_v2 import prepare_one_dataset, plotbars, xloc2binaryvec

# sklearn for PR and ROC curves
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

#%% 
# (opt.A) using my own metric, by np
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

# count the number of samples with wrong prediction
def count_wrong_predi(model, x, y, theta = 0.5):
    e=1e-10#to avoid division by zero
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

#%% lod data (to use the same dataset for each run)
images_org = load('data/images_10k_v2_snr1_7frand.npy')#7 f are ramdom
labels_org = load('data/labels_10k_v2_snr1_7frand.npy')
 

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

#%% shuffle [x_train, y_train] along the dimenstiion of '120',
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


#%% model.compile
loss = tf.keras.losses.BinaryCrossentropy()
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, 
                                     beta_1=0.9, beta_2=0.999, epsilon=1e-07)

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      ]


## (A) using model.compile to use model.fit
model.compile(
    optimizer= optimizer,
    loss=loss,
    metrics= METRICS,
    )

callback_log = [keras.callbacks.TensorBoard(log_dir='./logs')] # for using TensorBoard
# Launch TensorBoard from the command line (first cd: folder of this proj): 
    # tensorboard --logdir logs

# Create a callback that saves the model's weights
checkpoint_path = "training_1/cp.ckpt"
callback_cp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

#%% train
N_epoch = 10
start_time = time.time()

# (A) Train the model from Numpy data
history = model.fit(x_train, y_train, validation_data= (x_test, y_test), 
                    batch_size= 128, epochs=N_epoch, 
                    callbacks=[callback_log, callback_cp])

print('elapsed_time:',  time.time() - start_time) # =203 s for 100 epochs

#%% plot loss over epochs
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_loss(history, label, n):
    # Use a log scale (plt.semilogy) to show the wide range of values. 
    plt.semilogy(history.epoch,  history.history['loss'],
                 color=colors[n], label='Train '+label)
    plt.semilogy(history.epoch,  history.history['val_loss'],
            color=colors[n], label='Val '+label,
            linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

fig = plt.figure(figsize=(12, 10)) 
plot_loss(history, "loss", 2)

#%% plot metrics over epochs
def plot_metrics(history):
    metrics =  ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])
        plt.legend()

fig = plt.figure(figsize=(12, 10))    
plot_metrics(history)
    

#%% check test set
test_images = images_org[9000:]
test_labels = labels_org[9000:]

# loc labels -> binary labels
N = len(test_labels)
test_labels2 = np.zeros((N,120), dtype ='int32')
for i in range(N):
    test_labels2[i]=xloc2binaryvec(test_labels[i,0,:])
         
Sens, PPV = metric(model, test_images, test_labels2, 0.5)
print("TEST set: Sens-test: {:.3%}, PPV-test: {:.3%}".format(Sens, PPV))
   
    
# pick N samples for plot
num = 10
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
print("batch: Sens-test: {:.3%}, PPV-test: {:.3%}".format(Sens, PPV))
    

# plot 
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
    
#%% find samples with wrong predictions
count, Sens, PPV = count_wrong_predi(model, test_images, test_labels2)
print('Numboer of samples from test set with wrong prediction:', len(count))

# get the wrong samples
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
 

# plot images with wrong predictions
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


#%% plot a P-R curve on one wrong-predicted samle
mw = 1
demo_image = wrong_images[mw]
demo_labels_true = wrong_labels2[mw]
demo_labels_pred = labels_tmp[mw]

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

# Combining the display objects (ROC and PR) into a single plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
roc_display.plot(ax=ax1)
pr_display.plot(ax=ax2)
plt.show()

#%% plot a P-R curve on entire test set
test_labels_pre = model(test_images).numpy().reshape(-1) # (120000,)
test_labels_true = test_labels2.reshape(-1) # (120000,)

# P-R 
prec, recall, thresholds = precision_recall_curve(test_labels_true, test_labels_pre,
                                         pos_label=pos_label)
# alternative AUCpr (~AP), with a different computing method
AP = average_precision_score(test_labels_true, test_labels_pre, pos_label=pos_label)
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall, 
                                    average_precision=AP,
                                    estimator_name='demo').plot()

# ROC
fpr, tpr, _ = roc_curve(test_labels_true, test_labels_pre, pos_label=pos_label)
AUC = auc(fpr, tpr)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=AUC, estimator_name='demo').plot()

# Combining the display objects (ROC and PR) into a single plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
roc_display.plot(ax=ax1)
pr_display.plot(ax=ax2)
plt.show()



































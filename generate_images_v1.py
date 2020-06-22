# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 15:16:55 2020
generate 28*28 grey images with random vertical bars
@author: lwang
"""

import numpy as np
import matplotlib.pyplot as plt
# print(np.__version__)

# genearate loc of vertical bars (x,y)
def get_n_loc(n=3):
    # genearate x: n loc of freq in [0,..., 27]
    f_set = np.array(range(1,11))*3-3 
    # remove 1st and last samples
    f_set = f_set[1:-1] # 8 entries
    # results are from [0, n)
    # loc = np.random.randint(10, size= n)# with repeting
    loc = np.random.choice(len(f_set), n, replace=False) # without repeting
    
    loc2 = np.sort(f_set[loc]) # assending

    # generate y 
    def get_y_pair(ymin=0, ymax=27):
        y = np.random.randint(ymin, ymax+1, size=2)
        return np.sort(y)
     
    y_arr = np.zeros([n,2],dtype=int)    
    for i in range(n): 
        y_arr[i,] = get_y_pair()
    
    return loc2, y_arr

# generate grey images using bars (x,y)
def get_image(n=5):
    (x, y) = get_n_loc(n)
    
    ndim = (28, 28)
    image0 = np.zeros(ndim) 
    for i in range(len(x)):
        image0[x[i], y[i,0]:y[i,1]] = 255        
    return np.transpose(image0), (x, y)

# generate grey image dataset
def prepare_dataset(N_sam = 10000):
    n=4
    images = np.zeros((N_sam, 28, 28), dtype=int)
    labels = np.zeros((N_sam, n, 3),dtype=int)
    for i in range(N_sam): 
        image, (x, y) = get_image(n) 
        x= np.reshape(x,[n,1])
        label = np.concatenate((x,y), axis=1)
        images[i]=image
        labels[i]=label
    return images,labels

    
#%%
if __name__=='__main__':  
    n=4     
    image, (x, y) = get_image(n)
    # plot the sample
    fig = plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(x)
    plt.xlabel(y)
    
    x= np.reshape(x,[n,1])
    label = np.concatenate((x,y), axis=1)
    
    from time import time
    t0 = time()
    images,labels =prepare_dataset(1000)
    print('Elapsed time is ', time()-t0)
    
    #%% generate random data & save
    # images_all, labels_all = prepare_dataset(10000) # ~10 sec
    # save('images_10k.npy', images_all)
    # save('labels_10k.npy', labels_all)
        
    

















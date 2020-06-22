# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:47:51 2020

@author: lwang
"""


import numpy as np
import matplotlib.pyplot as plt
# swap in python is easy:
# y1, y2 = 2, 3
# if y1>y2:
#     y1, y2 = y2, y1

# get the ratio of intercetion [0 1]
def get_IOU(A, B):
    y1, y2 = A
    y1_, y2_ = B
    # make sure y1<=y2, y1_<=y2_
    if y1>y2:
        y1, y2 = y2, y1
    if y1_>y2_:
        y1_, y2_ = y2_, y1_  
    
    if y2<=y2_:
        return _IOU(y1, y2, y1_, y2_)
    else:#wape A and B
        y1, y1_ = y1_, y1
        y2, y2_ = y2_, y2
        return _IOU(y1, y2, y1_, y2_)
               

def _IOU(y1, y2, y1_, y2_):
    # y1<=y2, y1_<=y2_ and y2<=y2_
    numerator = np.min([np.max([0, y2-y1_]), np.abs(y2-y1)])
    denominator = np.max([y2_-y1, y2_-y1_])
    epsilon=1e-07
    IOU = numerator/(denominator+epsilon) # not divide by 0
    return IOU
        
#%%
if __name__=='__main__':  
    # case 1:    
    A = (1,3)
    B = (2,4)
    print(get_IOU(A, B))
    print(get_IOU(B, A))
    
    A = (1,3)
    B = (1,3)
    print(get_IOU(A, B))
    print(get_IOU(B, A))
    
    # case 2:    
    A = (1,3)
    B = (3,5)
    print(get_IOU(A, B))
    print(get_IOU(B, A))
        
    # case 3:    
    A = (3,3)
    B = (3,3)
    print(get_IOU(A, B))
    print(get_IOU(B, A))

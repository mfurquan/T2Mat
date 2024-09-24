#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:05:24 2024

@author: mfurquan
"""

import numpy as np

def membrane_inertia(H):
    mat = np.zeros((H.Nen,H.Nen,H.Ndf,H.Ndf))
    for i in range(H.Nen):
        for j in range(H.Nen):
            mat[i,j,:,:] = -H.N[i]*H.N[j] # *I(1x1)
    return mat

def membrane_stiffness(H):
    mat = np.zeros((H.Nen,H.Nen,H.Ndf,H.Ndf))
    for i in range(H.Nen):
        for j in range(H.Nen):
            mat[i,j,:,:] = np.dot(H.grad_N[i],H.grad_N[j])
    return mat
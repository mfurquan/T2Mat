#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:29:48 2024

@author: mfurquan
"""

import matplotlib.pyplot as plt

def plot(m,d,Ndf):
    for i in range(Ndf):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        tcf=ax.tricontourf(m.x[:,0],m.x[:,1],d[:,i],levels=20)#, z_test_refi, levels=levels, cmap='terrain')
        fig.colorbar(tcf)
        plt.show()
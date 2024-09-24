#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 17:47:53 2024

@author: mfurquan
"""

import numpy as np

class shape_fn:
    def __init__(self,Type,Order,Nsd):
        match Type:
            case 'Lagrange':
                match Order:
                    case 1:
                        match Nsd:
                            case 2:
                                self.N = lambda xi: np.array([(1.-xi[0])*(1.-xi[1]),
                                                              (1.+xi[0])*(1.-xi[1]),
                                                              (1.+xi[0])*(1.+xi[1]),
                                                              (1.-xi[0])*(1.+xi[1])])/4.
                                self.N_xi = lambda xi: np.array([[-(1.-xi[1]),(1.-xi[1]),(1.+xi[1]),-(1.+xi[1])],
                                                                 [-(1.-xi[0]),-(1.+xi[0]),(1.+xi[0]),(1.-xi[0])]])/4.
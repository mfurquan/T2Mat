#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 18:10:43 2024

@author: mfurquan
"""

import numpy as np

class quadrature:
    def __init__(self,Type,Nint1d,Nsd):
        self.Nint = Nint1d**Nsd
        match Type:
            case 'Legendre':
                match Nsd:
                    case 1:
                        match Nint1d:
                            case 1:
                                self.xq = np.array([0.])
                                self.wq = np.array([2.])
                            case 2:
                                self.xq = np.array([-1.,1.])/np.sqrt(3.)
                                self.wq = np.array([1.,1.])
                            case 3:
                                self.xq = np.array([-1.,0.,1.])*np.sqrt(3./5.)
                                self.wq = np.array([5.,8.,5.])/9.
                    case 2:
                        q1 = quadrature(Type,Nint1d,1)
                        self.xq = np.array([[x,y] for x in q1.xq for y in q1.xq])
                        self.wq = np.array([x*y for x in q1.wq for y in q1.wq])
                        
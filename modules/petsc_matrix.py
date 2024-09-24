#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 12:55:01 2024

@author: mfurquan
"""

from petsc4py import PETSc
import numpy as np

class matrix:
    def __init__(self,n):
        self.A = PETSc.Mat().create()
        self.A.setSizes([n,n])
        self.A.setUp()
    def populate(self,a_loc,lm_loc,Nen,Ndf):
        for i in range(Nen):
            for j in range(Nen):
                for idf in range(Ndf):
                    for jdf in range(Ndf):
                        self.A.setValues(lm_loc[i,idf],lm_loc[j,jdf],a_loc[i,j,idf,jdf],addv=True)
    def eliminate(self,irow,diag_val):
        self.A.zeroRows(irow,diag_val)
        self.A.assemble()
                        
class vector:
    def __init__(self,n):
        self.b = np.zeros(n)
    def populate(self,b_loc,lm_loc,Nen,Ndf):
        for i in range(Nen):
            for idf in range(Ndf):
                self.b[lm_loc[i,idf]] += b_loc[i,idf]
    def set_val(self,irow,val):
        self.b[irow] = val
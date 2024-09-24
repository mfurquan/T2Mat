#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:03:31 2024

@author: mfurquan
"""

import numpy as np
from modules.meshes import edge2nod

class fe_space:
    def __init__(self,mesh,shape_fn,quadr,Ndof):
        self.m = mesh
        self.Ndf = Ndof
        self.Nen = self.m.Nen
        self.LM = mesh.generate_LM(Ndof)
        self.q = quadr
        self.N_allq = np.apply_along_axis(shape_fn.N,1,quadr.xq)
        self.N_xi_allq = np.apply_along_axis(shape_fn.N_xi,1,quadr.xq)
        
    def assemble(self,matrix_fns,A,vector_fn=None,b=None):
        A_local = [np.zeros((self.Nen,self.Nen,self.Ndf,self.Ndf)) for i in range(len(matrix_fns))]
        if vector_fn != None:
            b_local = np.zeros((self.Nen,self.Ndf))
        for ie in self.m:
            for iq in range(self.q.Nint):
                self.N = self.N_allq[iq]
                self.N_xi = self.N_xi_allq[iq]
                self.x_xi = np.matmul(self.N_xi,self.m.xe)
                self.jac  = np.linalg.det(self.x_xi)
                self.xi_x = np.linalg.inv(self.x_xi)
                self.grad_N = np.transpose(np.matmul(self.xi_x,self.N_xi))
                for i in range(len(matrix_fns)):
                    A_local[i] += self.q.wq[iq]*matrix_fns[i](self)*self.jac
                    if vector_fn != None:
                        b_local += self.q.wq[iq]*vector_fn(self)*self.jac
            
                lm_local = self.LM[self.m.ien[ie,:],:]
                for i in range(len(matrix_fns)):
                    A[i].populate(A_local[i],lm_local,self.Nen,self.Ndf)
                if b != None:
                    b.populate(b_local,lm_local,self.Nen,self.Ndf)
        for a in A:
            a.A.assemblyBegin()
            a.A.assemblyEnd()
                    
    def set_dirichlet(self,ib,idf,A,diag_vals,b=None,b_val=None):
        bn = edge2nod(self.m.ibn[ib],self.m.ien,self.m.x)
        ifixed = self.LM[bn,idf]
        for i in range(len(A)):
            A[i].eliminate(ifixed,diag_vals[i])
        if b != None:
            b.set_val(ifixed,b_val)
            
        
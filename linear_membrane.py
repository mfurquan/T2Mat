#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:26:51 2024

@author: mfurquan
"""
import numpy as np
from modules.rectangle_mesh import rect
import modules.shape_fns as sh
import modules.quadrature_rules as qd
import modules.fe_spaces as fe
import modules.petsc_matrix as pt
import modules.operators as op
import modules.slepsc_solve as ss

Ndf = 1
shift = -25.0

m = rect(0.,0.,np.pi,np.pi,20,80)
s = sh.shape_fn('Lagrange',1,m.Nsd)
q = qd.quadrature('Legendre',2,m.Nsd)
H = fe.fe_space(m,s,q,Ndf)

M = pt.matrix(H.m.Nn*H.Ndf)
K = pt.matrix(H.m.Nn*H.Ndf)

H.assemble((op.membrane_inertia,op.membrane_stiffness),(M,K))
H.set_dirichlet(1,0,(M,K),(0.,1.))
H.set_dirichlet(3,0,(M,K),(0.,1.))

E = ss.eig_solver(M,K,shift)
E.solve()

E.plot(0,H)
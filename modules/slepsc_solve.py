#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:48:29 2024

@author: mfurquan
"""
from petsc4py import PETSc
from slepc4py import SLEPc
import modules.plot as pl

class eig_solver:
    def __init__(self,M,K,target):
        self.E = SLEPc.EPS(); self.E.create()
        self.E.setOperators(K.A,M.A)
        self.E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        self.E.setTarget(target)
        self.st = self.E.getST()
        self.st.setType(SLEPc.ST.Type.SINVERT)
        self.E.setDimensions(3,PETSc.DECIDE)
        self.E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        self.vr, self.wr = K.A.getVecs()
        self.vi, self.wi = K.A.getVecs()
    def solve(self):
        self.E.solve()
        Print = PETSc.Sys.Print

        Print()
        Print("******************************")
        Print("*** SLEPc Solution Results ***")
        Print("******************************")
        Print()

        its = self.E.getIterationNumber()
        Print("Number of iterations of the method: %d" % its)

        eps_type = self.E.getType()
        Print("Solution method: %s" % eps_type)

        nev, ncv, mpd = self.E.getDimensions()
        Print("Number of requested eigenvalues: %d" % nev)

        tol, maxit = self.E.getTolerances()
        Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))

        nconv = self.E.getConverged()
        Print("Number of converged eigenpairs %d" % nconv)
        if nconv > 0:
            Print()
            Print("        k          ||Ax-kx||/||kx|| ")
            Print("----------------- ------------------")
            for i in range(nconv):
                k = self.E.getEigenpair(i)#, vr, vi)
                error = self.E.computeError(i)
                if k.imag != 0.0:
                    Print(" %9f%+9f j %12g" % (k.real, k.imag, error))
                else:
                    Print(" %12f      %12g" % (k.real, error))
            Print()
    def plot(self,ieig,FEspace):
        k = self.E.getEigenpair(ieig, self.vr, self.vi)
        mr = self.vr.getArray()
        mi = self.vi.getArray()
        mr = mr.reshape([FEspace.m.Nn,FEspace.Ndf])
        mi = mi.reshape([FEspace.m.Nn,FEspace.Ndf])
        pl.plot(FEspace.m,mr,FEspace.Ndf)
        pl.plot(FEspace.m,mi,FEspace.Ndf)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on:         12th April, 2024
Last Modified on:   12th April, 2024

@authors: P. Vazao Balbao & R.F. Ausas
"""

#----------------------------------------------------------------------
# General libraries
import os
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------
# Our libraries
from ddcm_classes import *
#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# User defined main() functions: Program your options and problem below

def main_EX1_():

    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')
    
    # 1. Create (or import) the computational domain/mesh

    nsd, Ls, Ns = 2, [1.0, 1.0], [100, 100]

    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    print('-----------------------------')
    print('|:: Computational domain OK')
    print('-----------------------------')
    
    # 2. Compute a reference solution to initially distribute the data
    
    params = [Constant(1.0)]
    
    def mysource(x):
        return fd.Constant(1.0)

    def mysurfaceflux(x):
        return fd.Constant(0.0)

    def mybcvals(x):
        return fd.Constant(0.123456789)

    forcings = ForcingData(mysource, mysurfaceflux, mybcvals)

    ref_sol = StdProblem(domain, params, 'Setting-A', forcings, FLM_DarcyLinear, PoissonDualMixedHdiv, orders=[2, 1])

    ref_sol.solve()

    print('-----------------------------')
    print('|:: Reference solution OK')
    print('-----------------------------')

    '''
    # 3. Read or sample some synthetic data from a standard solution

    ddcm_data = DDCMData()

    ddcm_data.CreateSyntheticData(StdModelFlags('Darcy-Linear', 'Dual-Mixed', 'Uniaxial'), read_data=False)

    print('-----------------------------')
    print('|:: Synthetic data OK')
    print('-----------------------------')

    # 4. Solve the Data Driven problem

    ddcm_sol = DDCMProblem(ddcm_data, ddcm_formulation = 'NLSQP-5F-A')

    ddcm_sol.solve()

    print('-----------------------------')
    print('|> Data Driven solution OK')
    print('-----------------------------')
    '''
    
    elapsed_time = time.time() - ti
    print('|> End DDCM Computational | Total time= %1.1f s' %(elapsed_time))

    return 0

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
#--- Run
_ = main_EX1_()












#domain = ComputationalDomain(nsd, meshfile='cook_memb.msh', flag_quads=False)

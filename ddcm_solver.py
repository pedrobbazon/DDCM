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
    print('----------------------------------------------------------')
    print('|::| Reference computational domain: OK')
    print('----------------------------------------------------------')
    
    # 2. Compute a reference solution to initially distribute the data
    
    #________________________________
    def mysource(x):
        return fd.Constant(1.0)
    def mysurfaceflux(x):
        return None
    def mybcvals(x):
        return fd.Constant(0.123456789)
    forcings = ForcingData(mysource, mysurfaceflux, mybcvals)
    #________________________________

    params = [Constant(1.0)]

    ref_sol = StdProblem(domain, params, 'Pure-Dirichlet', forcings, FLM_DarcyLinear, PoissonDualMixedHdiv, orders=[2,1])

    ref_sol.solve("myrefsol")
    print('----------------------------------------------------------')
    print('|::| Reference solution: OK')
    print('----------------------------------------------------------')

    # 3. Read or sample some synthetic data from a standard solution: 
    #    The idea is to produce some plausible data similar to what an experiment would deliver

    ddcm_data = DDCMData(domain)

    nsd, Ls, Ns = 2, [1.0, 0.2], [100, 20]
    synth_domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    # Let suppose we do some continuation to compute the data:
    bcvals = [3.0]
    for bcvs in bcvals:

        def mybcvals(x):
            return fd.Constant(bcvs)
        synth_forcings = ForcingData(None, None, mybcvals)
        
        params = [Constant(1.0)]
        
        synth_sol = StdProblem(synth_domain, params, 'Uniaxial', synth_forcings, FLM_DarcyLinear, PoissonDualMixedHdiv, orders=[2,1])
        synth_sol.solve("mysynthsol")

        ddcm_data.AppendSyntheticData(synth_sol)

    ddcm_data.Distribute(ref_sol)
    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution: OK')
    print('----------------------------------------------------------')

    '''
    # 4. Solve the Data Driven problem: This problem should share the same 
    #    forcings (bcs, sources, etc.) as the standard problem above

    ddcm_sol = DDCMProblem(ddcm_data, ddcm_formulation = 'NLSQP-5F-A')

    ddcm_sol.solve()

    print('----------------------------------------------------------')
    print('|> Data Driven solution OK')
    print('----------------------------------------------------------')
    '''
    
    elapsed_time = time.time() - ti
    print('|> End DDCM Computation | Total time= %1.1f s' %(elapsed_time))

    return 0

#----------------------------------------------------------------------------

def main_EX2_():

    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')
    
    # 1. Create (or import) the computational domain/mesh

    nsd, Ls, Ns = 2, [1.0, 1.0], [100, 100]

    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    print('----------------------------------------------------------')
    print('|::| Reference computational domain: OK')
    print('----------------------------------------------------------')
    
    # 2. Compute a reference solution to initially distribute the data
        
    #________________________________
    def mysource(x):
        return fd.Constant(100.0) # so as to activate the nonlinearities
    def mysurfaceflux(x):
        return None
    def mybcvals(x):
        return fd.Constant(0.123456789)
    forcings = ForcingData(mysource, mysurfaceflux, mybcvals)
    #________________________________

    def kappa_of_u(u):
       kappa0 = fd.Constant(1.0)
       viol = fd.Constant(25.0)
       kappa = kappa0*(1.0 + 10.0*0.5 * (1.0 + tanh(viol*(u-2.0))))
       return kappa
    params = [kappa_of_u]

    ref_sol = StdProblem(domain, params, 'Pure-Dirichlet', forcings, None, PoissonMixedHdiv, orders=[1,0])

    ref_sol.solve("myresults")

    print('----------------------------------------------------------')
    print('|::| Reference solution: OK')
    print('----------------------------------------------------------')
    
    elapsed_time = time.time() - ti
    print('|> End DDCM Computation | Total time= %1.1f s' %(elapsed_time))

    return 0

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
#--- Run
_ = main_EX1_()















#domain = ComputationalDomain(nsd, meshfile='cook_memb.msh', flag_quads=False)

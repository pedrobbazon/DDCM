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

def main_EX0_():

    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')
    
    print('----------------------------------------------------------')
    print('|::| Reference Domain and solution begins ...')
    print('---------------------------------------------')

    # 1. Create (or import) the computational domain/mesh and Compute a reference solution

    nsd, Ls, Ns = 2, [1.0, 1.0], [100, 100]
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

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

    ref_prob = StdProblem(domain, params, 'Pure-Dirichlet', forcings, FLM_DarcyLinear, PoissonDualMixedHdiv, orders=[2,1])

    ref_prob.solve("myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: OK')
    print('----------------------------------------------------------')

    # 3. Read or sample some synthetic data from a standard solution: 
    #    The idea is to produce some plausible data similar to what an experiment would deliver

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution begins ...')
    print('-------------------------------------------------------')

    ddcm_data = PoissonDDCMData(domain)

    nsd, Ls, Ns = 2, [1.0, 1.0], [100, 100]
    synth_domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    synth_forcings = ForcingData(mysource, mysurfaceflux, mybcvals)
        
    params = [fd.Constant(1.0)]
    synth_prob = StdProblem(synth_domain, params, 'Pure-Dirichlet', synth_forcings, FLM_DarcyLinear, PoissonDualMixedHdiv, orders=[2,1])
    synth_prob.solve("mysynthsol")

    # Sample the synthetic solution: Program this according to your problem

    one_sample_per_element = False
    if(one_sample_per_element == True): # One point per element centroid
        X = interpolate(domain.msh.coordinates, domain.VDG0)
        xs = fd.assemble(X).dat.data
    else: # Sample points one a regular grid
        Nsampx, Nsampy  = 2, 2
        Dx, Dy = Ls[0] / Nsampx, Ls[1] / Nsampy
        xls = np.linspace(0.5*Dx, Ls[0]-0.5*Dx, Nsampx)
        yls = np.linspace(0.5*Dy, Ls[1]-0.5*Dy, Nsampy)
        xlsg, ylsg = np.meshgrid(xls, yls)
        xs = np.dstack([xlsg, ylsg]).reshape(Nsampx*Nsampy, 2)

    #- Pay attention: this is a concious decision
    # and it may change in other situations
    stdsyn_sol = synth_prob.std_solution
    g = fd.project(stdsyn_sol.g, synth_domain.VDG0)
    s = fd.project(stdsyn_sol.s, synth_domain.VDG0)
    gradu_sampled_vals = np.array(g.at(xs))
    flux_sampled_vals = np.array(s.at(xs))
    #-------------------------------------------  
        
    # Apply any manipulation/transformation you need to define
    # the synthetic data (rotations, scalings, removing outliers, etc)   
    gradu_samples = gradu_sampled_vals #np.concatenate(gradu_samples)
    flux_samples  = flux_sampled_vals  #np.concatenate(flux_samples)

    ddcm_data.SetSampledData(gradu_samples, flux_samples)

    ddcm_data.Distribute(ref_prob.std_solution, 'Project-VDG0', flag_plot=True)

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution: OK')
    print('----------------------------------------------------------')

    # 4. Solve the Data Driven problem: This problem should share the same 
    #    forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g,gdata,s,sdata):
        return fd.inner(g-gdata, g-gdata)*dx + fd.inner(s-sdata, s-sdata)*dx

    ddcm_prob = DDCMProblem(ddcm_data, myObjFunc, None, 'Pure-Dirichlet', forcings, NLSP5FA, orders=[0,1])

    ddcm_prob.solve("myddcmsol")

    print('----------------------------------------------------------')
    print('|> Data Driven solution OK')
    print('----------------------------------------------------------')
    
    elapsed_time = time.time() - ti
    print('----------------------------------------------------------')
    print('|> End DDCM Computation | Total time= %1.1f s' %(elapsed_time))
    print('----------------------------------------------------------')

    return 0

#----------------------------------------------------------------------------

def main_EX1_():

    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')
    
    print('----------------------------------------------------------')
    print('|::| Reference Domain and solution begins ...')
    print('---------------------------------------------')

    # 1. Create (or import) the computational domain/mesh and Compute a reference solution

    nsd, Ls, Ns = 2, [1.0, 1.0], [100, 100]
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

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

    ref_prob = StdProblem(domain, params, 'Pure-Dirichlet', forcings, FLM_DarcyLinear, PoissonDualMixedHdiv, orders=[2,1])

    ref_prob.solve("myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: OK')
    print('----------------------------------------------------------')

    # 3. Read or sample some synthetic data from a standard solution: 
    #    The idea is to produce some plausible data similar to what an experiment would deliver

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution begins ...')
    print('-------------------------------------------------------')

    ddcm_data = PoissonDDCMData(domain)

    nsd, Ls, Ns = 2, [0.2, 1.0], [20, 100]
    synth_domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    # Let us suppose we do some continuation to compute the data:
    bcvals = [3.0]
    gradu_samples, flux_samples = [], []
    for bcvs in bcvals:

        def mysynbcvals(x):
            return fd.Constant(bcvs)
        synth_forcings = ForcingData(None, None, mysynbcvals)
        
        params = [fd.Constant(1.0)]
        
        synth_prob = StdProblem(synth_domain, params, 'Uniaxial', synth_forcings, FLM_DarcyLinear, PoissonDualMixedHdiv, orders=[2,1])
        synth_prob.solve("mysynthsol")

        # Sample the synthetic solution: Program this according to your problem
        Nsampx, Nsampy  = 1, 30
        Dx, Dy = Ls[0] / Nsampx, Ls[1] / Nsampy
        xls = np.linspace(0.5*Dx, Ls[0]-0.5*Dx, Nsampx)
        yls = np.linspace(0.5*Dy, Ls[1]-0.5*Dy, Nsampy)
        xlsg, ylsg = np.meshgrid(xls, yls)
        xs = np.dstack([xlsg, ylsg]).reshape(Nsampx*Nsampy, 2)
        std_sol = synth_prob.std_solution
        #- Pay attention: this is a concious decision
        # and it may change in other situations
        g = fd.project(std_sol.g, synth_domain.VDG0)
        s = fd.project(std_sol.s, synth_domain.VDG0)
        gradu_sampled_vals = np.array(g.at(xs))
        flux_sampled_vals = np.array(s.at(xs))
        #-------------------------------------------  
        
        # Apply any manipulation/transformation you need to define
        # the synthetic data (rotations, scalings, removing outliers, etc)

        def RotationMatrix(theta):
            return np.array([[np.cos(theta), -np.sin(theta)], 
                             [np.sin(theta), np.cos(theta)]])
        
        for angle in np.linspace(0.5*np.pi, 2*np.pi, 4):
            R = RotationMatrix(angle)
            gradu_rot_values = np.reshape([R.dot(vec) for vec in gradu_sampled_vals], (-1, 2))
            flux_rot_values = np.reshape([R.dot(vec) for vec in flux_sampled_vals], (-1, 2))
            gradu_samples.append(gradu_rot_values)
            flux_samples.append(flux_rot_values)
        gradu_samples = np.concatenate(gradu_samples)
        flux_samples = np.concatenate(flux_samples)

    ddcm_data.SetSampledData(gradu_samples, flux_samples)

    ddcm_data.Distribute(ref_prob.std_solution, 'Project-VDG0', flag_plot=True)

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution: OK')
    print('----------------------------------------------------------')

    # 4. Solve the Data Driven problem: This problem should share the same 
    #    forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g,gdata,s,sdata):
        return fd.inner(g-gdata, g-gdata)*dx + fd.inner(s-sdata, s-sdata)*dx

    ddcm_prob = DDCMProblem(ddcm_data, myObjFunc, None, 'Pure-Dirichlet', forcings, NLSP5FA, orders=[0,1])

    ddcm_prob.solve("myddcmsol")

    print('----------------------------------------------------------')
    print('|> Data Driven solution OK')
    print('----------------------------------------------------------')
    
    elapsed_time = time.time() - ti
    print('----------------------------------------------------------')
    print('|> End DDCM Computation | Total time= %1.1f s' %(elapsed_time))
    print('----------------------------------------------------------')

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
_ = main_EX0_()















#domain = ComputationalDomain(nsd, meshfile='cook_memb.msh', flag_quads=False)

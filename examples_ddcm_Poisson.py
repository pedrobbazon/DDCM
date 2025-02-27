#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on:         17th May, 2024
Last Modified on:   10th January, 2025

@authors: P. Balbao & R.F. Ausas
"""

#----------------------------------------------------------------------
# General libraries
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------
# Our libraries
from ddcm_classes import *
from ddcm_utils import *
from DDCMSolvers import *

#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# User defined main() functions: Program your options and problem below

# Example using the simple linear Darcy model to generate the reference solution and sample the data from itself
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
    forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals)
    #________________________________

    params = {'kappa': fd.Constant(1.0)}

    ref_prob = PoissonStdProblem(domain, params, 'Pure-Dirichlet', forcings, FLM_DarcyLinear, PoissonDualMixedHdiv, orders=[2,1])

    ref_prob.solve(0.0, "myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: OK')
    print('----------------------------------------------------------')

    # 2. Read or sample some synthetic data from a standard solution: 
    #    The idea is to produce some plausible data similar to what an experiment would deliver

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution begins ...')
    print('-------------------------------------------------------')

    # Note in this case, I've made the data associated to the same domain (and mesh)
    # we used for the reference solution, but this doesn't have to be the case, it can be a finer mesh
    ddcm_data = PoissonDDCMData(domain)

    nsd, Ls, Ns = 2, [1.0, 1.0], [100, 100]
    synth_domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    synth_forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals)
        
    params = {'kappa': fd.Constant(1.0)}
    synth_prob = PoissonStdProblem(synth_domain, params, 'Pure-Dirichlet', synth_forcings, FLM_DarcyLinear, PoissonDualMixedHdiv, orders=[2,1])
    synth_prob.solve(0.0, "mysynthsol")

    # Sample the synthetic solution: Program this according to your problem

    one_sample_per_element = False
    if(one_sample_per_element == True): # One point per element centroid
        X = interpolate(domain.msh.coordinates, domain.VDG0)
        xs = fd.assemble(X).dat.data
    else: # Sample points one a regular grid
        Nsampx, Nsampy  = 5, 5
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
    gradu_samples = gradu_sampled_vals
    flux_samples  = flux_sampled_vals

    ddcm_data.SetSampledData(gradu_samples, flux_samples, add_origin=True, data_file='myDATA.txt')

    ddcm_data.Distribute(ref_prob.std_solution, 'Project-VDG0', flag_plot=True)

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution: %d data points' %(ddcm_data.sampled_grad.shape[0]))
    print('----------------------------------------------------------')

    # 3. Solve the Data Driven problem: This problem should share the same 
    #    forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g,gdata,s,sdata):
        return fd.inner(g-gdata, g-gdata)*fd.dx + fd.inner(s-sdata, s-sdata)*fd.dx

    ddcm_prob = DDCMProblem(ddcm_data, myObjFunc, None, 'Pure-Dirichlet', forcings, NLSP5FD, orders=[1,0])

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

# Example using the simple linear Darcy model to generate the reference solution and sample the data from itself
# using the proximity measure with the modulus
def main_EX10_():

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
    forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals)
    #________________________________

    params = {'kappa': fd.Constant(1.0)}

    ref_prob = PoissonStdProblem(domain, params, 'Pure-Dirichlet', forcings, FLM_DarcyLinear, PoissonDualMixedHdiv, orders=[2,1])

    ref_prob.solve(0.0, "myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: OK')
    print('----------------------------------------------------------')

    # 2. Read or sample some synthetic data from a standard solution: 
    #    The idea is to produce some plausible data similar to what an experiment would deliver

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution begins ...')
    print('-------------------------------------------------------')

    # Note in this case, I've made the data associated to the same domain (and mesh)
    # we used for the reference solution, but this doesn't have to be the case, it can be a finer mesh
    ddcm_data = PoissonDDCMData(domain)

    nsd, Ls, Ns = 2, [1.0, 1.0], [100, 100]
    synth_domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    synth_forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals)
        
    params = {'kappa': fd.Constant(1.0)}
    synth_prob = PoissonStdProblem(synth_domain, params, 'Pure-Dirichlet', synth_forcings, FLM_DarcyLinear, PoissonDualMixedHdiv, orders=[2,1])
    synth_prob.solve(0.0, "mysynthsol")

    # Sample the synthetic solution: Program this according to your problem

    one_sample_per_element = True
    if(one_sample_per_element == True): # One point per element centroid
        X = interpolate(domain.msh.coordinates, domain.VDG0)
        xs = fd.assemble(X).dat.data
    else: # Sample points one a regular grid
        Nsampx, Nsampy  = 5, 5
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
    gradu_samples = gradu_sampled_vals
    flux_samples  = flux_sampled_vals

    ddcm_data.SetSampledData(gradu_samples, flux_samples, add_origin=True, data_file='myDATA.txt')

    ddcm_data.Distribute(ref_prob.std_solution, 'Project-VDG0', flag_plot=True)

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution: %d data points' %(ddcm_data.sampled_grad.shape[0]))
    print('----------------------------------------------------------')

    # 3. Solve the Data Driven problem: This problem should share the same 
    #    forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g,gdata,s,sdata):
        eta = fd.Constant(0.1)
        #modgreg = eta*(sqrt(1.0 + fd.inner(g,g)/eta**2) - 1.0)
        #modsreg = eta*(sqrt(1.0 + fd.inner(s,s)/eta**2) - 1.0)
        #modgdata = sqrt(fd.inner(gdata,gdata))
        #modsdata = sqrt(fd.inner(sdata,sdata))
        modgreg = eta*(fd.sqrt(1.0 + fd.inner(g-gdata,g-gdata)/eta**2) - 1.0)
        modsreg = eta*(fd.sqrt(1.0 + fd.inner(s-sdata,s-sdata)/eta**2) - 1.0)
        return (modgreg + modsreg)*fd.dx
        #return 0.6*((modgreg-modgdata)**2 + (modsreg-modsdata)**2)*fd.dx(degree=4) + 0.4*(fd.inner(g-gdata, g-gdata) + fd.inner(s-sdata, s-sdata))*fd.dx

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

# Example using the simple linear Darcy model to generate the synthetic data
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
        return fd.Constant(100.0)
    def mysurfaceflux(x):
        return None
    def mybcvals(x):
        return fd.Constant(0.123456789)
    forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals)
    #________________________________

    params = {'kappa': fd.Constant(1.0)}

    ref_prob = PoissonStdProblem(domain, params, 'Pure-Dirichlet', forcings, FLM_DarcyLinear, PoissonDualMixedHdiv, orders=[2,1])

    ref_prob.solve(0.0, "myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: OK')
    print('----------------------------------------------------------')

    # 2. Read or sample some synthetic data from a standard solution: 
    #    The idea is to produce some plausible data similar to what an experiment would deliver

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution begins ...')
    print('-------------------------------------------------------')

    # Note in this case, I've made the data associated to the same domain (and mesh)
    # we used for the reference solution, but this doesn't have to be the case, it can be a finer mesh
    ddcm_data = PoissonDDCMData(domain)

    nsd, Ls, Ns = 2, [0.2, 1.0], [1, 5]
    synth_domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    # Let us suppose we do some continuation to compute the data:
    Nbc, Nth = 10, 10
    Nsampx, Nsampy  = 1, 1
    bcvals = np.linspace(1.0+1.e-15, 34.0, Nbc)
    gradu_samples = np.empty(shape=(Nbc*Nth*Nsampx*Nsampy,2), dtype=np.float64)
    flux_samples = np.empty(shape=(Nbc*Nth*Nsampx*Nsampy,2), dtype=np.float64)
    count = 0
    for bcvs in bcvals:

        def mysynbcvals(x):
            return fd.Constant(bcvs)
        synth_forcings = PoissonForcingData(None, None, mysynbcvals)
        
        params = {'kappa': fd.Constant(1.0)}
        synth_prob = PoissonStdProblem(synth_domain, params, 'Uniaxial', synth_forcings, FLM_DarcyLinear, PoissonDualMixedHdiv, orders=[2,1])
        synth_prob.solve(bcvs, "mysynthsol")

        # Sample the synthetic solution: Program this according to your problem
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
        gradu_sampled_vals = np.array(g.at(xs))[0]
        flux_sampled_vals = np.array(s.at(xs))[0]
        #-------------------------------------------  
        
        # Apply any manipulation/transformation you need to define
        # the synthetic data (rotations, scalings, removing outliers, etc)
        for angle in np.linspace(0.0, 2.*np.pi, Nth, endpoint=False):
            ct, st = np.cos(angle), np.sin(angle)
            gradu_samples[count,0] = ct * gradu_sampled_vals[0] - st * gradu_sampled_vals[1]
            gradu_samples[count,1] = st * gradu_sampled_vals[0] + ct * gradu_sampled_vals[1]
            flux_samples[count,0] = ct * flux_sampled_vals[0] - st * flux_sampled_vals[1]
            flux_samples[count,1] = st * flux_sampled_vals[0] + ct * flux_sampled_vals[1]
            count += 1
        #gradu_samples, flux_samples = RotateData2D(0.0, 2.0*np.pi, 4, False, 
        #                                           gradu_sampled_vals, flux_sampled_vals, gradu_samples, flux_samples)

    ddcm_data.SetSampledData(gradu_samples, flux_samples, add_origin=True, data_file='myDATA.txt')

    ddcm_data.Distribute(ref_prob.std_solution, 'Project-VDG0', flag_plot=True)

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution: %d data points' %(ddcm_data.sampled_grad.shape[0]))
    print('----------------------------------------------------------')

    # 3. Solve the Data Driven problem: This problem should share the same 
    #    forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g,gdata,s,sdata):
        return fd.inner(g-gdata, g-gdata)*fd.dx + fd.inner(s-sdata, s-sdata)*fd.dx

    ddcm_prob = DDCMProblem(ddcm_data, myObjFunc, None, 'Pure-Dirichlet', forcings, NLSP5FD, orders=[1,0])

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

# Example using k(u) nonlinear model (invented by me) to generate the synthetic data
def main_EX2_():

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
        return fd.Constant(100.0) # so as to activate the nonlinearities
    def mysurfaceflux(x):
        return None
    def mybcvals(x):
        return fd.Constant(0.123456789)
    forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals)
    #________________________________

    def kappa_of_u(u):
       kappa0 = fd.Constant(1.0)
       viol = fd.Constant(25.0)
       kappa = kappa0*(1.0 + 10.0*0.5 * (1.0 + fd.tanh(viol*(u-2.0))))
       return kappa
    params = [kappa_of_u]

    ref_prob = PoissonStdProblem(domain, params, 'Pure-Dirichlet', forcings, None, PoissonMixedHdiv, orders=[1,0])

    ref_prob.solve(0.0, "myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: OK')
    print('----------------------------------------------------------')

    # 2. Read or sample some synthetic data from a standard solution: 
    #    The idea is to produce some plausible data similar to what an experiment would deliver

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution begins ...')
    print('-------------------------------------------------------')

    # Note in this case, I've made the data associated to the same domain (and mesh)
    # we used for the reference solution, but this doesn't have to be the case, it can be a finer mesh
    ddcm_data = PoissonDDCMData(domain)

    nsd, Ls, Ns = 2, [0.2, 1.0], [20, 100]
    synth_domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    # Let us suppose we do some continuation to compute the data:
    Nbc, Nth = 4, 4
    Nsampx, Nsampy  = 1, 10
    bcvals = np.linspace(2.0, 8.0, Nbc)
    gradu_samples = np.empty(shape=(Nbc*Nth*Nsampx*Nsampy,2), dtype=np.float64)
    flux_samples = np.empty(shape=(Nbc*Nth*Nsampx*Nsampy,2), dtype=np.float64)
    count = 0
    for bcvs in bcvals:

        print('----')
        print('|||| B.C. Val on top wall= %1.2f' %(bcvs))
        def mysynbcvals(x):
            return fd.Constant(bcvs)
        synth_forcings = PoissonForcingData(None, None, mysynbcvals)
        
        params = [kappa_of_u]
        
        synth_prob = PoissonStdProblem(synth_domain, params, 'Uniaxial', synth_forcings, None, PoissonMixedHdiv, orders=[2,1])
        synth_prob.solve(bcvs, "mysynthsol")

        # Sample the synthetic solution: Program this according to your problem
        Dx, Dy = Ls[0] / Nsampx, Ls[1] / Nsampy
        xls = np.linspace(0.5*Dx, Ls[0]-0.5*Dx, Nsampx)
        yls = np.linspace(0.5*Dy, Ls[1]-0.5*Dy, Nsampy)
        xlsg, ylsg = np.meshgrid(xls, yls)
        xs = np.dstack([xlsg, ylsg]).reshape(Nsampx*Nsampy, 2)
        std_sol = synth_prob.std_solution
        # Pay attention: this is a concious decision
        # and it may change in other situations
        g = fd.project(std_sol.g, synth_domain.VDG0)
        s = fd.project(std_sol.s, synth_domain.VDG0)
        gradu_sampled_vals = np.array(g.at(xs))
        flux_sampled_vals = np.array(s.at(xs))
        #-------------------------------------------  
        
        # Apply any manipulation/transformation you need to define
        # the synthetic data (rotations, scalings, removing outliers, etc)
        for angle in np.linspace(0.0, 2.*np.pi, Nth, endpoint=False):
            ct, st = np.cos(angle), np.sin(angle)
            for s in range(Nsampx*Nsampy):
                gradu_samples[count,0] = ct * gradu_sampled_vals[s,0] - st * gradu_sampled_vals[s,1]
                gradu_samples[count,1] = st * gradu_sampled_vals[s,0] + ct * gradu_sampled_vals[s,1]
                flux_samples[count,0] = ct * flux_sampled_vals[s,0] - st * flux_sampled_vals[s,1]
                flux_samples[count,1] = st * flux_sampled_vals[s,0] + ct * flux_sampled_vals[s,1]
                count += 1
    
    ddcm_data.SetSampledData(gradu_samples, flux_samples, add_origin=True, data_file='myDATA.txt')

    ddcm_data.Distribute(ref_prob.std_solution, 'Project-VDG0', flag_plot=True)
    
    print('--------------------------------------------------------------')
    print('|::| Synthetic data computation/distribution: %d data points' %(ddcm_data.sampled_grad.shape[0]))
    print('--------------------------------------------------------------')

    # 3. Solve the Data Driven problem: This problem should share the same 
    #    forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g,gdata,s,sdata):
        return fd.inner(g-gdata, g-gdata)*fd.dx + fd.inner(s-sdata, s-sdata)*fd.dx

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

# Example using Forcheimer's nonlinear model to generate the synthetic data
def main_EX3_():

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
    forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals)
    #________________________________

    params = {'eta': fd.Constant(1.e-12), 'mu_over_kappa': fd.Constant(1.0), 'beta_x_rho': fd.Constant(10.0)}
    ref_prob = PoissonStdProblem(domain, params, 'Pure-Dirichlet', forcings, FLM_ForcheimerReg, PoissonDualMixedHdiv, orders=[2,1])

    ref_prob.solve(0.0, "myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: OK')
    print('----------------------------------------------------------')

    # 2. Read or sample some synthetic data from a standard solution: 
    #    The idea is to produce some plausible data similar to what an experiment would deliver

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution begins ...')
    print('-------------------------------------------------------')

    # Note in this case, I've made the data associated to the same domain (and mesh)
    # we used for the reference solution, but this doesn't have to be the case, it can be a finer mesh
    ddcm_data = PoissonDDCMData(domain)

    nsd, Ls, Ns = 2, [0.2, 1.0], [2, 5]
    synth_domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    # Let us suppose we do some continuation to compute the data:
    Nbc, Nth = 15, 5
    bcvals = np.linspace(1.0+1.e-15, 14.0, Nbc)
    gradu_samples = np.empty(shape=(Nbc*Nth,2), dtype=np.float64)
    flux_samples = np.empty(shape=(Nbc*Nth,2), dtype=np.float64)
    count = 0
    for bcvs in bcvals:

        print('----')
        print('|||| B.C. Val on top wall= %1.2f' %(bcvs))
        def mysynbcvals(x):
            return fd.Constant(bcvs)
        synth_forcings = PoissonForcingData(None, None, mysynbcvals)
        
        # They could be the same parameters as in the reference solution or we may add some perturbation
        params = {'eta': fd.Constant(1.e-12), 'mu_over_kappa': fd.Constant(1.0), 'beta_x_rho': fd.Constant(10.0)}
        synth_prob = PoissonStdProblem(synth_domain, params, 'Uniaxial', synth_forcings, FLM_ForcheimerReg, PoissonDualMixedHdiv, orders=[2,1])
        synth_prob.solve(bcvs, "mysynthsol")

        # Sample the synthetic solution: Program this according to your problem
        Nsampx, Nsampy  = 1, 1
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
        gradu_sampled_vals = np.array(g.at(xs))[0]
        flux_sampled_vals = np.array(s.at(xs))[0]
        #-------------------------------------------  
        
        # Apply any manipulation/transformation you need to define
        # the synthetic data (rotations, scalings, removing outliers, etc)
        for angle in np.linspace(0.0, 2.*np.pi, Nth, endpoint=False):
            ct, st = np.cos(angle), np.sin(angle)
            gradu_samples[count,0] = ct * gradu_sampled_vals[0] - st * gradu_sampled_vals[1]
            gradu_samples[count,1] = st * gradu_sampled_vals[0] + ct * gradu_sampled_vals[1]
            flux_samples[count,0] = ct * flux_sampled_vals[0] - st * flux_sampled_vals[1]
            flux_samples[count,1] = st * flux_sampled_vals[0] + ct * flux_sampled_vals[1]
            count += 1

    ddcm_data.SetSampledData(gradu_samples, flux_samples, add_origin=False, data_file='myDATA.txt')

    noise_params = {'type': 'gaussian', 'mean': 0.0, 'std': 0.03}
    ddcm_data.AddNoisetoData(noise_params, data_file='myDATA.txt')

    ddcm_data.Distribute(ref_prob.std_solution, 'Project-VDG0', flag_plot=True)

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution: %d data points' %(ddcm_data.sampled_grad.shape[0]))
    print('----------------------------------------------------------')

    # 3. Solve the Data Driven problem: This problem should share the same 
    #    forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g,gdata,s,sdata):
        return fd.inner(g-gdata, g-gdata)*fd.dx + fd.inner(s-sdata, s-sdata)*fd.dx

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

# Example using Forcheimer's nonlinear model to generate the synthetic data
def main_EX31_():

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
    forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals)
    #________________________________

    params = {'eta': fd.Constant(1.e-12), 'mu_over_kappa': fd.Constant(1.0), 'beta_x_rho': fd.Constant(10.0)}
    ref_prob = PoissonStdProblem(domain, params, 'Pure-Dirichlet', forcings, FLM_ForcheimerReg, PoissonDualMixedHdiv, orders=[2,1])

    ref_prob.solve(0.0, "myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: OK')
    print('----------------------------------------------------------')

    # 2. Read or sample some synthetic data from a standard solution: 
    #    The idea is to produce some plausible data similar to what an experiment would deliver

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution begins ...')
    print('-------------------------------------------------------')

    # Note in this case, I've made the data associated to the same domain (and mesh)
    # we used for the reference solution, but this doesn't have to be the case, it can be a finer mesh
    ddcm_data = PoissonDDCMData(domain)

    nsd, Ls, Ns = 2, [0.2, 1.0], [2, 5]
    synth_domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    # Let us suppose we do some continuation to compute the data:
    Nbc, Nth = 15, 1
    bcvals = np.linspace(0.0+1.e-15, 1.4, Nbc)
    gradu_samples = np.empty(shape=(Nbc*Nth,2), dtype=np.float64)
    flux_samples = np.empty(shape=(Nbc*Nth,2), dtype=np.float64)
    count = 0
    for bcvs in bcvals:

        print('----')
        print('|||| B.C. Val on top wall= %1.2f' %(bcvs))
        def mysynbcvals(x):
            return fd.Constant(bcvs)
        synth_forcings = PoissonForcingData(None, None, mysynbcvals)
        
        # They could be the same parameters as in the reference solution or we may add some perturbation
        params = {'eta': fd.Constant(1.e-12), 'mu_over_kappa': fd.Constant(1.0), 'beta_x_rho': fd.Constant(10.0)}
        synth_prob = PoissonStdProblem(synth_domain, params, 'Uniaxial', synth_forcings, FLM_ForcheimerReg, PoissonDualMixedHdiv, orders=[2,1])
        synth_prob.solve(bcvs, "mysynthsol")

        # Sample the synthetic solution: Program this according to your problem
        Nsampx, Nsampy  = 1, 1
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
        gradu_sampled_vals = np.array(g.at(xs))[0]
        flux_sampled_vals = np.array(s.at(xs))[0]
        #-------------------------------------------  
        
        # Apply any manipulation/transformation you need to define
        # the synthetic data (rotations, scalings, removing outliers, etc)
        for angle in np.linspace(0.0, 2.*np.pi, Nth, endpoint=False):
            ct, st = np.cos(angle), np.sin(angle)
            gradu_samples[count,0] = ct * gradu_sampled_vals[0] - st * gradu_sampled_vals[1]
            gradu_samples[count,1] = st * gradu_sampled_vals[0] + ct * gradu_sampled_vals[1]
            flux_samples[count,0] = ct * flux_sampled_vals[0] - st * flux_sampled_vals[1]
            flux_samples[count,1] = st * flux_sampled_vals[0] + ct * flux_sampled_vals[1]
            count += 1

    ddcm_data.SetSampledData(gradu_samples, flux_samples, add_origin=False, data_file='myDATA.txt')

    noise_params = {'type': 'gaussian', 'magnitude': 0.075, 'mean': 0.0, 'std': 1.e-10}
    ddcm_data.AddNoisetoData(noise_params, data_file='myDATA.txt')

    Nth = 4
    gradu_samples_new = np.empty(shape=(Nbc*Nth,2), dtype=np.float64)
    flux_samples_new = np.empty(shape=(Nbc*Nth,2), dtype=np.float64)
    auxgrad = ddcm_data.sampled_grad
    auxflux = ddcm_data.sampled_flux
    count = 0
    for ibc, bcvs in enumerate(bcvals):
        for angle in np.linspace(0.0, 2.*np.pi, Nth, endpoint=False):
            ct, st = np.cos(angle), np.sin(angle)
            gradu_samples_new[count,0] = ct * auxgrad[ibc,0] - st * auxgrad[ibc,1]
            gradu_samples_new[count,1] = st * auxgrad[ibc,0] + ct * auxgrad[ibc,1]
            flux_samples_new[count,0] = ct * auxflux[ibc,0] - st * auxflux[ibc,1]
            flux_samples_new[count,1] = st * auxflux[ibc,0] + ct * auxflux[ibc,1]
            count += 1
    
    ddcm_data.SetSampledData(gradu_samples_new, flux_samples_new, add_origin=False, data_file='myDATA.txt')

    ddcm_data.Distribute(ref_prob.std_solution, 'Project-VDG0', flag_plot=True)

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution: %d data points' %(ddcm_data.sampled_grad.shape[0]))
    print('----------------------------------------------------------')

    # 3. Solve the Data Driven problem: This problem should share the same 
    #    forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g,gdata,s,sdata):
        return fd.inner(g-gdata, g-gdata)*fd.dx + fd.inner(s-sdata, s-sdata)*fd.dx

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

# Use data set from file and use adjust coefficients in Forcheimers model from previous run
def main_EX32_():

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
    forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals)
    #________________________________

    #params = {'eta': fd.Constant(1.e-12), 'mu_over_kappa': fd.Constant(1.0), 'beta_x_rho': fd.Constant(10.0)}
    params = {'eta': fd.Constant(1.e-12), 'mu_over_kappa': fd.Constant(1.19722865), 'beta_x_rho': fd.Constant(8.44782321)}
    ref_prob = PoissonStdProblem(domain, params, 'Pure-Dirichlet', forcings, FLM_ForcheimerReg, PoissonDualMixedHdiv, orders=[2,1])

    ref_prob.solve(0.0, "myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: OK')
    print('----------------------------------------------------------')

    # 2. Read or sample some synthetic data from a standard solution: 
    #    The idea is to produce some plausible data similar to what an experiment would deliver

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution begins ...')
    print('-------------------------------------------------------')

    # Note in this case, I've made the data associated to the same domain (and mesh)
    # we used for the reference solution, but this doesn't have to be the case, it can be a finer mesh
    ddcm_data = PoissonDDCMData(domain)
 
    ddcm_data.ReadDataFromFile(datafile='myDATA_Ntheta4_with_noise.txt')

    ddcm_data.Distribute(ref_prob.std_solution, 'Project-VDG0', flag_plot=True)

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution: %d data points' %(ddcm_data.sampled_grad.shape[0]))
    print('----------------------------------------------------------')

    # 3. Solve the Data Driven problem: This problem should share the same 
    #    forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g,gdata,s,sdata):
        return fd.inner(g-gdata, g-gdata)*fd.dx + fd.inner(s-sdata, s-sdata)*fd.dx

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

# Use data set from file and adjust coefficients in linear Darcy model from previous run
def main_EX33_():

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
    forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals)
    #________________________________

    def kappa(u):
        return fd.Constant(1.0/3.04528688)
    params = {'kappa': kappa}
    ref_prob = PoissonStdProblem(domain, params, 'Pure-Dirichlet', forcings, FLM_DarcyLinear, PoissonMixedHdiv, orders=[1,0])

    ref_prob.solve(0.0, "myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: OK')
    print('----------------------------------------------------------')

    # 2. Read or sample some synthetic data from a standard solution: 
    #    The idea is to produce some plausible data similar to what an experiment would deliver

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution begins ...')
    print('-------------------------------------------------------')

    # Note in this case, I've made the data associated to the same domain (and mesh)
    # we used for the reference solution, but this doesn't have to be the case, it can be a finer mesh
    ddcm_data = PoissonDDCMData(domain)
 
    ddcm_data.ReadDataFromFile(datafile='myDATA_Ntheta4_with_noise.txt')

    ddcm_data.Distribute(ref_prob.std_solution, 'Project-VDG0', flag_plot=True)

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution: %d data points' %(ddcm_data.sampled_grad.shape[0]))
    print('----------------------------------------------------------')

    # 3. Solve the Data Driven problem: This problem should share the same 
    #    forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g,gdata,s,sdata):
        return fd.inner(g-gdata, g-gdata)*fd.dx + fd.inner(s-sdata, s-sdata)*fd.dx

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

#----------------------------------------------------------------------------

# Example of a reference solution based on an analytical expression
def main_EX4_():

    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')
    
    print('----------------------------------------------------------')
    print('|::| Reference Domain and solution begins ...')
    print('---------------------------------------------')

    # 1. Create (or import) the computational domain/mesh and Compute a reference solution

    nsd, Ls, Ns = 2, [1.0, 1.0], [200, 200]
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    #________________________________
    # These would be the ones corresponding to the analytical solution we propose

    def uexact(x):
        return fd.cos(fd.pi*x[0])*fd.cos(fd.pi*x[1])
    def sexact(x):
        kappa = fd.Constant(1.0)
        return -kappa * fd.grad(uexact(x))

    def mysource(x):
        return fd.div(-kappa*fd.grad(uexact(x)))
    def mysurfaceflux(x):
        return sexact(x)
    def mybcvals(x):
        return uexact(x)
    forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals)
    #________________________________

    # This will only be called for visualization purposes
    params = {'exact-potential': uexact, 'exact-flux': sexact}
    ref_prob = PoissonStdProblem(domain, params, 'Pure-Dirichlet', forcings, None, PoissonAnalytical, orders=[3,3])

    ref_prob.solve(0.0, "myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: OK')
    print('----------------------------------------------------------')

    # 2. Read or sample some synthetic data from a standard solution: 
    #    The idea is to produce some plausible data similar to what an experiment would deliver

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution begins ...')
    print('-------------------------------------------------------')

    nsd, Ls, Ns = 2, [1.0, 1.0], [50, 50]
    hsize = max(Ls[0]/Ns[0], Ls[1]/Ns[1])
    ddcm_domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)
    ddcm_data = PoissonDDCMData(ddcm_domain)

    # In this manufactured example, we create by hand fluxes and grads 
    # without actually solving a synthetic problem:
    ti = time.time()
    Nbc, Nth = 3, 4
    bcvals = np.linspace(1.0+1.e-15, 3.2, Nbc)
    gradu_samples = np.empty(shape=(Nbc*Nth,2), dtype=np.float64)
    flux_samples = np.empty(shape=(Nbc*Nth,2), dtype=np.float64)
    count = 0
    for bcvs in bcvals:
        # Generate a flux s and a grad g 
        g1 = 0.0
        s1 = 0.0
        L = 1.0
        g2 = (bcvs - 1.0)/L
        kappa = 1.0
        s2 = -kappa*g2
        gradu_sampled_vals = np.array([g1, g2])
        flux_sampled_vals  = np.array([s1, s2])
        #-------------------------------------------  
        
        # Apply any manipulation/transformation you need to define
        # the synthetic data (rotations, scalings, removing outliers, etc)
        # This implementation is much faster
        for angle in np.linspace(0.0, 2.*np.pi, Nth, endpoint=False):
            ct, st = np.cos(angle), np.sin(angle)
            gradu_samples[count,0] = ct * gradu_sampled_vals[0] - st * gradu_sampled_vals[1]
            gradu_samples[count,1] = st * gradu_sampled_vals[0] + ct * gradu_sampled_vals[1]
            flux_samples[count,0] = ct * flux_sampled_vals[0] - st * flux_sampled_vals[1]
            flux_samples[count,1] = st * flux_sampled_vals[0] + ct * flux_sampled_vals[1]
            count += 1

    print(' . Bcs x Rotate= ', time.time() - ti)

    ddcm_data.SetSampledData(gradu_samples, flux_samples, add_origin=True, data_file=None)

    ddcm_data.Distribute(ref_prob.std_solution, 'Interpolate-VDG0', flag_plot=True)

    npoints_data = ddcm_data.sampled_grad.shape[0]
    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution: %d data points' %(npoints_data))
    print('----------------------------------------------------------')

    # 3. Solve the Data Driven problem: This problem should share the same 
    #    forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g,gdata,s,sdata):
        return fd.inner(g-gdata, g-gdata)*fd.dx + fd.inner(s-sdata, s-sdata)*fd.dx

    ddcm_prob = DDCMProblem(ddcm_data, myObjFunc, None, 'Pure-Dirichlet', forcings, NLSP5FC, orders=[1,0])

    ddcm_prob.solve("myddcmsol")

    # 4. Compute some error norms
    
    print('   ... Computing some error norms ...')
    x = fd.SpatialCoordinate(ddcm_data.domain.msh)
    L2error_u = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u , norm_type='L2')
    H1error_u = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u , norm_type='H1')
    L2error_s = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s , norm_type='L2')
    L2error_lmd = fd.errornorm(fd.Constant(0), ddcm_prob.ddcm_solution.lmd, norm_type='L2')
    L2error_mu = fd.errornorm(fd.as_vector([fd.Constant(0),fd.Constant(0)]), ddcm_prob.ddcm_solution.mu, norm_type='L2')

    print('----------------------------------------------------------')
    print('|> Data Driven solution OK')
    print('----------------------------------------------------------')
    
    elapsed_time = time.time() - ti
    print('----------------------------------------------------------')
    print('|> End DDCM Computation | Total time= %1.1f s' %(elapsed_time))
    print('----------------------------------------------------------')

    return npoints_data, hsize, L2error_u, H1error_u, L2error_s, L2error_lmd, L2error_mu

# Example of a refence solution based on an analytical expression with anisotropic K
def main_EX5_():

    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')
    
    print('----------------------------------------------------------')
    print('|::| Reference Domain and solution begins ...')
    print('---------------------------------------------')

    # 1. Create (or import) the computational domain/mesh and Compute a reference solution

    nsd, Ls, Ns = 2, [1.0, 1.0], [100, 100]
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    #________________________________
    # These would be the ones corresponding to the analytical solution we propose

    def uexact(x):
        return fd.cos(fd.pi*x[0])*fd.cos(fd.pi*x[1])
    
    kappa_xx = 1.0
    kappa_xy = 0.0
    kappa_yy = 10.0
    kappa = fd.as_tensor([[kappa_xx, kappa_xy], [kappa_xy, kappa_yy]])

    def sexact(x):
        return -kappa * fd.grad(uexact(x))

    def mysource(x):
        return fd.div(-kappa*fd.grad(uexact(x)))
    def mysurfaceflux(x):
        return sexact(x)
    def mybcvals(x):
        return uexact(x)
    forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals)
    #________________________________

    # This will only be called for visualization purposes
    params = {'exact-potential': uexact, 'exact-flux': sexact}
    ref_prob = PoissonStdProblem(domain, params, 'Pure-Dirichlet', forcings, None, PoissonAnalytical, orders=[3,3])

    ref_prob.solve(0.0, "myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: OK')
    print('----------------------------------------------------------')

    # 2. Read or sample some synthetic data from a standard solution: 
    #    The idea is to produce some plausible data similar to what an experiment would deliver

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution begins ...')
    print('-------------------------------------------------------')

    nsd, Ls, Ns = 2, [1.0, 1.0], [50, 50]
    hsize = max(Ls[0]/Ns[0], Ls[1]/Ns[1])
    ddcm_domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)
    ddcm_data = PoissonDDCMData(ddcm_domain)

    # In this manufactured example, we create by hand fluxes and grads 
    # without actually solving a synthetic problem:
    ti = time.time()
    Nbc, Nth = 400, 800
    bcvals = np.linspace(1.0+1.e-15, 12.0, Nbc)
    gradu_samples = np.empty(shape=(Nbc*Nth,2), dtype=np.float64)
    flux_samples = np.empty(shape=(Nbc*Nth,2), dtype=np.float64)
    count = 0
    for bcvs in bcvals:
        # Generate a flux s and a grad g 
        g1 = 0.0
        L = 1.0
        g2 = (bcvs - 1.0)/L
        gradu_sampled_vals = np.array([g1, g2])
        #-------------------------------------------  
        
        # Apply any manipulation/transformation you need to define
        # the synthetic data (rotations, scalings, removing outliers, etc)
        # This implementation is much faster
        for angle in np.linspace(0.0, 2.*np.pi, Nth, endpoint=False):
            ct, st = np.cos(angle), np.sin(angle)
            gradu_samples[count,0] = ct * gradu_sampled_vals[0] - st * gradu_sampled_vals[1]
            gradu_samples[count,1] = st * gradu_sampled_vals[0] + ct * gradu_sampled_vals[1]
            
            flux_samples[count,0] = -(kappa_xx*gradu_samples[count,0] + kappa_xy*gradu_samples[count,1])
            flux_samples[count,1] = -(kappa_xy*gradu_samples[count,0] + kappa_yy*gradu_samples[count,1])
            count += 1

    print(' . Bcs x Rotate= ', time.time() - ti)

    ddcm_data.SetSampledData(gradu_samples, flux_samples, add_origin=True, data_file=None)

    ddcm_data.Distribute(ref_prob.std_solution, 'Interpolate-VDG0', flag_plot=True)

    npoints_data = ddcm_data.sampled_grad.shape[0]
    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution: %d data points' %(npoints_data))
    print('----------------------------------------------------------')

    # 3. Solve the Data Driven problem: This problem should share the same 
    #    forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g,gdata,s,sdata):
        return fd.inner(g-gdata, g-gdata)*fd.dx + fd.inner(s-sdata, s-sdata)*fd.dx

    ddcm_prob = DDCMProblem(ddcm_data, myObjFunc, None, 'Pure-Dirichlet', forcings, NLSP5FA, orders=[0,1])

    ddcm_prob.solve("myddcmsol")

    # 4. Compute some error norms
    
    print('   ... Computing some error norms ...')
    x = fd.SpatialCoordinate(ddcm_data.domain.msh)
    L2error_u = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u , norm_type='L2')
    H1error_u = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u , norm_type='H1')
    L2error_s = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s , norm_type='L2')

    print('----------------------------------------------------------')
    print('|> Data Driven solution OK')
    print('----------------------------------------------------------')
    
    elapsed_time = time.time() - ti
    print('----------------------------------------------------------')
    print('|> End DDCM Computation | Total time= %1.1f s' %(elapsed_time))
    print('----------------------------------------------------------')

    return npoints_data, hsize, L2error_u, H1error_u, L2error_s

# Convergence study for the Forcheimer's model based on an analytical solution
def main_EX50_(N=20):

    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')
    
    print('----------------------------------------------------------')
    print('|::| Reference Domain and solution begins ...')
    print('---------------------------------------------')

    # 1. Create (or import) the computational domain/mesh and Compute a reference solution

    nsd, Ls, Ns = 2, [1.0, 1.0], [N, N]
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    #________________________________
    # These would be the ones corresponding to the analytical solution we propose

    def uexact(x):
        beta = fd.Constant(10.0)
        mu = fd.Constant(1.0)
        kappa = fd.Constant(1.0)
        return -beta*(x[0]**2*fd.sqrt(x[0]**2 + x[1]**2)/3 + x[1]**2*fd.sqrt(x[0]**2 + x[1]**2)/3) - mu*x[0]**2/(2*kappa) - mu*x[1]**2/(2*kappa)

    def sexact(x):
        return fd.as_vector([x[0], x[1]])
    
    def gradexact(x):
        beta = fd.Constant(10.0)
        mu = fd.Constant(1.0)
        kappa = fd.Constant(1.0)
        return fd.as_vector([-beta*x[0]*fd.sqrt(x[0]**2 + x[1]**2) - mu*x[0]/kappa,
                             -beta*x[0]*fd.sqrt(x[0]**2 + x[1]**2) - mu*x[1]/kappa])

    def mysource(x):
        return fd.Constant(2.0)
    def mysurfaceflux(x):
        return sexact(x)
    def mybcvals(x):
        return uexact(x)
    forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals)
    #________________________________

    # This will only be called for visualization purposes
    params = {'eta': fd.Constant(1.e-6), 'mu_over_kappa': fd.Constant(1.0), 'beta_x_rho': fd.Constant(10.0)}
    ref_prob = PoissonStdProblem(domain, params, 'Pure-Dirichlet', forcings, FLM_ForcheimerReg, PoissonDualMixedHdiv, orders=[2,1])

    ref_prob.solve(0.0, "myrefsol")

    # 4. Compute some error norms
    
    print('   ... Computing some error norms ...')
    x = fd.SpatialCoordinate(domain.msh)
    L2error_u = fd.errornorm(uexact(x), ref_prob.std_solution.u , norm_type='L2')
    H1error_u = fd.errornorm(uexact(x), ref_prob.std_solution.u , norm_type='H1')
    L2error_s = fd.errornorm(sexact(x), ref_prob.std_solution.s , norm_type='L2')
    hsize = max(Ls[0]/Ns[0], Ls[1]/Ns[1])

    return hsize, L2error_u, H1error_u, L2error_s

#----------------------------------------------------------------------------
# Example to illustrate in the paper the workflow of the data distribution
# considering Forcheimer's model with anisotropic K
def main_EX51_():

    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')

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
    forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals)
    #________________________________

    vals_kyy = [1.0, 5.0, 10.0, 20.0]
    for val in vals_kyy:
        kappa_xx = 1.0
        kappa_yy = val
        muokappa = fd.as_tensor([[fd.Constant(1/kappa_xx), fd.Constant(0.0)], [fd.Constant(0.0), fd.Constant(1/kappa_yy)]])
        params = {'eta': fd.Constant(1.e-10), 'mu_over_kappa': muokappa, 'beta_x_rho': fd.Constant(1.0)}
        ref_prob = PoissonStdProblem(domain, params, 'Pure-Dirichlet', forcings, FLM_ForcheimerRegAni, PoissonDualMixedHdivAni, orders=[2,1])
        #params = {'kappa': fd.as_tensor([[kappa_xx, 0.0], [0.0, kappa_yy]])}
        #ref_prob = PoissonStdProblem(domain, params, 'Pure-Dirichlet', forcings, None, PoissonMixedHdivAni, orders=[2,1])

        ref_prob.solve(val, "myrefsol")
        x = fd.SpatialCoordinate(domain.msh)
        tot_source = fd.assemble(mysource(x) * fd.dx(domain=domain.msh))
        n = fd.FacetNormal(domain.msh)
        tot_inout = fd.assemble(fd.dot(ref_prob.std_solution.s,n)*fd.ds)
        print(tot_source, tot_inout)

    raise Exception('jkjfjfjfjfjfjfjfjfjfjfjfj')

    # 1. The idea is to produce some plausible data similar to what an experiment would deliver

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution begins ...')
    print('-------------------------------------------------------')

    # Note in this case, I've made the data associated to the same domain (and mesh)
    # we used for the reference solution, but this doesn't have to be the case, it can be a finer mesh
    ddcm_data = PoissonDDCMData(domain)

    nsd, Ls, Ns = 2, [0.2, 1.0], [2, 5]
    synth_domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    # Let us suppose we do some continuation to compute the data:
    Nbc, Nth = 10, 10
    bcvals = np.linspace(1.0+1.e-15, 12.0, Nbc)
    gradu_samples = np.empty(shape=(Nbc*Nth,2), dtype=np.float64)
    flux_samples = np.empty(shape=(Nbc*Nth,2), dtype=np.float64)
    count = 0
    for bcvs in bcvals:

        print('----')
        print('|||| B.C. Val on top wall= %1.2f' %(bcvs))
        def mysynbcvals(x):
            return fd.Constant(bcvs)
        synth_forcings = PoissonForcingData(None, None, mysynbcvals)
        
        # They could be the same parameters as in the reference solution or we may add some perturbation
        params = {'eta': fd.Constant(1.e-6), 'mu_over_kappa': fd.Constant(1.0), 'beta_x_rho': fd.Constant(10.0)}
        synth_prob = PoissonStdProblem(synth_domain, params, 'Uniaxial', synth_forcings, FLM_ForcheimerReg, PoissonDualMixedHdiv, orders=[2,1])
        synth_prob.solve(bcvs, "mysynthsol")

        # Sample the synthetic solution: Program this according to your problem
        Nsampx, Nsampy  = 1, 1
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
        gradu_sampled_vals = np.array(g.at(xs))[0]
        flux_sampled_vals = np.array(s.at(xs))[0]

    
    print('----------------------------------------------------------')
    print('|::| Reference Domain and solution begins ...')
    print('---------------------------------------------')

    # 1. Create (or import) the computational domain/mesh and Compute a reference solution

    nsd, Ls, Ns = 2, [1.0, 1.0], [100, 100]
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    return 0

#------------------------------------------------------------------------------
# Example to illustrate in the paper the workflow of the data distribution
def main_EX6_():

    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')

    # 1. Create (or import) the computational domain/mesh and Compute a reference solution

    nsd, Ls, Ns = 2, [1.0, 1.0], [100, 100]
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    #________________________________
    def mysource(x):
        return fd.Constant(100.0)
    def mysurfaceflux(x):
        return None
    def mybcvals(x):
        return fd.Constant(0.123456789)
    forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals)
    #________________________________

    params = {'eta': fd.Constant(1.e-6), 'mu_over_kappa': fd.Constant(1.0), 'beta_x_rho': fd.Constant(100.0)}
    ref_prob = PoissonStdProblem(domain, params, 'Pure-Dirichlet', forcings, FLM_ForcheimerReg, PoissonDualMixedHdiv, orders=[2,1])

    ref_prob.solve(0.0, "myrefsol")

    # 1. The idea is to produce some plausible data similar to what an experiment would deliver

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution begins ...')
    print('-------------------------------------------------------')

    # Note in this case, I've made the data associated to the same domain (and mesh)
    # we used for the reference solution, but this doesn't have to be the case, it can be a finer mesh
    ddcm_data = PoissonDDCMData(domain)

    nsd, Ls, Ns = 2, [0.2, 1.0], [2, 5]
    synth_domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    # Let us suppose we do some continuation to compute the data:
    Nbc, Nth = 10, 10
    bcvals = np.linspace(1.0+1.e-15, 12.0, Nbc)
    gradu_samples = np.empty(shape=(Nbc*Nth,2), dtype=np.float64)
    flux_samples = np.empty(shape=(Nbc*Nth,2), dtype=np.float64)
    count = 0
    for bcvs in bcvals:

        print('----')
        print('|||| B.C. Val on top wall= %1.2f' %(bcvs))
        def mysynbcvals(x):
            return fd.Constant(bcvs)
        synth_forcings = PoissonForcingData(None, None, mysynbcvals)
        
        # They could be the same parameters as in the reference solution or we may add some perturbation
        params = {'eta': fd.Constant(1.e-6), 'mu_over_kappa': fd.Constant(1.0), 'beta_x_rho': fd.Constant(10.0)}
        synth_prob = PoissonStdProblem(synth_domain, params, 'Uniaxial', synth_forcings, FLM_ForcheimerReg, PoissonDualMixedHdiv, orders=[2,1])
        synth_prob.solve(bcvs, "mysynthsol")

        # Sample the synthetic solution: Program this according to your problem
        Nsampx, Nsampy  = 1, 1
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
        gradu_sampled_vals = np.array(g.at(xs))[0]
        flux_sampled_vals = np.array(s.at(xs))[0]

    
    print('----------------------------------------------------------')
    print('|::| Reference Domain and solution begins ...')
    print('---------------------------------------------')

    # 1. Create (or import) the computational domain/mesh and Compute a reference solution

    nsd, Ls, Ns = 2, [1.0, 1.0], [100, 100]
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    return 0

#----------------------------------------------------------------------------

# Example adding a nonlinear reaction term to induze coupling between the equations
def main_EX7_():

    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')
    
    print('----------------------------------------------------------')
    print('|::| Reference Domain and solution begins ...')
    print('---------------------------------------------')

    # 1. Create (or import) the computational domain/mesh and Compute a reference solution

    mymeshfile, tags = CreateMesh3Inclusions('inclusions')

    nsd, Ls, Ns = 2, [1.0, 1.0], [100, 100]
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=mymeshfile, flag_quads=False)

    # The following lines define a piecewise constant function for the reaction coefficient
    Qh = fd.FunctionSpace(domain.msh, "DG", degree=0)
    source = fd.TrialFunction(Qh)
    vp = fd.TestFunction(Qh)
    ap = source*vp*fd.dx
    sigmaA, sigmaB = 0.0, 10.0
    ninclusions = 3
    background_marker = tags['background_marker']
    inclusion_marker = tags['inclusion_marker']
    Lp = fd.Constant(sigmaA)*vp*fd.dx(background_marker)
    Lp += fd.Constant(sigmaB)*vp*fd.dx(inclusion_marker+0)
    Lp += fd.Constant(sigmaB)*vp*fd.dx(inclusion_marker+1)
    Lp += fd.Constant(sigmaB)*vp*fd.dx(inclusion_marker+2)

    sourceh = fd.Function(Qh, name='sigma')
    opts={"ksp_type": "preonly", "pc_type": "lu"}
    fd.solve(ap==Lp, sourceh, bcs=[], solver_parameters=opts)
    VTKFile("source.pvd").write(sourceh)
    
    #________________________________
    def mysource(x):
        return sourceh # so as to activate the nonlinearities
    def mysurfaceflux(x):
        return fd.Constant(0.0)
    def mybcvals(x):
        return fd.Constant(0.123456789)
    forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals)
    #________________________________

    def kappa(u):
       kappa0 = fd.Constant(1.0)
       return kappa0
        
    def reaction(u): # Linear case with discontinuous sigma
        return fd.Constant(sigmaB)*u
    def dreaction(u): # Linear case with discontinuous sigma
        return fd.Constant(sigmaB)
    
    '''
    def reaction(u): # Quadratic case
       return fd.Constant(sigmaB)*u**2
    def dreaction(u):
       return 2.0*fd.Constant(sigmaB)*u
    '''
    
    '''
    def reaction(u): # Exponencial
       beta = fd.Constant(0.4)
       return fd.Constant(sigmaB)*fd.exp(-beta*u)
    def dreaction(u): # Exponencial
       beta = fd.Constant(0.4)
       return fd.Constant(sigmaB)*beta*fd.exp(-beta*u)    
    '''
    '''
    def reaction(u): # Logistic
       sigma0 = fd.Constant(1.0)
       beta = fd.Constant(1.0)
       kappa = sigma0*u*(1.0 - u/beta)
       return kappa
    def reaction(u): # Michaelis
       sigma0 = fd.Constant(1.0)
       beta = fd.Constant(1.0)
       return sigma0*u/(beta + u)
    '''

    params = {'kappa': kappa, 'reaction': reaction}
    #ref_prob = PoissonStdProblem(domain, params, 'Pure-Neumann', forcings, None, PoissonMixedHdiv, orders=[1,0])
    ref_prob = PoissonStdProblem(domain, params, 'Pure-Neumann', forcings, None, PoissonPrimal, orders=[1])

    ref_prob.solve(0.0, "myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: OK')
    print('----------------------------------------------------------')

    # 2. Read or sample some synthetic data from a standard solution: 
    #    The idea is to produce some plausible data similar to what an experiment would deliver

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution begins ...')
    print('-------------------------------------------------------')

    # Note in this case, I've made the data associated to the same domain (and mesh)
    # we used for the reference solution, but this doesn't have to be the case, it can be a finer mesh
    ddcm_data = PoissonDDCMData(domain)

    nsd, Ls, Ns = 2, [0.2, 1.0], [2, 10]
    synth_domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    # Let us suppose we do some continuation to compute the data:
    Nbc, Nth = 10, 10
    Nsampx, Nsampy  = 1, 1
    bcvals = np.linspace(1.0, 8.0, Nbc)
    gradu_samples = np.empty(shape=(Nbc*Nth*Nsampx*Nsampy,2), dtype=np.float64)
    flux_samples = np.empty(shape=(Nbc*Nth*Nsampx*Nsampy,2), dtype=np.float64)
    count = 0
    for bcvs in bcvals:

        print('----')
        print('|||| B.C. Val on top wall= %1.2f' %(bcvs))
        def mysynbcvals(x):
            return fd.Constant(bcvs)
        synth_forcings = PoissonForcingData(None, None, mysynbcvals)
        
        params = {'kappa': kappa, 'reaction': None}
        synth_prob = PoissonStdProblem(synth_domain, params, 'Uniaxial', synth_forcings, None, PoissonMixedHdiv, orders=[2,1])
        synth_prob.solve(bcvs, "mysynthsol")

        # Sample the synthetic solution: Program this according to your problem
        Dx, Dy = Ls[0] / Nsampx, Ls[1] / Nsampy
        xls = np.linspace(0.5*Dx, Ls[0]-0.5*Dx, Nsampx)
        yls = np.linspace(0.5*Dy, Ls[1]-0.5*Dy, Nsampy)
        xlsg, ylsg = np.meshgrid(xls, yls)
        xs = np.dstack([xlsg, ylsg]).reshape(Nsampx*Nsampy, 2)
        std_sol = synth_prob.std_solution
        # Pay attention: this is a concious decision
        # and it may change in other situations
        g = fd.project(std_sol.g, synth_domain.VDG0)
        s = fd.project(std_sol.s, synth_domain.VDG0)
        gradu_sampled_vals = np.array(g.at(xs))
        flux_sampled_vals = np.array(s.at(xs))
        #-------------------------------------------  
        
        # Apply any manipulation/transformation you need to define
        # the synthetic data (rotations, scalings, removing outliers, etc)
        for angle in np.linspace(0.0, 2.*np.pi, Nth, endpoint=False):
            ct, st = np.cos(angle), np.sin(angle)
            for s in range(Nsampx*Nsampy):
                gradu_samples[count,0] = ct * gradu_sampled_vals[s,0] - st * gradu_sampled_vals[s,1]
                gradu_samples[count,1] = st * gradu_sampled_vals[s,0] + ct * gradu_sampled_vals[s,1]
                flux_samples[count,0] = ct * flux_sampled_vals[s,0] - st * flux_sampled_vals[s,1]
                flux_samples[count,1] = st * flux_sampled_vals[s,0] + ct * flux_sampled_vals[s,1]
                count += 1
    
    ddcm_data.SetSampledData(gradu_samples, flux_samples, add_origin=True, data_file='myDATA.txt')

    ddcm_data.Distribute(ref_prob.std_solution, 'Project-VDG0', flag_plot=True)
    
    print('--------------------------------------------------------------')
    print('|::| Synthetic data computation/distribution: %d data points' %(ddcm_data.sampled_grad.shape[0]))
    print('--------------------------------------------------------------')

    # 3. Solve the Data Driven problem: This problem should share the same 
    #    forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g,gdata,s,sdata):
        return fd.inner(g-gdata, g-gdata)*fd.dx + fd.inner(s-sdata, s-sdata)*fd.dx

    ddcm_params = {'reaction': reaction, 'dreaction': dreaction}
    ddcm_prob = DDCMProblem(ddcm_data, myObjFunc, None, 'Pure-Neumann', forcings, NLSP5FA, orders=[0,1], params=ddcm_params)

    ddcm_prob.solve("myddcmsol")

    print('----------------------------------------------------------')
    print('|> Data Driven solution OK')
    print('----------------------------------------------------------')
    
    elapsed_time = time.time() - ti
    print('----------------------------------------------------------')
    print('|> End DDCM Computation | Total time= %1.1f s' %(elapsed_time))
    print('----------------------------------------------------------')

    return 0

# Example adding a nonlinear reaction term to induze coupling between the equations
def main_EX8_():

    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')
    
    print('----------------------------------------------------------')
    print('|::| Reference Domain and solution begins ...')
    print('---------------------------------------------')

    # 1. Create (or import) the computational domain/mesh and Compute a reference solution

    case_corner = 1
    mymeshfile, tags = CreateMeshReentrantCorner('reentrant', case_corner)
    if(case_corner == 1):
        omega = 1.75*fd.pi
    elif(case_corner == 2):
        omega = 1.5*fd.pi
    elif(case_corner == 3):
        omega = 1.25*fd.pi

    nsd, Ls, Ns = 2, [1.0, 1.0], [100, 100]
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=mymeshfile, flag_quads=False)
    
    #________________________________
    def mysource(x):
        return fd.Constant(0.0)
    def mysurfaceflux(x):
        return fd.Constant(0.0)
    def mybcvals(x):
        theta = fd.conditional(fd.gt(fd.atan2(x[1]+1e-12, x[0]+1e-12),0), fd.atan2(x[1]+1e-12, x[0]+1e-12), fd.atan2(x[1]+1e-12, x[0]+1e-12) + 2*fd.pi)
        r = fd.sqrt(x[0]**2 + x[1]**2)
        alpha = fd.pi/omega
        return (r**alpha)*fd.sin(alpha*theta)

    forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals)
    #________________________________

    def kappa(u):
       kappa0 = fd.Constant(1.0)
       return kappa0

    params = {'kappa': kappa, 'reaction': None}
    #ref_prob = PoissonStdProblem(domain, params, 'Pure-Dirichlet', forcings, None, PoissonMixedHdiv, orders=[1,0])
    ref_prob = PoissonStdProblem(domain, params, 'Pure-Dirichlet', forcings, None, PoissonPrimal, orders=[1])

    ref_prob.solve(0.0, "myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: OK')
    print('----------------------------------------------------------')

    # 2. Read or sample some synthetic data from a standard solution: 
    #    The idea is to produce some plausible data similar to what an experiment would deliver

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution begins ...')
    print('-------------------------------------------------------')

    # Note in this case, I've made the data associated to the same domain (and mesh)
    # we used for the reference solution, but this doesn't have to be the case, it can be a finer mesh
    ddcm_data = PoissonDDCMData(domain)
    
    one_sample_per_element = False
    if(one_sample_per_element == True): # One point per element centroid
        synth_domain = ComputationalDomain(nsd, Ls, Ns, meshfile=mymeshfile, flag_quads=False)
        synth_prob = PoissonStdProblem(synth_domain, params, 'Pure-Dirichlet', forcings, None, PoissonPrimal, orders=[1])
        synth_prob.solve(0.0, "mysynthsol")
        X = interpolate(domain.msh.coordinates, domain.VDG0)
        xs = fd.assemble(X).dat.data
        stdsyn_sol = synth_prob.std_solution
        g = fd.project(stdsyn_sol.g, synth_domain.VDG0)
        s = fd.project(stdsyn_sol.s, synth_domain.VDG0)
        gradu_samples = np.array(g.at(xs))
        flux_samples = np.array(s.at(xs))
    else:
        nsd, Ls, Ns = 2, [0.2, 1.0], [2, 4]
        synth_domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

        # Let us suppose we do some continuation to compute the data:
        Nbc, Nth = 20, 10
        Nsampx, Nsampy  = 1, 1
        bcvals = 1.0 + np.logspace(np.log10(0.5),np.log10(20),Nbc)
        gradu_samples = np.empty(shape=(Nbc*Nth*Nsampx*Nsampy,2), dtype=np.float64)
        flux_samples = np.empty(shape=(Nbc*Nth*Nsampx*Nsampy,2), dtype=np.float64)
        count = 0
        for bcvs in bcvals:

            print('----')
            print('|||| B.C. Val on top wall= %1.2f' %(bcvs))
            def mysynbcvals(x):
                return fd.Constant(bcvs)
            synth_forcings = PoissonForcingData(None, None, mysynbcvals)
            
            params = {'kappa': kappa, 'reaction': None}
            synth_prob = PoissonStdProblem(synth_domain, params, 'Uniaxial', synth_forcings, None, PoissonMixedHdiv, orders=[2,1])
            synth_prob.solve(bcvs, "mysynthsol")

            # Sample the synthetic solution: Program this according to your problem
            Dx, Dy = Ls[0] / Nsampx, Ls[1] / Nsampy
            xls = np.linspace(0.5*Dx, Ls[0]-0.5*Dx, Nsampx)
            yls = np.linspace(0.5*Dy, Ls[1]-0.5*Dy, Nsampy)
            xlsg, ylsg = np.meshgrid(xls, yls)
            xs = np.dstack([xlsg, ylsg]).reshape(Nsampx*Nsampy, 2)
            std_sol = synth_prob.std_solution
            # Pay attention: this is a concious decision
            # and it may change in other situations
            g = fd.project(std_sol.g, synth_domain.VDG0)
            s = fd.project(std_sol.s, synth_domain.VDG0)
            gradu_sampled_vals = np.array(g.at(xs))
            flux_sampled_vals = np.array(s.at(xs))
            #-------------------------------------------  
            
            # Apply any manipulation/transformation you need to define
            # the synthetic data (rotations, scalings, removing outliers, etc)
            for angle in np.linspace(0.0, 2.*np.pi, Nth, endpoint=False):
                ct, st = np.cos(angle), np.sin(angle)
                for s in range(Nsampx*Nsampy):
                    gradu_samples[count,0] = ct * gradu_sampled_vals[s,0] - st * gradu_sampled_vals[s,1]
                    gradu_samples[count,1] = st * gradu_sampled_vals[s,0] + ct * gradu_sampled_vals[s,1]
                    flux_samples[count,0] = ct * flux_sampled_vals[s,0] - st * flux_sampled_vals[s,1]
                    flux_samples[count,1] = st * flux_sampled_vals[s,0] + ct * flux_sampled_vals[s,1]
                    count += 1
    
    ddcm_data.SetSampledData(gradu_samples, flux_samples, add_origin=True, data_file='myDATA.txt')

    ddcm_data.Distribute(ref_prob.std_solution, 'Project-VDG0', flag_plot=True)
    
    print('--------------------------------------------------------------')
    print('|::| Synthetic data computation/distribution: %d data points' %(ddcm_data.sampled_grad.shape[0]))
    print('--------------------------------------------------------------')

    # 3. Solve the Data Driven problem: This problem should share the same 
    #    forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g,gdata,s,sdata):
        return fd.inner(g-gdata, g-gdata)*fd.dx + fd.inner(s-sdata, s-sdata)*fd.dx

    ddcm_params = {'reaction': None, 'dreaction': None}
    ddcm_prob = DDCMProblem(ddcm_data, myObjFunc, None, 'Pure-Dirichlet', forcings, NLSP5FA, orders=[0,1], params=ddcm_params)

    ddcm_prob.solve("myddcmsol")

    print('----------------------------------------------------------')
    print('|> Data Driven solution OK')
    print('----------------------------------------------------------')
    
    elapsed_time = time.time() - ti
    print('----------------------------------------------------------')
    print('|> End DDCM Computation | Total time= %1.1f s' %(elapsed_time))
    print('----------------------------------------------------------')

    return 0

# ===== NEW FORMULATION ===== #

# Example of a reference solution based on an analytical expression
def main_EX90_(formulation='Natural', N=20):

    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')
    
    print('----------------------------------------------------------')
    print('|::| Reference Domain and solution begins ...')
    print('---------------------------------------------')

    # 1. Create (or import) the computational domain/mesh and Compute a reference solution

    nsd, Ls, Ns = 2, [1.0, 1.0], [N, N]
    hsize = max(Ls[0]/Ns[0], Ls[1]/Ns[1])
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    #________________________________
    # These would be the ones corresponding to the analytical solution we propose

    h = fd.CellDiameter(domain.msh)

    params = {'perm': fd.Constant(1.0), 'zeda': fd.Constant(-10.0), 
              'theta': fd.Constant(0.0), 'kappa': fd.Constant(1.0), 
              'alpha': fd.Constant(0.0), 'beta': fd.Constant(0.0), 
              'gamma': fd.Constant(0.0), 'eta': fd.Constant(0.0),
              'ls': h, 'lmu': h, 'formulation': formulation} 
    
    
    # Propose an manufactured potential field and its associated gradient
    def uexact(x):
        return fd.cos(fd.pi*x[0])*fd.cos(fd.pi*x[1]) 
    def eexact(x):
        return fd.grad(uexact(x))
    # Propose an analytical expression for lambda
    def lmbdexact(x):
        x0, y0  = fd.Constant(0.5), fd.Constant(0.5)
        A = fd.Constant(1.0)
        return fd.exp(A*((x[0]-x0)**2 + (x[1]-y0)**2))
    
    # Propose a flux-gradient relationship based on the dataset  
    def etilde(x):
        A = fd.Constant(0.2)         # Amplitude
        ome = fd.Constant(4.0)       # Frequency
        phi = fd.Constant(0.0)       # Phase angle 
        return fd.as_vector([eexact(x)[0] + A*fd.sin(ome*fd.pi*x[0] + phi),
                             eexact(x)[1] + A*fd.sin(ome*fd.pi*x[1] + phi)])
    def stilde(x):
        A = fd.Constant(0.1)    # Amplitude
        ome = fd.Constant(3.0)  # Frequency
        phi = fd.Constant(0.0)  # Phase angle
        eps = fd.Constant(1.0e-10)
        met = fd.sqrt(etilde(x)[0]**2 + etilde(x)[1]**2) + eps
        etn = fd.as_vector([etilde(x)[0]/met, etilde(x)[1]/met])
        mst = met + A*fd.sin(ome*fd.pi*met)
        return fd.as_vector([-mst*etn[0], -mst*etn[1]])
    
    # def etilde(x):
    #     A = fd.Constant(0.01)        # Amplitude
    #     f = fd.Constant(2.0)         # Frequency
    #     phi = fd.Constant(0.0)       # Phase angle 
    #     return fd.as_vector([eexact(x)[0] + A*fd.sin(f*fd.pi*x[0] + phi),
    #                         eexact(x)[1] + A*fd.sin(f*fd.pi*x[1] + phi)])
    # def stilde(x):
    #     A = fd.Constant(0.01)    # Amplitude
    #     f = fd.Constant(3.0)     # Frequency
    #     phi = fd.Constant(0.0)   # Phase angle 
    #     return fd.as_vector([-eexact(x)[0] + A*fd.sin(f*fd.pi*x[0] + phi),
    #                          -eexact(x)[1] + A*fd.sin(f*fd.pi*x[1] + phi)])
    
    # by eq.(22)
    def muexact(x):
        return etilde(x) - eexact(x)
    # by eq.(21)
    def fdual(x):
        zeda = params['zeda']
        return zeda*lmbdexact(x) + fd.div(muexact(x))
    # by eq.(23)
    def sexact(x):
        kappa = params['kappa']
        return stilde(x) + (1/kappa)*fd.grad(lmbdexact(x))
    # by eq.(24)
    def qexact(x):
        zeda = params['zeda']
        return zeda*uexact(x) + fd.div(sexact(x))
    
    def mysource(x):
        return qexact(x)
    def mysurfaceflux(x):
        return sexact(x)
    def mybcvals(x):
        return uexact(x)
    def mylmbd_boundary_vals(x):
        return lmbdexact(x)
    def mymu_boundary_vals(x):
        return muexact(x)
    def mydual_source(x):
        return fdual(x)
            
    forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals, mylmbd_boundary_vals, mymu_boundary_vals, mydual_source)
    #________________________________

    # This will only be called for visualization purposes
    params_anal = {'exact-potential': uexact, 'exact-flux': sexact, 'exact-grad': eexact, 'exact-lmbd' : lmbdexact, 
                   'exact-mu': muexact, 'exact-etilde': etilde, 'exact-stilde': stilde}
    ref_prob = PoissonStdProblem(domain, params_anal, 'Irrelevant', forcings, None, PoissonAnalytical, orders=[3,3])

    ref_prob.solve(0.0, "myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: OK')
    print('----------------------------------------------------------')

    # 2. Read or sample some synthetic data from a standard solution: 
    #    The idea is to produce some plausible data similar to what an experiment would deliver

    # The data will be sampled from the analytical expresion on-the-fly
    ddcm_data = PoissonDDCMData(domain, Analytical=True, fluxex=stilde, gradex=etilde)

    # 3. Solve the Data Driven problem: This problem should share the same 
    #    forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g,gdata,s,sdata):
        kappa = params['kappa']
        return 0.5*fd.inner(g-gdata, g-gdata)*fd.dx + 0.5*kappa*fd.inner(s-sdata, s-sdata)*fd.dx

    if(formulation == 'Natural'):
        ords = [0,1]
    elif(formulation == 'Equal-Order'):
        ords = [1,1]
    ddcm_prob = DDCMProblem(ddcm_data, myObjFunc, None, 'Mixed-Dirichlet-Neumann', forcings, DDCMStab, orders=ords, params=params)

    ddcm_prob.solve("myddcmsol")

    # 4. Compute some error norms
    
    print('... Computing some error norms ...')
    x = fd.SpatialCoordinate(ddcm_data.domain.msh)
    L2error_u = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u , norm_type = 'L2')
    H1error_u = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u , norm_type = 'H1')
    H1error_lmd = fd.errornorm(lmbdexact(x), ddcm_prob.ddcm_solution.lmd, norm_type = 'H1')
    L2error_s = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s , norm_type = 'L2')
    L2error_e = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g , norm_type = 'L2')
    L2error_mu = fd.errornorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type = 'L2')
    if formulation == 'Equal-Order':
        Hdiverror_s = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s , norm_type = 'Hdiv')
        Hdiverror_e = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g , norm_type = 'Hdiv')
        Hdiverror_mu = fd.errornorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type = 'Hdiv')
    else:
        Hdiverror_s = np.nan
        Hdiverror_e = np.nan
        Hdiverror_mu = np.nan

    print('----------------------------------------------------------')
    print('|> Data Driven solution OK')
    print('----------------------------------------------------------')
    
    elapsed_time = time.time() - ti
    print('----------------------------------------------------------')
    print('|> End DDCM Computation | Total time= %1.1f s' %(elapsed_time))
    print('----------------------------------------------------------')

    return hsize, L2error_u, H1error_u, H1error_lmd, L2error_s, L2error_e, L2error_mu, Hdiverror_s, Hdiverror_e, Hdiverror_mu

#  Example of a convergence study for the Forcheimer's model based on an analytical solution
def main_EX91_(formulation='Natural', N=20):
    
    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')
    
    print('----------------------------------------------------------')
    print('|::| Reference Domain and solution begins ...')
    print('---------------------------------------------')

    # 1. Create (or import) the computational domain/mesh and Compute a reference solution

    nsd, Ls, Ns = 2, [1.0, 1.0], [N, N]
    hsize = max(Ls[0]/Ns[0], Ls[1]/Ns[1])
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    #________________________________
    # These would be the ones corresponding to the analytical solution we propose
    h = fd.CellDiameter(domain.msh)

    # perm : permeability tensor; visc: dynamic viscosity; zeda:
    # forch_coef: Forcheimer coefficient; zeda: reaction coefficient
    params = {'perm': fd.Constant(1.0), 'visc' : fd.Constant(1.0),
              'forch_coef': fd.Constant(1.0), 'zeda': fd.Constant(1.0), 
              'theta': fd.Constant(0.5), 'kappa': fd.Constant(1.0), 
              'alpha': fd.Constant(0.125), 'beta': fd.Constant(0.5), 
              'gamma': fd.Constant(0.125), 'eta': fd.Constant(0.5),
              'ls': h, 'lmu': h, 'formulation': formulation} 

    
    def uexact(x):
        forch_coef = params['forch_coef']
        visc = params['visc']
        perm = params['perm']
        return - visc/(2*perm)*(x[0]**2 + x[1]**2) - (forch_coef/3)*(x[0]**2 + x[1]**2)**(3/2) 
    
    def eexact(x):
        forch_coef = params['forch_coef']
        visc = params['visc']
        perm = params['perm']
        return fd.as_vector([-forch_coef*x[0]*fd.sqrt(x[0]**2 + x[1]**2) - visc*x[0]/perm,
                             -forch_coef*x[1]*fd.sqrt(x[0]**2 + x[1]**2) - visc*x[1]/perm])
    
    # Propose an analytical expression for lambda
    def lmbdexact(x):
        x0, y0  = fd.Constant(0.5), fd.Constant(0.5)
        A = fd.Constant(1.0)
        return fd.exp(A*((x[0]-x0)**2 + (x[1]-y0)**2))
    
    # def sexact(x):
        # return fd.as_vector([x[0], x[1]])
    
    # Propose a flux-gradient relationship based on the dataset
    sref = lambda x: fd.as_vector([x[0], x[1]])      
    def stilde(x):
        A = fd.Constant(0.1)    # Amplitude
        f = fd.Constant(1.0)     # Frequency
        phi = fd.Constant(0.0)   # Phase angle 
        return fd.as_vector([sref(x)[0] + A*fd.sin(f*fd.pi*sref(x)[0] + phi),
                            sref(x)[1] + A*fd.sin(f*fd.pi*sref(x)[1] + phi)])  
    def etilde(x):
        A = fd.Constant(0.1)        # Amplitude
        f = fd.Constant(1.0)         # Frequency
        phi = fd.Constant(0.0)       # Phase angle 
        visc = params['visc']        # Viscosity
        perm = params['perm']        # Permeability
        forch_coef = params['forch_coef']   # Forcheimer's coefficient 
        # Regularization of (s*s)^(1/2)
        epsilon = 1e-2
        modfluxreg = epsilon*(fd.sqrt(1.0 + fd.inner(stilde(x),stilde(x))/epsilon**2) - 1.0)
        
        return fd.as_vector([-visc*perm*stilde(x)[0] - forch_coef*modfluxreg*stilde(x)[0] + A*fd.sin(f*fd.pi*x[0] + phi),
                            -visc*perm*stilde(x)[1] - forch_coef*modfluxreg*stilde(x)[1] + A*fd.sin(f*fd.pi*x[1] + phi)])
     # by eq.(22)
    def muexact(x):
        return etilde(x) - eexact(x)
    # by eq.(21)
    def fdual(x):
        zeda = params['zeda']
        return zeda*lmbdexact(x) + fd.div(muexact(x))
    # by eq.(23)
    def sexact(x):
        kappa = params['kappa']
        return stilde(x) + (1/kappa)*fd.grad(lmbdexact(x))
    # by eq.(24)
    def qexact(x):
        zeda = params['zeda']
        return zeda*uexact(x) + fd.div(sexact(x))
    
    def mysource(x):
        return qexact(x)
    def mysurfaceflux(x):
        return sexact(x)
    def mybcvals(x):
        return uexact(x)
    def mylmbd_boundary_vals(x):
        return lmbdexact(x)
    def mymu_boundary_vals(x):
        return muexact(x)
    def mydual_source(x):
        return fdual(x)
            
    forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals, mylmbd_boundary_vals, mymu_boundary_vals, mydual_source)
      
    # def mysource(x):
    #     zeda = params['zeda']
    #     return fd.div(sexact(x)) + zeda*uexact(x)
    # def mysurfaceflux(x):
    #     return sexact(x)
    # def mybcvals(x):
    #     return uexact(x)
    # forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals)
    #________________________________

    
    # This will only be called for visualization purposes
    params_anal = {'exact-potential': uexact, 'exact-flux': sexact, 'exact-grad': eexact, 'exact-lmbd' : lmbdexact, 
                   'exact-mu': muexact, 'exact-etilde': etilde, 'exact-stilde': stilde}
    ref_prob = PoissonStdProblem(domain, params_anal, 'Pure-Dirichlet', forcings, None, PoissonAnalytical, orders=[3,3])

    ref_prob.solve(0.0, "myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: OK')
    print('----------------------------------------------------------')

    # 2. Read or sample some synthetic data from a standard solution: 
    #    The idea is to produce some plausible data similar to what an experiment would deliver

    # The data will be sampled from the analytical expresion on-the-fly
    ddcm_data = PoissonDDCMData(domain, Analytical=True, fluxex=stilde, gradex=etilde)

    # 3. Solve the Data Driven problem: This problem should share the same 
    #    forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g,gdata,s,sdata):
        kappa = params['kappa'] 
        return (0.5)*fd.inner(g-gdata, g-gdata)*fd.dx + (0.5*kappa)*fd.inner(s-sdata, s-sdata)*fd.dx

    if(formulation == 'Natural'):
        ords = [0,1]
    elif(formulation == 'Equal-Order'):
        ords = [1,1]
    ddcm_prob = DDCMProblem(ddcm_data, myObjFunc, None, 'Pure-Dirichlet', forcings, DDCMStab, orders=ords, params=params)
    ddcm_prob.solve("myddcmsol")

    # 4. Compute some error norms
    
    print('... Computing some error norms ...')
    x = fd.SpatialCoordinate(ddcm_data.domain.msh)
    L2error_u = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u , norm_type = 'L2')
    H1error_u = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u , norm_type = 'H1')
    H1error_lmd = fd.errornorm(lmbdexact(x), ddcm_prob.ddcm_solution.lmd, norm_type = 'H1')
    L2error_s = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s , norm_type = 'L2')
    L2error_e = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g , norm_type = 'L2')
    L2error_mu = fd.errornorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type = 'L2')
    if formulation == 'Equal-Order':
        Hdiverror_s = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s , norm_type = 'Hdiv')
        Hdiverror_e = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g , norm_type = 'Hdiv')
        Hdiverror_mu = fd.errornorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type = 'Hdiv')
    else:
        Hdiverror_s = np.nan
        Hdiverror_e = np.nan
        Hdiverror_mu = np.nan

    print('----------------------------------------------------------')
    print('|> Data Driven solution OK')
    print('----------------------------------------------------------')
    
    elapsed_time = time.time() - ti
    print('----------------------------------------------------------')
    print('|> End DDCM Computation | Total time= %1.1f s' %(elapsed_time))
    print('----------------------------------------------------------')

    return hsize, L2error_u, H1error_u, H1error_lmd, L2error_s, L2error_e, L2error_mu, Hdiverror_s, Hdiverror_e, Hdiverror_mu


# Example of a convergence study in convex and non-convex geometries with a known analytical solution
def main_EX92_(formulation='Natural', hsize = 0.1, case_corner = 4):
    
    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')
    
    print('----------------------------------------------------------')
    print('|::| Reference Domain and solution begins ...')
    print('---------------------------------------------')
    
    # 1. Create (or import) the computational domain/mesh and Compute a reference solution    
    lc = hsize             # set the characteristic length of the mesh 
    mymeshfile, tags = CreateMeshReentrantCorner('reentrant', case_corner, lc)
    if(case_corner == 1):
        omega = 1.75*fd.pi
    elif(case_corner == 2):
        omega = 1.5*fd.pi
    elif(case_corner == 3):
        omega = 1.25*fd.pi
    elif(case_corner == 4):
        omega = 0.5*fd.pi
        
    nsd, Ls, Ns = 2, [1.0, 1.0], [100, 100]
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=mymeshfile, flag_quads=False)
    #________________________________
    # These would be the ones corresponding to the analytical solution we propose

    h = fd.CellDiameter(domain.msh)
    params = {'perm': fd.Constant(1.0), 'zeda': fd.Constant(0.0), 
              'theta': fd.Constant(0.5), 'kappa': fd.Constant(1.0), 
              'alpha': fd.Constant(0.125), 'beta': fd.Constant(0.5), 
              'gamma': fd.Constant(0.125), 'eta': fd.Constant(0.5),
              'ls': h, 'lmu': h, 'formulation': formulation} 

    # Analytical expressions
    def uexact(x):
        angle = fd.atan2(x[1] + 1e-12, x[0] + 1e-12)
        theta = fd.conditional(fd.gt(angle, 0), angle, angle + 2 * fd.pi)
        r = fd.sqrt(x[0]**2 + x[1]**2)
        alpha = fd.pi/omega
        return (r**alpha)*fd.sin(alpha*theta)
    def sexact(x):
        perm = params['perm']
        return -perm * fd.grad(uexact(x))
    def eexact(x):
        return fd.grad(uexact(x))

    # Forcings 
    def mysource(x):
        perm = params['perm']
        zeda = params['zeda']
        return fd.div(-perm*fd.grad(uexact(x))) + zeda*uexact(x)
    def mysurfaceflux(x):
        return sexact(x)
    def mybcvals(x):
        return uexact(x)
    forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals)
    #________________________________

    # This will only be called for visualization purposes
    params_anal = {'exact-potential': uexact, 'exact-flux': sexact, 'exact-grad': eexact}
    ref_prob = PoissonStdProblem(domain, params_anal, 'Pure-Dirichlet', forcings, None, PoissonAnalytical, orders=[3,3])

    ref_prob.solve(0.0, "myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: OK')
    print('----------------------------------------------------------')

    # 2. Read or sample some synthetic data from a standard solution: 
    #    The idea is to produce some plausible data similar to what an experiment would deliver

    # The data will be sampled from the analytical expresion on-the-fly
    ddcm_data = PoissonDDCMData(domain, Analytical=True, fluxex=sexact, gradex=eexact)

    # 3. Solve the Data Driven problem: This problem should share the same 
    #    forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g,gdata,s,sdata):
        return fd.inner(g-gdata, g-gdata)*fd.dx + fd.inner(s-sdata, s-sdata)*fd.dx

    if(formulation == 'Natural'):
        ords = [0,1]
    elif(formulation == 'Equal-Order'):
        ords = [1,1]
        
    ddcm_prob = DDCMProblem(ddcm_data, myObjFunc, None, 'Pure-Dirichlet', forcings, DDCMStab, orders=ords, params=params)
    ddcm_prob.solve("myddcmsol")

    # 4. Compute some error norms
    print('   ... Computing some error norms ...')
    x = fd.SpatialCoordinate(ddcm_data.domain.msh)
    L2error_u = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u , norm_type='L2')
    H1error_u = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u , norm_type='H1')
    H1error_lmd = fd.errornorm(fd.Constant(0), ddcm_prob.ddcm_solution.lmd, norm_type='H1')
    error_s = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s , norm_type='L2')
    error_e = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g , norm_type='L2')
    error_mu = fd.errornorm(fd.as_vector([fd.Constant(0),fd.Constant(0)]), ddcm_prob.ddcm_solution.mu, norm_type='L2')

    print('----------------------------------------------------------')
    print('|> Data Driven solution OK')
    print('----------------------------------------------------------')
    
    elapsed_time = time.time() - ti
    print('----------------------------------------------------------')
    print('|> End DDCM Computation | Total time= %1.1f s' %(elapsed_time))
    print('----------------------------------------------------------')

    return L2error_u, H1error_u, H1error_lmd, error_s, error_e, error_mu


# Example of a reference solution based on an analytical expression
def main_EX93_(formulation='Natural', N=20):

    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')
    
    print('----------------------------------------------------------')
    print('|::| Reference Domain and solution begins ...')
    print('---------------------------------------------')

    # 1. Create (or import) the computational domain/mesh and Compute a reference solution

    nsd, Ls, Ns = 2, [1.0, 1.0], [N, N]
    hsize = max(Ls[0]/Ns[0], Ls[1]/Ns[1])
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=True)

    #________________________________
    # These would be the ones corresponding to the analytical solution we propose
    
    def perm(x):
        return fd.Constant(1.0) #+ fd.Constant(0.2)*fd.cos(fd.pi*x[0])*fd.cos(303*fd.pi*x[1])
    
    def zeda(x):
        return fd.Constant(1.0)

    params = {'perm': perm, 'zeda': zeda} 
    
    # Propose an manufactured potential field and its associated gradient
    def uexact(x):
        return fd.cos(fd.pi*x[0])*fd.cos(fd.pi*x[1])

    def sexact(x):
        perm = params['perm']
        return -perm(x)*fd.grad(uexact(x))

    def eexact(x):
        return fd.grad(uexact(x))

    def qexact(x):
        zeda = params['zeda']
        return zeda(x)*uexact(x) + fd.div(sexact(x))
    
    def mysource(x):
        return qexact(x)
    def mysurfaceflux(x):
        return sexact(x)
    def mybcvals(x):
        return uexact(x)
            
    forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals)
    #________________________________

    # This will only be called for visualization purposes
    params_anal = {'exact-potential': uexact, 'exact-flux': sexact, 'exact-grad': eexact}
    ref_prob = PoissonStdProblem(domain, params, 'Pure-Dirichlet', forcings, None, PoissonMixedEO, orders=[1,1])

    ref_prob.solve(N, "myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: OK')
    print('----------------------------------------------------------')

    # 4. Compute some error norms
    
    print('... Computing some error norms ...')
    x = fd.SpatialCoordinate(ref_prob.domain.msh)
    std_sol = ref_prob.std_solution
    L2error_u = fd.errornorm(uexact(x), std_sol.u , norm_type='L2')
    H1error_u = fd.errornorm(uexact(x), std_sol.u , norm_type='H1')
    L2error_s = fd.errornorm(sexact(x), std_sol.s , norm_type= 'L2')
    L2error_e = fd.errornorm(eexact(x), std_sol.g , norm_type= 'L2')
    if formulation == 'Equal-Order':
        Hdiverror_s = fd.errornorm(sexact(x), std_sol.s , norm_type= 'Hdiv')
        Hdiverror_e = fd.errornorm(eexact(x), std_sol.g , norm_type= 'Hdiv')
    else:
        Hdiverror_s = np.nan
        Hdiverror_e = np.nan
        
    print('----------------------------------------------------------')
    print('|> Data Driven solution OK')
    print('----------------------------------------------------------')
    
    elapsed_time = time.time() - ti
    print('----------------------------------------------------------')
    print('|> End DDCM Computation | Total time= %1.1f s' %(elapsed_time))
    print('----------------------------------------------------------')

    return hsize, L2error_u, H1error_u, L2error_s, L2error_e, Hdiverror_s, Hdiverror_e


# On the effect of stabilization parameters and polynomial degree in the natural/equal-order formulation
def main_EX94_(formulation='Natural', N=20, **kwargs):

    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')
    
    print('----------------------------------------------------------')
    print('|::| Reference Domain and solution begins ...')
    print('---------------------------------------------')

    # 1. Create (or import) the computational domain/mesh and Compute a reference solution

    nsd, Ls, Ns = 2, [1.0, 1.0], [N, N]
    hsize = max(Ls[0]/Ns[0], Ls[1]/Ns[1])
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    #________________________________
    # These would be the ones corresponding to the analytical solution we propose

    h = fd.CellDiameter(domain.msh)

	
    params = {'perm': fd.Constant(1.0), 'zeda': fd.Constant(0.001), 
              'theta': fd.Constant(kwargs.get('theta', 0.0)), 	'kappa': fd.Constant(1.0), 
              'alpha': fd.Constant(kwargs.get('alpha', 0.0)), 	'beta': fd.Constant(kwargs.get('beta', 0.0)), 
              'gamma': fd.Constant(kwargs.get('gamma', 0.0)), 	'eta': fd.Constant(kwargs.get('eta', 0.0)),
              'ls': h, 'lmu': h, 'formulation': formulation} 
    
    
    # Propose an manufactured potential field and its associated gradient and also the exact flux
    def uexact(x):
        return fd.cos(fd.pi*x[0])*fd.cos(fd.pi*x[1]) 
    def eexact(x):
        return fd.grad(uexact(x))
    
    # Propose an analytical expression for lambda
    def lmbdexact(x):
        x0, y0  = fd.Constant(0.0), fd.Constant(0.0)
        A = fd.Constant(0.01)
        return A*(fd.Constant(1.0)*((x[0]-x0)**2 + (x[1]-y0)**2))
    
    # Propose a flux-gradient relationship based on the dataset  
    def etilde(x):
        A = fd.Constant(0.01)         # Amplitude
        ome = fd.Constant(4.0)       # Frequency
        phi = fd.Constant(0.0)       # Phase angle 
        return fd.as_vector([eexact(x)[0] + A*fd.sin(ome*fd.pi*x[0] + phi),
                             eexact(x)[1] + A*fd.sin(ome*fd.pi*x[1] + phi)])
    def sexact(x):
        A = fd.Constant(0.035)    # Amplitude
        ome = fd.Constant(3.0)  # Frequency
        phi = fd.Constant(0.0)  # Phase angle
        eps = fd.Constant(1.0e-10)
        met = fd.sqrt(eexact(x)[0]**2 + eexact(x)[1]**2) + eps
        etn = fd.as_vector([eexact(x)[0]/met, eexact(x)[1]/met])
        # mst = met + A*fd.sin(ome*fd.pi*met)
        mst = met - A*met**3
        return fd.as_vector([-mst*etn[0], -mst*etn[1]])
    
    # def sexact(x):
    #     perm = params['perm']
    #     return -perm * eexact(x)
    # Propose an analytical expression for lambda (enforces the continuity equation)
    # def lmbdexact(x):
    #     x0, y0  = fd.Constant(0.5), fd.Constant(0.5)
    #     A = fd.Constant(1.0)
    #     return fd.exp(A*((x[0]-x0)**2 + (x[1]-y0)**2))
    # def lmbdexact(x):
    #     A = fd.Constant(0.01)    # Amplitude
    #     ome = fd.Constant(0.25)     # Frequency
    #     phi = fd.Constant(0.0)   # Phase angle
    #     eps = fd.Constant(1.0e-10)
    #     met = fd.sqrt(eexact(x)[0]**2 + eexact(x)[1]**2) + eps
    #     etn = fd.as_vector([eexact(x)[0]/met, eexact(x)[1]/met])
    #     mst = met + A*fd.sin(ome*fd.pi*met)
    #     return A*fd.exp(x[0]**2 + x[1]**2)
    

    # We define etilde and stilde as the known exact gradient added to a small frequency periodic disturbance
    # def etilde(x):
    #     A = fd.Constant(0.01)         # Amplitude
    #     ome = fd.Constant(1.0)       # Frequency
    #     phi = fd.Constant(0.0)       # Phase angle 
    #     return fd.as_vector([eexact(x)[0] + A*fd.sin(ome*fd.pi*x[0] + phi),
    #                          eexact(x)[1] + A*fd.sin(ome*fd.pi*x[1] + phi)])
    
    # by eq.(22)
    def muexact(x):
        return etilde(x) - eexact(x)
    # by eq.(21)
    def fdual(x):
        zeda = params['zeda']
        return zeda*lmbdexact(x) + fd.div(muexact(x))
    # by eq.(23)
    def stilde(x):
        kappa = params['kappa']
        return sexact(x) - (1/kappa)*fd.grad(lmbdexact(x))
    # by eq.(24)
    def qexact(x):
        zeda = params['zeda']
        return zeda*uexact(x) + fd.div(sexact(x))
    
    def mysource(x):
        return qexact(x)
    def mysurfaceflux(x):
        return sexact(x)
    def mybcvals(x):
        return uexact(x)
    def mylmbd_boundary_vals(x):
        return lmbdexact(x)
    def mymu_boundary_vals(x):
        return muexact(x)
    def mydual_source(x):
        return fdual(x)
            
    forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals, mylmbd_boundary_vals, mymu_boundary_vals, mydual_source)
    #________________________________

    # This will only be called for visualization purposes
    params_anal = {'exact-potential': uexact, 'exact-flux': sexact, 'exact-grad': eexact, 'exact-lmbd' : lmbdexact, 
                   'exact-mu': muexact, 'exact-etilde': etilde, 'exact-stilde': stilde}
    ref_prob = PoissonStdProblem(domain, params_anal, 'Irrelevant', forcings, None, PoissonAnalytical, orders=[3,3])

    ref_prob.solve(0.0, "myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: OK')
    print('----------------------------------------------------------')

    # 2. Read or sample some synthetic data from a standard solution: 
    #    The idea is to produce some plausible data similar to what an experiment would deliver

    # The data will be sampled from the analytical expresion on-the-fly
    ddcm_data = PoissonDDCMData(domain, Analytical=True, fluxex=stilde, gradex=etilde)

    # 3. Solve the Data Driven problem: This problem should share the same 
    #    forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g,gdata,s,sdata):
        kappa = params['kappa']
        return 0.5*fd.inner(g-gdata, g-gdata)*fd.dx + 0.5*kappa*fd.inner(s-sdata, s-sdata)*fd.dx

    if(formulation == 'Natural'):
        ords = [0,1]
    elif(formulation == 'Equal-Order'):
        ords = [1,1]
    ddcm_prob = DDCMProblem(ddcm_data, myObjFunc, None, 'Mixed-Dirichlet-Neumann', forcings, DDCMStab, orders=ords, params=params)

    ddcm_prob.solve("myddcmsol")

    # 4. Compute some error norms
    
    print('... Computing some error norms ...')
    x = fd.SpatialCoordinate(ddcm_data.domain.msh)
    L2error_u = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u , norm_type = 'L2')
    H1error_u = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u , norm_type = 'H1')
    L2error_lmd = fd.errornorm(lmbdexact(x), ddcm_prob.ddcm_solution.lmd, norm_type = 'L2')
    H1error_lmd = fd.errornorm(lmbdexact(x), ddcm_prob.ddcm_solution.lmd, norm_type = 'H1')
    L2error_s = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s , norm_type = 'L2')
    L2error_e = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g , norm_type = 'L2')
    L2error_mu = fd.errornorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type = 'L2')
    if formulation == 'Equal-Order':
        Hdiverror_s = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s , norm_type = 'Hdiv')
        Hdiverror_e = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g , norm_type = 'Hdiv')
        Hdiverror_mu = fd.errornorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type = 'Hdiv')
    else:
        Hdiverror_s = np.nan
        Hdiverror_e = np.nan
        Hdiverror_mu = np.nan

    print('----------------------------------------------------------')
    print('|> Data Driven solution OK')
    print('----------------------------------------------------------')
    
    elapsed_time = time.time() - ti
    print('----------------------------------------------------------')
    print('|> End DDCM Computation | Total time= %1.1f s' %(elapsed_time))
    print('----------------------------------------------------------')

    return hsize, L2error_u, H1error_u, L2error_lmd, H1error_lmd, L2error_s, L2error_e, L2error_mu, Hdiverror_s, Hdiverror_e, Hdiverror_mu
    


# ===== AUXILIARY FUNCTIONS ===== #

def plotErrors(hhs=[], L2u=[], H1u=[], H1l=[], EsL2=[], EeL2=[], EmL2=[], EsHdiv=[], EeHdiv=[],
               EmHdiv=[], formulation='Natural', flag_save_fig=False, fig_names = [], flag_plot_fig = True):
    fig, ax = plt.subplots(figsize=(8,6))
    if L2u:
        ax.loglog(hhs,L2u,'-or',label='$\\|u-u_h\\|_{L^2}$')
    if H1u:
        ax.loglog(hhs,H1u,'-ob',label='$\\|u-u_h\\|_{H^1}$')
    if H1l:
        ax.loglog(hhs,H1l,'-om',label='$\\|\\lambda-\\lambda_h\\|_{H^1}$')
    ax.set_xlabel('$h$')
    ax.set_ylabel('Error')
    if ax.has_data():
        ax.legend()
    ax.grid(which='both',linestyle='--',linewidth=0.5,alpha=0.7)
    ax.set_title(formulation)
    ax.tick_params(axis='both',which='major')
    ax.tick_params(axis='both',which='minor')
    if flag_save_fig:
        plt.savefig(f'{fig_names[0]}.svg')
    if flag_plot_fig:
        plt.show()
    
    fig, ax = plt.subplots(figsize=(8,6))
    if EsL2:
        ax.loglog(hhs,EsL2,'-og',label='$\\|\\mathbf{s}-\\mathbf{s}_h\\|_{L^2}$')
    if EeL2:
        ax.loglog(hhs,EeL2,'-oc',label='$\\|\\mathbf{e}-\\mathbf{e}_h\\|_{L^2}$')
    if EmL2:
        ax.loglog(hhs,EmL2,'-ok',label='$\\|\\boldsymbol{\\mu}-\\boldsymbol{\\mu}_h\\|_{L^2}$')
    
    if formulation == 'Equal-Order':
        if EsHdiv:
            ax.loglog(hhs,EsHdiv,'--^g',label='$\\|\\mathbf{s}-\\mathbf{s}_h\\|_{\\text{Hdiv}}$')
        if EeHdiv:
            ax.loglog(hhs,EeHdiv,'--^c',label='$\\|\\mathbf{e}-\\mathbf{e}_h\\|_{\\text{Hdiv}}$')
        if EmHdiv:
            ax.loglog(hhs,EmHdiv,'--^k',label='$\\|\\boldsymbol{\\mu}-\\boldsymbol{\\mu}_h\\|_{\\text{Hdiv}}$')
    
    ax.set_xlabel('$h$')
    ax.set_ylabel('Error')
    if ax.has_data():
        ax.legend()
    ax.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.tick_params(axis='both', which='major')
    ax.tick_params(axis='both', which='minor')
    ax.set_title(formulation)
    if flag_save_fig:
        plt.savefig(f'{fig_names[1]}.svg')
    if flag_plot_fig:
        plt.show()

def printConvergenceRates(conv_rates):
    headers = ["Norm", "Convergence rate"]
    table_data = [[norm] + [f"{rate:.4}" for rate in val] for norm, val in conv_rates.items()]
    print("Estimated Convergence Rates:")
    print(tabulate(table_data, headers=headers, tablefmt="github"))
    
   
# ===== RUN ===== #
import sys
from tabulate import tabulate


main_selector = np.int32(sys.argv[1])

if(main_selector == 10): # Proximity measure with the modulus
    main_EX10_()

elif(main_selector == 50): # Convergence study of Forcheimer's model with manufactured solution
    hhs, L2u, H1u, L2s = [], [], [], []
    for N in [20, 40, 80, 160]:
        hsize, L2error_u, H1error_u, L2error_s = main_EX50_(N)
        print(hsize, L2error_u, H1error_u, L2error_s)
        hhs.append(hsize)
        L2u.append(L2error_u)
        H1u.append(H1error_u)
        L2s.append(L2error_s)

    plt.loglog(hhs, L2u, '-or')
    plt.loglog(hhs, H1u, '-ob')
    plt.loglog(hhs, L2s, '-og')
    plt.xlabel('$h$')
    plt.ylabel('Error')
    plt.legend(['$\|u-u_h\|_{L^2}$', '$\|u-u_h\|_{H^1}$',  '$\|\mathbf{s}-\mathbf{s}_h\|_{L^2}$'])
    plt.show()

elif(main_selector == 51): # Forcheimer's model with anysotropic K
    main_EX51_()

elif(main_selector == 4): # Data-driven solution based on reference solution given by analytical expression
    npoints_data, hsize, L2error_u, H1error_u, L2error_s = main_EX4_()
    print("npoints= %d\nhsize= %f\nL2error_u= %f\nH1error_u=%f\nL2error_s=%f\n" %(npoints_data, hsize, L2error_u, H1error_u, L2error_s))

elif(main_selector == 7): # Data-driven solution with reaction term
    main_EX7_()

elif(main_selector == 8): # Data-driven solution with a domain with a reentrant corner
    main_EX8_()

elif(main_selector == 90): 
    
    formulation = 'Natural'
    # formulation = 'Equal-Order'
   
    # Store all the errors in a dictionary 
    hhs, L2u, H1u, H1l, EsL2, EeL2, EmL2 = [], [], [], [], [], [], []
    EsHdiv, EeHdiv, EmHdiv = [], [], []
    errors = {'L2u': L2u, 'H1u': H1u, 'H1l': H1l,'EsL2': EsL2, 'EeL2': EeL2, 'EmL2': EmL2, 
              'EsHdiv': EsHdiv, 'EeHdiv': EeHdiv, 'EmHdiv': EmHdiv}
    conv_rates = {key: [] for key in errors.keys()}
    
    # Loop over different mesh refinements 
    for N in [25, 50, 100, 200] :
        output = main_EX90_(formulation, N)
        hhs.append(output[0]); L2u.append(output[1]); H1u.append(output[2]); H1l.append(output[3]); EsL2.append(output[4]) 
        EeL2.append(output[5]); EmL2.append(output[6]); EsHdiv.append(output[7]); EeHdiv.append(output[8]); EmHdiv.append(output[9]) 
        
    # Compute the associated convergence rates 
    for key, value in errors.items():
        if np.all(np.isnan(value)):
            conv_rates[key] = [np.nan]
        else:
            p = (np.log(value[-1] / value[-2]) / np.log(hhs[-1] / hhs[-2]))
            conv_rates[key].append(p)

    # Plots
    flag_save_fig = True
    fig_names = ['Darcy_errors_u_lmd','Darcy_errors_s_e_mu']
    plotErrors(hhs, L2u, H1u, H1l, EsL2, EeL2, EmL2, EsHdiv, EeHdiv, EmHdiv, formulation, flag_save_fig, fig_names)          
    
    # Output
    printConvergenceRates(conv_rates)

elif(main_selector == 91):
    
    # formulation = 'Natural'
    formulation = 'Equal-Order'
    
    hhs, L2u, H1u, H1l, EsL2, EeL2, EmL2 = [], [], [], [], [], [], []
    EsHdiv, EeHdiv, EmHdiv = [], [], []
    errors = {'L2u': L2u, 'H1u': H1u, 'H1l': H1l, 'EsL2': EsL2, 'EeL2': EeL2, 'EmL2': EmL2, 
              'EsHdiv': EsHdiv, 'EeHdiv': EeHdiv, 'EmHdiv': EmHdiv}
    conv_rates = {key: [] for key in errors.keys()}
    
   # Loop over different mesh refinements 
    for N in [25, 50, 100, 200] :
        output = main_EX91_(formulation, N)
        hhs.append(output[0]); L2u.append(output[1]); H1u.append(output[2]); H1l.append(output[3]); EsL2.append(output[4]) 
        EeL2.append(output[5]); EmL2.append(output[6]); EsHdiv.append(output[7]); EeHdiv.append(output[8]); EmHdiv.append(output[9]) 
            
    # Compute the associated convergence rates 
    for key, value in errors.items():
        if np.all(np.isnan(value)):
            conv_rates[key] = [np.nan]
        else:
            p = (np.log(value[-1] / value[-2]) / np.log(hhs[-1] / hhs[-2]))
            conv_rates[key].append(p)
                 
    # Plots
    flag_save_fig = False
    fig_names = ['Forcheimer_errors_u_lmd','Forcheimer_errors_s_e_mu']
    plotErrors(hhs, L2u, H1u, H1l, EsL2, EeL2, EmL2, EsHdiv, EeHdiv, EmHdiv, formulation, flag_save_fig, fig_names)         
            
    # Output
    printConvergenceRates(conv_rates)

elif(main_selector == 92):
    
    hhs = [0.1, 0.05, 0.025, 0.0125]
    # Specify the formulation to be used
    formulation = 'Natural'  
    # formulation = 'Equal-Order'

    # Specify each case corner to be used
    case_corners = [1, 2, 3, 4]
    c_corners_labels = ['$\\frac{7\pi}{4}$', '$\\frac{3\pi}{2}$', '$\\frac{5\pi}{4}$','$\\frac{\pi}{2}$']

    # Allocate memory for storing the errors for each case corner and different mesh sizes 
    L2u = np.empty((len(case_corners), len(hhs)))
    H1u, H1l, Es, Ee, Em = (np.empty_like(L2u) for _ in range(5))

    # Store all the errors in a dictionary 
    errors = {'L2u': L2u, 'H1u': H1u, 'H1l': H1l, 'Es': Es, 'Ee': Ee, 'Em': Em}
    conv_rates = {key: [] for key in errors.keys()}
    norm_labels = {
        'L2u': '$\|u-u_h\|_{L^2}$', 
        'H1u': '$\|u-u_h\|_{H^1}$', 
        'H1l': '$\|\lambda-\lambda_h\|_{H^1}$',
        'Es': '$\|\mathbf{s}-\mathbf{s}_h\|_{L^2}$', 
        'Ee': '$\|\mathbf{e}-\mathbf{e}_h\|_{L^2}$',
        'Em': '$\|\\boldsymbol{\mu}-\\boldsymbol{\mu}_h\|_{L^2}$'
    }

    # Loop through each case corner and mesh size
    for case_corner_index, case_corner in enumerate(case_corners):
        for hsize_index, hsize in enumerate(hhs):
            # Call the function
            L2error_u, H1error_u, H1error_lmd, error_s, error_e, error_mu = main_EX92_(formulation, hsize, case_corner)
            errors['L2u'][case_corner_index, hsize_index] = L2error_u
            errors['H1u'][case_corner_index, hsize_index] = H1error_u
            errors['H1l'][case_corner_index, hsize_index] = H1error_lmd
            errors['Es'][case_corner_index, hsize_index] = error_s 
            errors['Ee'][case_corner_index, hsize_index] = error_e 
            errors['Em'][case_corner_index, hsize_index] = error_mu

    # Generate plots for each error metric
    for key, value in errors.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        for case_corner_index, case_corner in enumerate(case_corners):
            # Plot
            ax.loglog(hhs, value[case_corner_index, :], '-o', label=f'$\omega=${c_corners_labels[case_corner_index]}')
            # Compute the associated convergence rate
            p = (np.log(value[case_corner_index, -1] / value[case_corner_index, -2]) /
                np.log(hhs[-1] / hhs[-2]))
            conv_rates[key].append(p)

        # Configure the plot
        ax.set_xlabel('$h$')
        ax.set_ylabel(norm_labels[key])
        ax.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.tick_params(axis='both', which='major')
        ax.tick_params(axis='both', which='minor')
        ax.set_title(formulation)
        ax.legend()
        # plt.savefig(f'{key}_errors_{formulation}.svg')
        # plt.show()
    

    # Output
    headers = ["Norm"] + c_corners_labels
    table_data = [[err] + [f"{rate:.4f}" for rate in val] for err, val in conv_rates.items()]
    print("Estimated Convergence Rates:")
    print(tabulate(table_data, headers=headers, tablefmt="github"))
     
elif(main_selector == 93): 
    
    formulation = 'Natural'
    # formulation = 'Equal-Order'

    
    # Store all the errors in a dictionary 
    hhs, L2u, H1u, H1l, EsL2, EeL2, EmL2 = [], [], [], [], [], [], []
    EsHdiv, EeHdiv, EmHdiv = [], [], []
    errors = {'L2u': L2u, 'H1u': H1u, 'H1l': H1l, 'EsL2': EsL2, 'EeL2': EeL2, 'EmL2': EmL2, 
              'EsHdiv': EsHdiv, 'EeHdiv': EeHdiv, 'EmHdiv': EmHdiv}
    conv_rates = {key: [] for key in errors.keys()}
    
   # Loop over different mesh refinements 
    for N in [25, 50, 100, 200] :
        output = main_EX93_(formulation, N)
        hhs.append(output[0]);  L2u.append(output[1]);    H1u.append(output[2]);    EsL2.append(output[3]) 
        EeL2.append(output[4]); EsHdiv.append(output[5]); EeHdiv.append(output[6])
		
    # Compute the associated convergence rates 
    for key, value in errors.items():
        if np.all(np.isnan(value)):
            conv_rates[key] = [np.nan]
        else:
            p = (np.log(value[-1] / value[-2]) / np.log(hhs[-1] / hhs[-2]))
            conv_rates[key].append(p)
              
    # Plots
    flag_save_fig = False
    fig_names = ['xxx_errors_u_lmd','xxx_errors_s_e_mu']
    plotErrors(hhs, L2u, H1u, H1l, EsL2, EeL2, EmL2, EsHdiv, EeHdiv, EmHdiv, formulation, flag_save_fig)          
    
    # Output
    printConvergenceRates(conv_rates)

elif(main_selector == 94):
		
    formulation = 'Natural'; stab_params = {None:None}; fig_val_list = ['u', 'l', 's_e_mu' ]
    # formulation = 'Equal-Order'; 
    # stab_params = {'theta': 0.5, 'alpha': 0.125, 'beta': 0.5, 'gamma': 0.125, 'eta': 0.5}; 
    # stab_params = {None:None}
    # fig_val_list = ['uL2', 'uH1','lL2' ,'lH1', 'sL2', 'sHdiv', 'eL2', 'eHdiv', 'muL2', 'muHdiv']
    line_styles = {'theta': '-o','alpha':'--s','beta' :'-.^', 'gamma': ':p','eta': '--d'}
    
    def initializePlot(fig_num, xlabel, ylabel, title):
        """Initialize the plot with labels, title, grid, and tick parameters."""
        plt.figure(fig_num, figsize=(8, 6))
        plt.xlabel(xlabel, fontsize=11)
        plt.ylabel(ylabel, fontsize=11)
        plt.title(title, fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tick_params(axis='both', which='major', labelsize=11)
        plt.tick_params(axis='both', which='minor', labelsize=11)
    
    def plotLines(fig_num, hhs, data, label, line_type='-o'):
            """Plot lines on an existing figure."""
            plt.figure(fig_num)  
            plt.loglog(hhs, data,  line_type, linewidth=2, label=label)
            
    if formulation == 'Natural':
        # Only one plot natural formulation
        for fig_num in range(1,4):
            initializePlot(fig_num, '$h$', 'Error', f'{formulation}')  
    elif formulation == 'Equal-Order':  
        # Five plots for the equal-order fomulation, one for each variable: u, l, s, e, mu
        for fig_num in range(1,11):
            initializePlot(fig_num, '$h$', 'Error', f'{formulation}')    
    else:
        raise ValueError('Invalid formulation')
    
    # Loop over each stabilization parameters
    for param, val in stab_params.items():
        kwargs = {} if formulation == 'Natural' else {param: val}
        print(f'==== Solving for {kwargs} ====')
        
        # Store all the errors in a dictionary 
        hhs, L2u, H1u, L2l, H1l, EsL2, EeL2, EmL2 = [], [], [], [], [], [], [], []
        EsHdiv, EeHdiv, EmHdiv = [], [], []
        errors = {'L2u': L2u, 'H1u': H1u, 'L2l': L2l, 'H1l': H1l, 'EsL2': EsL2, 'EeL2': EeL2, 'EmL2': EmL2, 
            'EsHdiv': EsHdiv, 'EeHdiv': EeHdiv, 'EmHdiv': EmHdiv}
        conv_rates = {key: [] for key in errors.keys()}
        
        # Loop over different mesh refinements
        for N in [50] :
            output = main_EX94_(formulation, N, **kwargs)
            hhs.append(output[0]);  	L2u.append(output[1]);   	H1u.append(output[2]) 	    
            L2l.append(output[3]);      H1l.append(output[4]);      EsL2.append(output[5])
            EeL2.append(output[6]); 	EmL2.append(output[7]);     EsHdiv.append(output[8])
            EeHdiv.append(output[9]);	EmHdiv.append(output[10])
                
        # Compute the associated convergence rates 
        for key, value in errors.items():
            if np.all(np.isnan(value)):
                conv_rates[key] = [np.nan]
            else:
                p = (np.log(value[-1] / value[-2]) / np.log(hhs[-1] / hhs[-2]))
                conv_rates[key].append(p)
                
        # Output
        printConvergenceRates(conv_rates)
    
        
        if formulation == 'Natural':
            # Plot u in L2 and H1
            plotLines(1, hhs, L2u, '$\\|u - u_h\\|_{L^2}$', '-o')
            plotLines(1, hhs, H1u, '$\\|u - u_h\\|_{H^1}$',  '-o')
            # Plot lambda in L2 and H1
            plotLines(2, hhs, L2l, '$\\|\\lambda - \\lambda_h\\|_{L^2}$','--s')
            plotLines(2, hhs, H1l, '$\\|\\lambda - \\lambda_h\\|_{H^1}$','--s')
            # Plot s in L2 and Hdiv
            plotLines(3, hhs, EsL2,  '$\\|\\mathbf{s} - \\mathbf{s}_h\\|_{L^2}$', '-.^')
            # Plot e in L2 and Hdiv
            plotLines(3, hhs, EeL2,  '$\\|\\mathbf{e} - \\mathbf{e}_h\\|_{L^2}$', ':p')
            # Plot mu in L2 and Hdiv
            plotLines(3, hhs, EmL2,  '$\\|\\boldsymbol{\\mu} - \\boldsymbol{\\mu}_h\\|_{L^2}$', '--d')
 
        elif formulation == 'Equal-Order':
            # Plot u in L2 and H1
            plotLines(1, hhs, L2u, '$\\|u - u_h\\|_{L^2}$', '-o')
            plotLines(2, hhs, H1u, '$\\|u - u_h\\|_{H^1}$', '-o')
            # Plot lambda in L2 and H1
            plotLines(3, hhs, L2l, '$\\|\\lambda - \\lambda_h\\|_{L^2}$','--s')
            plotLines(4, hhs, H1l, '$\\|\\lambda - \\lambda_h\\|_{H^1}$', '--s')
            # Plot s in L2 and Hdiv
            plotLines(5, hhs, EsL2,  '$\\|\\mathbf{s} - \\mathbf{s}_h\\|_{L^2}$', '-.^')
            plotLines(6, hhs, EsHdiv,'$\\|\\mathbf{s} - \\mathbf{s}_h\\|_{Hdiv}$','-.^')
            # Plot e in L2 and Hdiv
            plotLines(7, hhs, EeL2,  '$\\|\\mathbf{e} - \\mathbf{e}_h\\|_{L^2}$', ':p')
            plotLines(8, hhs, EeHdiv,'$\\|\\mathbf{e} - \\mathbf{e}_h\\|_{Hdiv}$',':p')
            # Plot mu in L2 and Hdiv
            plotLines(9, hhs, EmL2,  '$\\|\\boldsymbol{\\mu} - \\boldsymbol{\\mu}_h\\|_{L^2}$','--d')
            plotLines(10, hhs, EmHdiv,'$\\|\\boldsymbol{\\mu} - \\boldsymbol{\\mu}_h\\|_{Hdiv}$', '--d')
        
        else:
            raise ValueError('Invalid formulation')
        
        # Add legends and save
        if formulation == 'Natural':
            for fig_num in range(1,4):
                plt.figure(fig_num)
                plt.legend(loc='best', fontsize=11, ncol=2, fancybox=True, shadow=True)
                plt.savefig(f'error_{fig_val_list[fig_num-1]}.svg')
        elif formulation == 'Equal-Order':
            for fig_num in range(1,11):
                plt.figure(fig_num)
                plt.legend(loc='best', fontsize=11, ncol=2, fancybox=True, shadow=True)
                plt.savefig(f'error_{fig_val_list[fig_num-1]}.svg')
        else: 
            raise ValueError('Invalid formulation')    
    plt.show()
     
else:
    print('\n   ||||>>> Main selector option not recognized: ', main_selector)

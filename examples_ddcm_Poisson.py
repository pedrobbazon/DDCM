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
        modgreg = eta*(sqrt(1.0 + fd.inner(g-gdata,g-gdata)/eta**2) - 1.0)
        modsreg = eta*(sqrt(1.0 + fd.inner(s-sdata,s-sdata)/eta**2) - 1.0)
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
       kappa = kappa0*(1.0 + 10.0*0.5 * (1.0 + tanh(viol*(u-2.0))))
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
        beta = fd.Constant(10)
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

#==================================================================================================================================================#

# Example of a reference solution based on an analytical expression
def main_EX9_(formulation='Natural', N=20, flag_norm_type = 'L2'):

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

    params = {'perm': fd.Constant(1.0), 'zeda': fd.Constant(1.0), 
              'theta': fd.Constant(0.5), 'kappa': fd.Constant(1.0), 
              'alpha': fd.Constant(0.125), 'beta': fd.Constant(0.5), 
              'gamma': fd.Constant(0.125), 'eta': fd.Constant(0.5),
              'ls': h, 'lmu': h, 'formulation': formulation} 

    def uexact(x):
        return fd.cos(fd.pi*x[0])*fd.cos(fd.pi*x[1])
    def sexact(x):
        perm = params['perm']
        return -perm * fd.grad(uexact(x))
    def eexact(x):
        return fd.grad(uexact(x))

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
    
    print('... Computing some error norms ...')
    x = fd.SpatialCoordinate(ddcm_data.domain.msh)
    L2error_u = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u , norm_type='L2')
    H1error_u = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u , norm_type='H1')
    H1error_lmd = fd.errornorm(fd.Constant(0), ddcm_prob.ddcm_solution.lmd, norm_type='H1')
    error_s = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s , norm_type= flag_norm_type)
    error_e = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g , norm_type= flag_norm_type)
    error_mu = fd.errornorm(fd.as_vector([fd.Constant(0),fd.Constant(0)]), ddcm_prob.ddcm_solution.mu, norm_type=flag_norm_type)

    print('----------------------------------------------------------')
    print('|> Data Driven solution OK')
    print('----------------------------------------------------------')
    
    elapsed_time = time.time() - ti
    print('----------------------------------------------------------')
    print('|> End DDCM Computation | Total time= %1.1f s' %(elapsed_time))
    print('----------------------------------------------------------')

    return hsize, L2error_u, H1error_u, H1error_lmd, error_s, error_e, error_mu

# Example of a convergence study in convex and non-convex geometries with a known analytical solution
def main_EX91_(formulation='Natural', hsize = 0.1, case_corner = 4):
    
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



#--- Run
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
elif(main_selector == 9): 
    # formulation = 'Natural'
    formulation = 'Equal-Order'
    if formulation == 'Natural':
        flag_norm_labels = ['L2']
    elif formulation == 'Equal-Order':
        flag_norm_labels = ['L2', 'Hdiv']
    else:
        raise ValueError('Invalid options')
    
    hhs, L2u, H1u, H1l, EsL2, EeL2, EmL2 = [], [], [], [], [], [], []
    EsHdiv, EeHdiv, EmHdiv = [], [], []
    
    # Loop over different mesh refinements 
    for index, flag_norm_type in enumerate(flag_norm_labels):
        hsize, L2error_u, H1error_u, H1error_lmd, error_s, error_e, error_mu = [], [], [], [], [], [], []    # reset before changing the flag norm
        for N in [25, 50, 100, 200]:
            hsize, L2error_u, H1error_u, H1error_lmd, error_s, error_e, error_mu = main_EX9_(formulation, N, flag_norm_type)
            if flag_norm_type == 'L2':
                hhs.append(hsize)
                L2u.append(L2error_u)
                H1u.append(H1error_u)
                H1l.append(H1error_lmd)
                EsL2.append(error_s)
                EeL2.append(error_e)
                EmL2.append(error_mu)
            elif flag_norm_type == 'Hdiv':
                EsHdiv.append(error_s) 
                EeHdiv.append(error_e)
                EmHdiv.append(error_mu)
            else:
                raise ValueError('Error!')
            
    # Plot the ||u-uh||_L2, ||u-uh||_H1 and ||lambda-lambda_h||_H1
    fig, ax = plt.subplots(figsize=(8,6))
    ax.loglog(hhs, L2u, '-or', label = '$\|u-u_h\|_{L^2}$')
    ax.loglog(hhs, H1u, '-ob', label = '$\|u-u_h\|_{H^1}$')
    ax.loglog(hhs, H1l, '-om', label = '$\|\lambda-\lambda_h\|_{H^1}$')
    ax.set_xlabel('$h$')
    ax.set_ylabel('Error')
    ax.legend()
    ax.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.tick_params(axis='both', which='major')
    ax.tick_params(axis='both', which='minor')
    if(formulation == 'Natural'):
        ax.set_title('Natural')
    elif(formulation == 'Equal-Order'):
        ax.set_title('Equal Order')
    plt.savefig('errors_u_lmd_'+formulation+'.svg')
    plt.show()
    
    # Plot the ||s-sh||_V, ||e-eh||_V and ||mu-mu_h||_V, being V: L2 or Hdiv depending on the selected formulation
    fig, ax = plt.subplots(figsize=(8,6))
    ax.loglog(hhs, EsL2, '-og', label = '$\|\mathbf{s}-\mathbf{s}_h\|_{L^2}$')
    ax.loglog(hhs, EeL2, '-oc', label = '$\|\mathbf{e}-\mathbf{e}_h\|_{L^2}$')
    ax.loglog(hhs, EmL2, '-ok', label = '$\|\\boldsymbol{\mu}-\\boldsymbol{\mu}_h\|_{L^2}$' )
    if formulation == 'Equal-Order':
        ax.loglog(hhs, EsHdiv, '--^g', label = '$\|\mathbf{s}-\mathbf{s}_h\|_{\\text{Hdiv}}$')
        ax.loglog(hhs, EeHdiv, '--^c', label = '$\|\mathbf{e}-\mathbf{e}_h\|_{\\text{Hdiv}}$')
        ax.loglog(hhs, EmHdiv, '--^k', label = '$\|\\boldsymbol{\mu}-\\boldsymbol{\mu}_h\|_{\\text{Hdiv}}$' )
    ax.set_xlabel('$h$')
    ax.set_ylabel('Error')
    ax.legend()
    ax.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.tick_params(axis='both', which='major')
    ax.tick_params(axis='both', which='minor')
    if(formulation == 'Natural'):
        ax.set_title('Natural')
    elif(formulation == 'Equal-Order'):
        ax.set_title('Equal Order')
    else:
        raise ValueError('Invalid option!')
    plt.savefig('errors_s_e_mu_'+formulation+'.svg')
    plt.show()
        
elif(main_selector == 91):
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
            L2error_u, H1error_u, H1error_lmd, error_s, error_e, error_mu = main_EX91_(formulation, hsize, case_corner)
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

        
else:
    print('\n   ||||>>> Main selector option not recognized: ', main_selector)

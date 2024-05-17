#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on:         12th April, 2024
Last Modified on:   17th May, 2024

@authors: P. Vazao Balbao & R.F. Ausas
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

# Example using the simple linear Darcy model to generate the reference solution and sample the data from it
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
        Nsampx, Nsampy  = 3, 3
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

    nsd, Ls, Ns = 2, [0.2, 1.0], [2, 10]
    synth_domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    # Let us suppose we do some continuation to compute the data:
    bcvals = [2.0, 4.0, 6.0, 8.0]
    gradu_samples, flux_samples = [], []
    for bcvs in bcvals:

        def mysynbcvals(x):
            return fd.Constant(bcvs)
        synth_forcings = PoissonForcingData(None, None, mysynbcvals)
        
        params = {'kappa': fd.Constant(1.0)}
        synth_prob = PoissonStdProblem(synth_domain, params, 'Uniaxial', synth_forcings, FLM_DarcyLinear, PoissonDualMixedHdiv, orders=[2,1])
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
        gradu_sampled_vals = np.array(g.at(xs))
        flux_sampled_vals = np.array(s.at(xs))
        #-------------------------------------------  
        
        # Apply any manipulation/transformation you need to define
        # the synthetic data (rotations, scalings, removing outliers, etc)
        gradu_samples, flux_samples = RotateData2D(0.0, 2.0*np.pi, 4, False, 
                                                   gradu_sampled_vals, flux_sampled_vals, gradu_samples, flux_samples)

    ddcm_data.SetSampledData(np.concatenate(gradu_samples), np.concatenate(flux_samples), add_origin=True, data_file='myDATA.txt')

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

# Example using k(u) nonlinear model (invinted by me) to generate the synthetic data
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
    bcvals = [2.0, 4.0, 6.0, 8.0]
    gradu_samples, flux_samples = [], []
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
        Nsampx, Nsampy  = 1, 10
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
        gradu_samples, flux_samples = RotateData2D(0.0, 2.0*np.pi, 4, False, 
                                                   gradu_sampled_vals, flux_sampled_vals, gradu_samples, flux_samples)
    
    ddcm_data.SetSampledData(np.concatenate(gradu_samples), np.concatenate(flux_samples), add_origin=True, data_file='myDATA.txt')

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

    params = {'eta': fd.Constant(1.e-6), 'mu_over_kappa': fd.Constant(1.0), 'beta_x_rho': fd.Constant(100.0)}
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
    bcvals = np.linspace(1.0+1.e-15, 10.0, 50)
    gradu_samples, flux_samples = [], []
    for bcvs in bcvals:

        print('----')
        print('|||| B.C. Val on top wall= %1.2f' %(bcvs))
        def mysynbcvals(x):
            return fd.Constant(bcvs)
        synth_forcings = PoissonForcingData(None, None, mysynbcvals)
        
        # They could be the same parameters as in the reference solution or we may add some perturbation
        params = {'eta': fd.Constant(1.e-6), 'mu_over_kappa': fd.Constant(1.0), 'beta_x_rho': fd.Constant(1.0)}
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
        gradu_sampled_vals = np.array(g.at(xs))
        flux_sampled_vals = np.array(s.at(xs))
        #-------------------------------------------  
        
        # Apply any manipulation/transformation you need to define
        # the synthetic data (rotations, scalings, removing outliers, etc)
        gradu_samples, flux_samples = RotateData2D(0.0, 2.0*np.pi, 4, False, 
                                                   gradu_sampled_vals, flux_sampled_vals, gradu_samples, flux_samples)

    ddcm_data.SetSampledData(np.concatenate(gradu_samples), np.concatenate(flux_samples), add_origin=False, data_file='myDATA.txt')

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

# Example of a refence solution based on an analytical expression
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
        return cos(pi*x[0])*cos(pi*x[1])
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

    # This will be only called for visualation purposes
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

    # Note in this case, I've made the data associated to the same domain (and mesh)
    # we used for the reference solution, but this doesn't have to be the case, it can be a finer mesh
    nsd, Ls, Ns = 2, [1.0, 1.0], [50, 50]
    ddcm_domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)
    ddcm_data = PoissonDDCMData(ddcm_domain)

    # In this manufactured example, we create by hand fluxes and grads 
    # without actually solving a synthetic problem:
    bcvals = np.linspace(1.0+1.e-15, 10.0, 10)
    gradu_samples, flux_samples = [], []
    for bcvs in bcvals:
        # Generate a flux s and a grad g 
        g1 = 0.0
        s1 = 0.0
        L = 1.0
        g2 = (bcvs - 1.0)/L
        kappa = 1.0
        s2 = -kappa*g2
        gradu_sampled_vals = np.array([[g1, g2]])
        flux_sampled_vals  = np.array([[s1, s2]])
        #-------------------------------------------  
        
        # Apply any manipulation/transformation you need to define
        # the synthetic data (rotations, scalings, removing outliers, etc)
        gradu_samples, flux_samples = RotateData2D(0.0, 2.0*np.pi, 4, False, 
                                                   gradu_sampled_vals, flux_sampled_vals, gradu_samples, flux_samples)

    ddcm_data.SetSampledData(np.concatenate(gradu_samples), np.concatenate(flux_samples), add_origin=True, data_file='myDATA.txt')

    ddcm_data.Distribute(ref_prob.std_solution, 'Interpolate-VDG0', flag_plot=True)

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
#--- Run
_ = main_EX1_()















#domain = ComputationalDomain(nsd, meshfile='cook_memb.msh', flag_quads=False)

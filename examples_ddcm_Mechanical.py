#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on:         30th May, 2024
Last Modified on:   28th August, 2024

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

# Example using the simple linear Elasticity model to generate the reference solution and sample the data from it
def main_Elas_EX0_():

    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')
    
    print('----------------------------------------------------------')
    print('|::| Reference Domain and solution begins ...')
    print('---------------------------------------------')

    # 1. Create (or import) the computational domain/mesh and Compute a reference solution

    nsd, Ls, Ns = 2, [1.0, 1.0], [100, 100]
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    #________________________________
    # These are the only things I need to specify for an Indent-Block
    def mybodyforce(x):
        return fd.as_vector([fd.Constant(0.0),fd.Constant(-0.05)]) 
    def mysurfaceforceTop(x):
        return fd.as_vector([fd.Constant(0.0), -0.05*exp(-5.0*(x[0])**2)])
    mysurfaceforce = {'Top': mysurfaceforceTop}
    boundary_tags = {'Left': 1 , 'Right': 2, 'Bottom': 3, 'Top': 4}
    forcings = MechanicalForcingData(mybodyforce, mysurfaceforce, None, boundary_tags)
    #________________________________

    params = {'YoungMod': fd.Constant(1.0), 'PoissonRat': fd.Constant(0.475)}

    ref_prob = MechanicalStdProblem(domain, params, 'Indent-Block', forcings, SSLP_LinearElas, MechanicalPrimalCG, orders=[1])

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
    ddcm_data = MechanicalDDCMData(domain)

    nsd, Ls, Ns = 2, [1.0, 1.0], [100, 100]
    synth_domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    synth_forcings = forcings
        
    synth_prob = MechanicalStdProblem(synth_domain, params, 'Indent-Block', synth_forcings, SSLP_LinearElas, MechanicalPrimalCG, orders=[1])
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
    g = fd.project(stdsyn_sol.g, synth_domain.TDG0)
    s = fd.project(stdsyn_sol.s, synth_domain.TDG0)
    symgradu_sampled_vals = np.array(g.at(xs))
    sigma_sampled_vals = np.array(s.at(xs))
    #-------------------------------------------  
        
    # Apply any manipulation/transformation you need to define
    # the synthetic data (rotations, scalings, removing outliers, etc)
    symgradu_samples = symgradu_sampled_vals
    sigma_samples  = sigma_sampled_vals

    ddcm_data.SetSampledData(symgradu_samples, sigma_samples, add_origin=False, data_file='myDATA.txt')

    ddcm_data.Distribute(ref_prob.std_solution, 'Project-TDG0', flag_plot=True)

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution: %d data points' %(ddcm_data.sampled_stress.shape[0]))
    print('----------------------------------------------------------')

    # 3. Solve the Data Driven problem: This problem should share the same 
    #    forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g,gdata,s,sdata):
        return fd.inner(g-gdata, g-gdata)*fd.dx + fd.inner(s-sdata, s-sdata)*fd.dx
    def myDerObjFunc(g, gdata, dg, s, sdata, dsi):
        return [fd.inner(dsi, s-sdata)*fd.dx, fd.inner(dg, g-gdata)*fd.dx]

    ddcm_prob = DDCMMechProblem(ddcm_data, myObjFunc, myDerObjFunc, 'Indent-Block', forcings, NLSP5FA_MECH, orders=[0,1])

    ddcm_prob.solve("myddcmsol")

    print('----------------------------------------------------------')
    print('|> Data Driven solution OK')
    print('----------------------------------------------------------')
    
    elapsed_time = time.time() - ti
    print('----------------------------------------------------------')
    print('|> End DDCM Computation | Total time= %1.1f s' %(elapsed_time))
    print('----------------------------------------------------------')

    return 0

# Example to illustrate the data-driven approach on an indentation problem sampling data
# from a) uniaxial test, b) cantilever beam or something like that
def main_Elas_EX02_():

    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')
    
    print('----------------------------------------------------------')
    print('|::| Reference Domain and solution begins ...')
    print('---------------------------------------------')

    # 1. Create (or import) the computational domain/mesh and Compute a reference solution

    nsd, Ls, Ns = 2, [1.0, 1.0], [100, 100]
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    #________________________________
    # These are the only things I need to specify for an Indent-Block
    def mybodyforce(x):
        return fd.as_vector([fd.Constant(0.0),fd.Constant(-0.05)]) 
    def mysurfaceforceTop(x):
        return fd.as_vector([fd.Constant(0.0), -0.05*exp(-5.0*(x[0])**2)])
    mysurfaceforce = {'Top': mysurfaceforceTop}
    boundary_tags = {'Left': 1 , 'Right': 2, 'Bottom': 3, 'Top': 4}
    forcings = MechanicalForcingData(mybodyforce, mysurfaceforce, None, boundary_tags)
    #________________________________

    params = {'YoungMod': 1.0, 'PoissonRat': 0.475}

    ref_prob = MechanicalStdProblem(domain, params, 'Indent-Block', forcings, SSLP_LinearElas, MechanicalPrimalCG, orders=[1])

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
    ddcm_data = MechanicalDDCMData(domain)

    nsd, Ls, Ns = 2, [0.2, 1.0], [3, 15]
    synth_domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    # Let us suppose we do some continuation to compute the data:
    Nbc, Nth = 3, 5
    bcvals = np.linspace(0.01+1.e-15, 0.02, Nbc)
    symgradu_samples = np.empty(shape=(Nbc*Nth,2,2), dtype=np.float64)
    sigma_samples = np.empty(shape=(Nbc*Nth,2,2), dtype=np.float64)
    count = 0
    for bcvs in bcvals:

        print('----')
        print('|||| B.C. Val on top wall= %1.2f' %(bcvs))
        def mysynbcvals(x):
            return fd.Constant(-bcvs)

        boundary_tags = {'Left': 1 , 'Right': 2, 'Bottom': 3, 'Top': 4}
        synth_forcings = MechanicalForcingData(None, None, mysynbcvals, boundary_tags)
        
        synth_prob = MechanicalStdProblem(synth_domain, params, 'Uniaxial', synth_forcings, SSLP_LinearElas, MechanicalPrimalCG, orders=[1])
        synth_prob.solve(np.abs(bcvs), "mysynthsol")

        # Sample the synthetic solution: Program this according to your problem
        Nsampx, Nsampy  = 1, 1
        Dx, Dy = Ls[0] / Nsampx, Ls[1] / Nsampy
        xls = np.linspace(0.5*Dx, Ls[0]-0.5*Dx, Nsampx)
        yls = np.linspace(0.5*Dy, Ls[1]-0.5*Dy, Nsampy)
        xlsg, ylsg = np.meshgrid(xls, yls)
        xs = np.dstack([xlsg, ylsg]).reshape(Nsampx*Nsampy, 2)

        #- Pay attention: this is a concious decision
        # and it may change in other situations
        stdsyn_sol = synth_prob.std_solution
        g = fd.project(stdsyn_sol.g, synth_domain.TDG0)
        s = fd.project(stdsyn_sol.s, synth_domain.TDG0)
        symgradu_sampled_vals = np.array(g.at(xs))
        sigma_sampled_vals = np.array(s.at(xs))
        #-------------------------------------------        
        # Apply any manipulation/transformation you need to define
        # the synthetic data (rotations, scalings, removing outliers, etc)
        for angle in np.linspace(0.0, 2.*np.pi, Nth, endpoint=False):
            R = RotationMatrix(angle)
            symgradu_samples[count] = R.T @ symgradu_sampled_vals @ R
            sigma_samples[count] = R.T @ sigma_sampled_vals @ R
            print(sigma_sampled_vals)
            count += 1

    ddcm_data.SetSampledData(symgradu_samples, sigma_samples, add_origin=False, data_file='myDATA.txt')

    ddcm_data.Distribute(ref_prob.std_solution, 'Project-TDG0', flag_plot=True)

    raise Exception('ldldldldld')

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution: %d data points' %(ddcm_data.sampled_stress.shape[0]))
    print('----------------------------------------------------------')

    # 3. Solve the Data Driven problem: This problem should share the same 
    #    forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g,gdata,s,sdata):
        return fd.inner(g-gdata, g-gdata)*fd.dx + fd.inner(s-sdata, s-sdata)*fd.dx
    def myDerObjFunc(g, gdata, dg, s, sdata, dsi):
        return [fd.inner(dsi, s-sdata)*fd.dx, fd.inner(dg, g-gdata)*fd.dx]

    ddcm_prob = DDCMMechProblem(ddcm_data, myObjFunc, myDerObjFunc, 'Indent-Block', forcings, NLSP5FA_MECH, orders=[0,1])

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

# Example using the simple linear Elasticity model to generate the reference solution and sample the data from it
def main_Elas_EX01_():

    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')
    
    print('----------------------------------------------------------')
    print('|::| Reference Domain and solution begins ...')
    print('---------------------------------------------')

    # 1. Create (or import) the computational domain/mesh and Compute a reference solution

    nsd, Ls, Ns = 2, [1.0, 1.0], [100, 100]
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    #________________________________
    # These are the only things I need to specify for an Indent-Block
    def mybodyforce(x):
        return fd.as_vector([fd.Constant(0.0),fd.Constant(-0.05)]) 
    def mysurfaceforceTop(x):
        return fd.as_vector([fd.Constant(0.0), -0.05*exp(-5.0*(x[0])**2)])
    mysurfaceforce = {'Top': mysurfaceforceTop}
    boundary_tags = {'Left': 1 , 'Right': 2, 'Bottom': 3, 'Top': 4}
    forcings = MechanicalForcingData(mybodyforce, mysurfaceforce, None, boundary_tags)
    #________________________________

    params = {'YoungMod': fd.Constant(1.0), 'PoissonRat': fd.Constant(0.475)}

    ref_prob = MechanicalStdProblem(domain, params, 'Stretch-Block', forcings, SSLP_LinearElas, MechanicalPrimalCG, orders=[1])

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
    ddcm_data = MechanicalDDCMData(domain)

    nsd, Ls, Ns = 2, [1.0, 1.0], [100, 100]
    synth_domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    synth_forcings = forcings
        
    synth_prob = MechanicalStdProblem(synth_domain, params, 'Stretch-Block', synth_forcings, SSLP_LinearElas, MechanicalPrimalCG, orders=[1])
    synth_prob.solve(0.0, "mysynthsol")

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
    g = fd.project(stdsyn_sol.g, synth_domain.TDG0)
    s = fd.project(stdsyn_sol.s, synth_domain.TDG0)
    symgradu_sampled_vals = np.array(g.at(xs))
    sigma_sampled_vals = np.array(s.at(xs))
    #-------------------------------------------  
        
    # Apply any manipulation/transformation you need to define
    # the synthetic data (rotations, scalings, removing outliers, etc)
    symgradu_samples = symgradu_sampled_vals
    sigma_samples  = sigma_sampled_vals

    ddcm_data.SetSampledData(symgradu_samples, sigma_samples, add_origin=False, data_file='myDATA.txt')

    ddcm_data.Distribute(ref_prob.std_solution, 'Project-TDG0', flag_plot=True)

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution: %d data points' %(ddcm_data.sampled_stress.shape[0]))
    print('----------------------------------------------------------')

    # 3. Solve the Data Driven problem: This problem should share the same 
    #    forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g,gdata,s,sdata):
        return fd.inner(g-gdata, g-gdata)*fd.dx + fd.inner(s-sdata, s-sdata)*fd.dx
    def myDerObjFunc(g, gdata, dg, s, sdata, dsi):
        return [fd.inner(dsi, s-sdata)*fd.dx, fd.inner(dg, g-gdata)*fd.dx]

    ddcm_prob = DDCMMechProblem(ddcm_data, myObjFunc, myDerObjFunc, 'Indent-Block', forcings, NLSP5FA_MECH, orders=[0,1])

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

# Example using the simple linear Elasticity model to generate the reference solution and sample the data from it
def main_Elas_EX1_():

    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')
    
    print('----------------------------------------------------------')
    print('|::| Reference Domain and solution begins ...')
    print('---------------------------------------------')

    # 1. Create (or import) the computational domain/mesh and Compute a reference solution

    nsd, Ls, Ns = 2, [1.0, 0.01], [100, 30]
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    #________________________________
    # These are the only things I need to specify for an Indent-Block
    def mybodyforce(x):
        return fd.as_vector([fd.Constant(0.0),fd.Constant(-0.001)]) 
    def mysurfaceforceTop(x):
        return fd.as_vector([fd.Constant(0.0), -0.0*exp(-5.0*(x[0])**2)])
    mysurfaceforce = {'Top': mysurfaceforceTop}
    boundary_tags = {'Left': 1 , 'Right': 2, 'Bottom': 3, 'Top': 4}
    forcings = MechanicalForcingData(mybodyforce, mysurfaceforce, None, boundary_tags)
    #________________________________

    params = {'YoungMod': fd.Constant(1.0), 'PoissonRat': fd.Constant(0.0)}

    ref_prob = MechanicalStdProblem(domain, params, 'Beam', forcings, SSLP_LinearElas, MechanicalPrimalCG, orders=[1])

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
    ddcm_data = MechanicalDDCMData(domain)

    nsd, Ls, Ns = 2, [1.0, 0.01], [100, 30]
    synth_domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    synth_forcings = forcings
        
    synth_prob = MechanicalStdProblem(synth_domain, params, 'Beam', synth_forcings, SSLP_LinearElas, MechanicalPrimalCG, orders=[1])
    synth_prob.solve(0.0, "mysynthsol")

    # Sample the synthetic solution: Program this according to your problem

    one_sample_per_element = False
    if(one_sample_per_element == True): # One point per element centroid
        X = interpolate(domain.msh.coordinates, domain.VDG0)
        xs = fd.assemble(X).dat.data
    else: # Sample points one a regular grid
        Nsampx, Nsampy  = 128, 8
        Dx, Dy = Ls[0] / Nsampx, Ls[1] / Nsampy
        xls = np.linspace(0.5*Dx, Ls[0]-0.5*Dx, Nsampx)
        yls = np.linspace(0.5*Dy, Ls[1]-0.5*Dy, Nsampy)
        xlsg, ylsg = np.meshgrid(xls, yls)
        xs = np.dstack([xlsg, ylsg]).reshape(Nsampx*Nsampy, 2)

    #- Pay attention: this is a concious decision
    # and it may change in other situations
    stdsyn_sol = synth_prob.std_solution
    g = fd.project(stdsyn_sol.g, synth_domain.TDG0)
    s = fd.project(stdsyn_sol.s, synth_domain.TDG0)
    symgradu_sampled_vals = np.array(g.at(xs))
    sigma_sampled_vals = np.array(s.at(xs))
    #-------------------------------------------  
        
    # Apply any manipulation/transformation you need to define
    # the synthetic data (rotations, scalings, removing outliers, etc)
    symgradu_samples = symgradu_sampled_vals
    sigma_samples  = sigma_sampled_vals

    ddcm_data.SetSampledData(symgradu_samples, sigma_samples, add_origin=False, data_file='myDATA.txt')

    ddcm_data.Distribute(ref_prob.std_solution, 'Project-TDG0', flag_plot=True)

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution: %d data points' %(ddcm_data.sampled_stress.shape[0]))
    print('----------------------------------------------------------')

    # 3. Solve the Data Driven problem: This problem should share the same 
    #    forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g,gdata,s,sdata):
        return fd.inner(g-gdata, g-gdata)*fd.dx + fd.inner(s-sdata, s-sdata)*fd.dx
    def myDerObjFunc(g, gdata, dg, s, sdata, dsi):
        return [fd.inner(dsi, s-sdata)*fd.dx, fd.inner(dg, g-gdata)*fd.dx]

    ddcm_prob = DDCMMechProblem(ddcm_data, myObjFunc, myDerObjFunc, 'Beam', forcings, NLSP5FA_MECH, orders=[0,1])

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
main_Elas_EX02_()
#npoints_data, hsize, L2error_u, H1error_u, L2error_s = main_EX4_()
#print("npoints= %d\nhsize= %f\nL2error_u= %f\nH1error_u=%f\nL2error_s=%f\n" %(npoints_data, hsize, L2error_u, H1error_u, L2error_s))














#domain = ComputationalDomain(nsd, meshfile='cook_memb.msh', flag_quads=False)

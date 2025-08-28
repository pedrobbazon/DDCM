#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on:         17th May, 2024
Last Modified on:   10th January, 2025

@authors: P. Balbao & R.F. Ausas
"""

# ===============================================
# General libraries
# ===============================================
import numpy as np
import sys
import time

# ===============================================
# Our libraries
# ===============================================
from ddcm_classes import *
from ddcm_utils import *
from PoissonSolvers import *
from DDCMSolvers import *
from save_results_ddcm import *

# ===============================================
# Gustavos's fomulation
# ===============================================
def main_EX01_(formulation='Natural', N=20, order_poly='Linear'):
    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')

    print('----------------------------------------------------------')
    print('|> Reference Domain and solution begins ...')
    print('---------------------------------------------')

    # Create (or import) the computational domain/mesh and Compute a reference solution
    nsd, Ls, Ns = 2, [1.0, 1.0], [N, N]
    hsize = max(Ls[0] / Ns[0], Ls[1] / Ns[1])
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    # These would be the ones corresponding to the analytical solution we propose
    h = fd.CellDiameter(domain.msh)
    params = {'perm': fd.Constant(1.0), 'zeta': fd.Constant(1.0), 'kappa': fd.Constant(1.0),
              'theta': fd.Constant(0.5), 'alpha': fd.Constant(0.125),  'beta': fd.Constant(0.5),
              'gamma': fd.Constant(0.125), 'eta': fd.Constant(0.5),
              'ls': h, 'lmu': h, 'formulation': formulation}

    # Propose an manufactured potential field and its associated gradient and also the exact flux
    def uexact(x):
        return fd.cos(fd.pi * x[0]) * fd.cos(fd.pi * x[1])

    def eexact(x):
        return fd.grad(uexact(x))

    def lmbdexact(x):
        A = fd.Constant(0.05)  # Increase A slightly for more variation
        ome = fd.Constant(2.0)  # Frequency
        phi = fd.Constant(0.0)  # Phase angle
        return - (A / (ome * fd.pi)) * (
                fd.cos(ome * fd.pi * x[0] + phi) + fd.cos(ome * fd.pi * x[1] + phi)
        )

    # Propose a flux-gradient relationship based on the dataset
    def etilde(x):
        A = fd.Constant(0.05)  # Amplitude
        ome = fd.Constant(4.0)  # Frequency
        phi = fd.Constant(0.0)  # Phase angle
        return fd.as_vector([eexact(x)[0] + A * fd.sin(ome * fd.pi * x[0] + phi),
                             eexact(x)[1] + A * fd.sin(ome * fd.pi * x[1] + phi)])

    def sexact(x):
        A = fd.Constant(0.025)  # Amplitude
        eps = fd.Constant(1.0e-10)
        met = fd.sqrt(eexact(x)[0] ** 2 + eexact(x)[1] ** 2) + eps
        etn = fd.as_vector([eexact(x)[0], eexact(x)[1]])
        mst = met # - A * met ** 3
        return fd.as_vector([-mst * etn[0], -mst * etn[1]])

    # by eq.(22)
    def muexact(x):
        return etilde(x) - eexact(x)

    # by eq.(21)
    def fdual(x):
        zeta = params['zeta']
        return zeta * lmbdexact(x) + fd.div(muexact(x))

    # by eq.(23)
    def stilde(x):
        kappa = params['kappa']
        return sexact(x) - (1 / kappa) * fd.grad(lmbdexact(x))

    # by eq.(24)
    def qexact(x):
        zeta = params['zeta']
        return zeta * uexact(x) + fd.div(sexact(x))

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

    forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals, mylmbd_boundary_vals, mymu_boundary_vals,
                                  mydual_source)

    # This will only be called for visualization purposes
    params_anal = {'exact-potential': uexact, 'exact-flux': sexact, 'exact-grad': eexact, 'exact-lmbd': lmbdexact,
                   'exact-mu': muexact, 'exact-etilde': etilde, 'exact-stilde': stilde}
    ref_prob = PoissonStdProblem(domain, params_anal, 'Irrelevant', forcings, None, PoissonAnalytical, orders=[3, 3])

    ref_prob.solve(0.0, "my_gus_refsol")

    print('----------------------------------------------------------')
    print('|> Reference domain and solution: OK')
    print('----------------------------------------------------------')

    # Read or sample some synthetic data from a standard solution:
    # The idea is to produce some plausible data similar to what an experiment would deliver

    # The data will be sampled from the analytical expresion on-the-fly
    ddcm_data = PoissonDDCMData(domain, Analytical=True, fluxex=stilde, gradex=etilde)

    # Solve the Data Driven problem: This problem should share the same
    # forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g, gdata, s, sdata):
        kappa = params['kappa']
        return 0.5 * fd.inner(g - gdata, g - gdata) * fd.dx + 0.5 * kappa * fd.inner(s - sdata, s - sdata) * fd.dx

    if (formulation == 'Natural' and order_poly == 'Linear'):
        ords = [0, 1]
    elif (formulation == 'Natural' and order_poly == 'Quadratic'):
        ords = [1, 2]
    elif (formulation == 'Natural' and order_poly == 'Cubic'):
        ords = [2, 3]
    elif (formulation == 'Equal-Order' and order_poly == 'Linear'):
        ords = [1, 1]
    elif (formulation == 'Equal-Order' and order_poly == 'Quadratic'):
        ords = [2, 2]
    elif (formulation == 'Equal-Order' and order_poly == 'Cubic'):
        ords = [3, 3]
    else:
        raise ValueError('Invalid formulation and polynomial order')
    ddcm_prob = DDCMProblem(ddcm_data, myObjFunc, None, 'Pure-Dirichlet', forcings, DDCMStab, orders=ords,
                            params=params)
    ddcm_prob.solve(0.0, "my_gus_ddcmsol")

    # Compute some error norms
    print('----------------------------------------------------------')
    print('|> Computing some error norms ')
    print('----------------------------------------------------------')
    x = fd.SpatialCoordinate(ddcm_data.domain.msh)

    norms_output = {key: None for key in ['h', 'L2u', 'H1u', 'L2l', 'H1l', 'L2s', 'L2e', 'L2m', 'Hdiv_s', 'Hdiv_e', 'Hdiv_m']}

    norms_output['h'] = hsize
    norms_output['L2u'] = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u, norm_type='L2')
    norms_output['L2l'] = fd.errornorm(lmbdexact(x), ddcm_prob.ddcm_solution.lmd, norm_type='L2')
    norms_output['H1u'] = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u, norm_type='H1')
    norms_output['H1l'] = fd.errornorm(lmbdexact(x), ddcm_prob.ddcm_solution.lmd, norm_type='H1')
    norms_output['L2s'] = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s, norm_type='L2')
    norms_output['L2e'] = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g, norm_type='L2')
    norms_output['L2m'] = fd.errornorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type='L2')

    if (formulation == 'Equal-Order'):
        norms_output['Hdiv_s'] = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s, norm_type='Hdiv')
        norms_output['Hdiv_e'] = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g, norm_type='Hdiv')
        norms_output['Hdiv_m'] = fd.errornorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type='Hdiv')


    print('----------------------------------------------------------')
    print('|> Data Driven solution OK')
    print('----------------------------------------------------------')

    elapsed_time = time.time() - ti
    print('----------------------------------------------------------')
    print('|> End DDCM Computation | Total time= %1.1f s' % (elapsed_time))
    print('----------------------------------------------------------')

    return norms_output

# ===============================================
# Codina's fomulation
# ===============================================
def main_EX02_(formulation: str = 'VMS-Primal', N: int = 20, order_poly: str = 'Linear'):

    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')
    
    print('----------------------------------------------------------')
    print('|> Reference Domain and solution begins')
    print('----------------------------------------------------------')

    # Define the computational domain
    nsd, Ls, Ns = 2, [1.0, 1.0], [N, N]
    hsize = max(Ls[0]/Ns[0], Ls[1]/Ns[1])
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    # Get the characteristic mesh size
    h = fd.CellDiameter(domain.msh)
    
    # Set physical constants and formulation parameters
    params = {'perm': fd.Constant(1.0), 'zeta': fd.Constant(1.0), 'kappa': fd.Constant(1.0),
              'cu': fd.Constant(0.0),
              'ce': fd.Constant(0.0125), 
              'cs': fd.Constant(0.0125), 
              'cl': fd.Constant(0.0),
              'cm': fd.Constant(0.0125), 
              'Pprime': 'L2',
              'L0': h, 
              'L1': fd.Constant(1.0)/h,
              'formulation': formulation} 

    # Propose an manufactured potential field and its associated gradient and also the exact flux
    def uexact(x):
        return fd.cos(fd.pi*x[0])*fd.cos(fd.pi*x[1])
    def eexact(x):
        return fd.grad(uexact(x))

    def lmbdexact(x): 
        A = fd.Constant(0.05)   # Increase A slightly for more variation
        ome = fd.Constant(2.0)  # Frequency
        phi = fd.Constant(0.0)   # Phase angle
        return  - (A / (ome * fd.pi)) * (
            fd.cos(ome * fd.pi * x[0] + phi) + fd.cos(ome * fd.pi * x[1] + phi)
        )

    # Propose a flux-gradient relationship based on the dataset  
    def etilde(x):
        A = fd.Constant(0.05)        # Amplitude
        ome = fd.Constant(4.0)       # Frequency
        phi = fd.Constant(0.0)       # Phase angle
        return fd.as_vector([eexact(x)[0] + A*fd.sin(ome*fd.pi*x[0] + phi),
                             eexact(x)[1] + A*fd.sin(ome*fd.pi*x[1] + phi)])

    def sexact(x):
        A = fd.Constant(0.025)  # Amplitude
        eps = fd.Constant(1.0e-10)
        met = fd.sqrt(eexact(x)[0]**2 + eexact(x)[1]**2) + eps
        etn = fd.as_vector([eexact(x)[0], eexact(x)[1]])
        mst = met # - A*met**3
        return fd.as_vector([-mst*etn[0], -mst*etn[1]])
    
    # by eq.(22)
    def muexact(x):
        return etilde(x) - eexact(x)
    # by eq.(21)
    def fdual(x):
        zeta = params['zeta']
        return zeta*lmbdexact(x) + fd.div(muexact(x))
    # by eq.(23)
    def stilde(x):
        kappa = params['kappa']
        return sexact(x) - (1/kappa)*fd.grad(lmbdexact(x))
    # by eq.(24)
    def qexact(x):
        zeta = params['zeta']
        return zeta*uexact(x) + fd.div(sexact(x))
    
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
    
    # This will only be called for visualization purposes
    params_anal = {'exact-potential': uexact, 'exact-flux': sexact, 'exact-grad': eexact, 'exact-lmbd' : lmbdexact, 
                   'exact-mu': muexact, 'exact-etilde': etilde, 'exact-stilde': stilde}
    ref_prob = PoissonStdProblem(domain, params_anal, 'Irrelevant', forcings, None, PoissonAnalytical, orders=[3,3])

    ref_prob.solve(0.0, "my_cod_refsol")

    print('----------------------------------------------------------')
    print('|> Reference domain and solution: Ok')
    print('----------------------------------------------------------')

    # The data will be sampled from the analytical expresion on-the-fly
    ddcm_data = PoissonDDCMData(domain, Analytical=True, fluxex=stilde, gradex=etilde)
  
    print('----------------------------------------------------------')
    print('|> Data Driven solution begins')
    print('----------------------------------------------------------')

    def myObjFunc(g,gdata,s,sdata):
        kappa = params['kappa']
        return 0.5*fd.inner(g-gdata, g-gdata)*fd.dx + 0.5*kappa*fd.inner(s-sdata, s-sdata)*fd.dx

    if(formulation == 'VMS-Primal' and order_poly == 'Linear'):
        ords = [1,1]
    elif(formulation == 'VMS-Primal' and order_poly == 'Quadratic'):
        ords = [2,2]
    elif(formulation == 'VMS-Primal' and order_poly == 'Cubic'):
        ords = [3,3]
    elif(formulation == 'VMS-Dual' and order_poly == 'Linear'):
        ords = [0,1]
    else:
        raise ValueError('Invalid formulation and polynomial order')
    
    ddcm_prob = DDCMProblem(ddcm_data, myObjFunc, None, 'Pure-Dirichlet', forcings, DDCMStabCod, orders=ords, params=params)

    ddcm_prob.solve(0.0,"my_cod_ddcmsol")

    # Compute some error norms
    print('----------------------------------------------------------')
    print('|> Computing some error norms ')
    print('----------------------------------------------------------')
    x = fd.SpatialCoordinate(ddcm_data.domain.msh)

    norms_output = {key: None for key in ['h', 'L2u', 'H1u', 'L2l', 'H1l', 'L2s', 'L2e', 'L2m', 'Hdiv_s', 'Hdiv_m']}
    
    norms_output['h'] = hsize
    norms_output['L2u'] = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u , norm_type = 'L2')
    norms_output['L2l'] = fd.errornorm(lmbdexact(x), ddcm_prob.ddcm_solution.lmd , norm_type = 'L2')
    norms_output['L2s'] = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s , norm_type = 'L2')
    norms_output['L2e'] = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g , norm_type = 'L2')
    norms_output['L2m'] = fd.errornorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type = 'L2')
        
    if (formulation == 'VMS-Primal'):
        norms_output['H1u'] = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u, norm_type = 'H1')
        norms_output['H1l'] = fd.errornorm(lmbdexact(x), ddcm_prob.ddcm_solution.lmd, norm_type = 'H1')
        norms_output['Hdiv_s'] = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s , norm_type = 'Hdiv')
        norms_output['Hdiv_e'] = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g , norm_type = 'Hdiv')
        norms_output['Hdiv_m'] = fd.errornorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type = 'Hdiv')
    else:
        raise ValueError('Formulation not recognized')

    print('----------------------------------------------------------')
    print('|> Data Driven solution: Ok')
    print('----------------------------------------------------------')
    
    elapsed_time = time.time() - ti
    print('----------------------------------------------------------')
    print('|> End DDCM Computation | Total time= %1.1f s' %(elapsed_time))
    print('----------------------------------------------------------')

    return norms_output

# ===============================================
# Helper functions
# ===============================================
def get_norm_list(formulation: str):
    if (formulation == 'Natural'):
        return ['h', 'L2u', 'H1u','L2l', 'H1l', 'L2s', 'L2e', 'L2m']
    elif (formulation == 'Equal-Order'):
        return ['h', 'L2u', 'H1u','L2l', 'H1l', 'L2s', 'L2e', 'L2m', 'Hdiv_s', 'Hdiv_e' ,'Hdiv_m']
    elif (formulation == 'VMS-Primal'):
        return ['h', 'L2u', 'H1u', 'L2l', 'H1l', 'L2s', 'L2e', 'L2m', 'Hdiv_s', 'Hdiv_e', 'Hdiv_m']
    elif (formulation == 'VMS-Dual'):
        return ['h', 'L2u', 'L2l', 'L2s', 'L2e', 'L2m', 'Hdiv_s', 'Hdiv_m']
    else:
        return ValueError("Invalid formulation")
    
def get_filename_base(formulation):
    return f'table_{formulation}'
    
# ===============================================
# Test case 01: Convex domain 
# ===============================================
main_selector = np.int32(sys.argv[1])

# Comparison between formulations
def run_test_case_01(order_poly: str = 'Linear'):
    # Get the norms according to the formulation employed
    norms_gus = get_norm_list('Equal-Order')
    norms_cod = get_norm_list('VMS-Primal')

    errors_gus = {key: [] for key in norms_gus}
    errors_cod = {key: [] for key in norms_cod}

    for N in [25, 50, 100, 200]:
        print('===================================================')
        print(f' \t Solving Equal-Order with N = {N}')
        print('===================================================')
        # Run main function
        norms_output_gus = main_EX01_(formulation = 'Equal-Order', N = N, order_poly = order_poly)
        norms_output_cod= main_EX02_(formulation = 'VMS-Primal', N = N, order_poly = order_poly)

        # Store the errors
        for key, value in norms_output_gus.items():
            errors_gus[key].append(value)

        for key, value in norms_output_cod.items():
            errors_cod[key].append(value)

    # Save the results
    filename_base_gus = get_filename_base('Equal-Order')
    filename_base_cod = get_filename_base('VMS-Primal')

    os.makedirs(filename_base_gus, exist_ok=True)
    os.makedirs(filename_base_cod, exist_ok=True)

    results_gus = DDCM_save_results(norms_gus, errors_gus, 'Equal-Order')
    results_cod = DDCM_save_results(norms_cod, errors_cod, 'VMS-Primal')

    results_gus.save_results(order_poly, filename_base_gus, '00_Equal-Order')
    results_cod.save_results(order_poly, filename_base_cod, '00_VMS-Primal')

def run_test_case_02(formulation: str = 'VMS-Primal', order_poly: str = 'Linear', foldername: str = 'output'):
    
    # Get the norms according to the formulation employed
    norms = get_norm_list(formulation)
    errors = {key: [] for key in norms}
    
    for N in [25, 50, 100, 200]:
        print('===================================================')
        print(f' \t Solving {formulation} with N = {N}')
        print('===================================================')
        # Run main function
        norms_output = main_EX02_(formulation, N, order_poly)
        
        # Store the errors
        for key, value in norms_output.items():
            errors[key].append(value)
    
    # Save the results
    filename_base = get_filename_base(formulation)
    os.makedirs(foldername, exist_ok=True)
    results = DDCM_save_results(norms, errors, formulation)
    results.save_results(order_poly, filename_base, foldername)
        
# ===============================================
# Run
# ===============================================
if (main_selector == 1):
    run_test_case_01(order_poly='Quadratic')
elif (main_selector == 2):
    run_test_case_02(formulation='VMS-Primal', order_poly='Quadratic', foldername='01_VMS-Primal')

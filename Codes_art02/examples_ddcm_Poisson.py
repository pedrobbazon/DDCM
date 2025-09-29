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
import sys
# ===============================================
# Our libraries
# ===============================================
from ddcm_classes import *
from PoissonSolvers import *
from DDCMSolvers import *
from save_results_ddcm import *
import constants as ct

# ===============================================
# Gustavos's fomulation
# ===============================================
def main_ex01_(formulation='Equal-Order', n=20, order_poly='Linear'):
    ti = time.time()

    print('==========================================================')
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')

    print('----------------------------------------------------------')
    print('|> Reference Domain and solution begins')

    # Create (or import) the computational domain/mesh and Compute a reference solution
    nsd, ls, ns = 2, [1.0, 1.0], [n, n]
    hsize = max(ls[0] / ns[0], ls[1] / ns[1])
    domain = ComputationalDomain(nsd, ls, ns, meshfile=None, flag_quads=False)

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
        etn = fd.as_vector([eexact(x)[0]/met, eexact(x)[1]/met])
        mst = met - A * met ** 3
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

    ref_prob.solve(0.0, ct.MY_REF_SOL)

    print('----------------------------------------------------------')
    print('|> Reference domain and solution: OK')

    # Read or sample some synthetic data from a standard solution:
    # The idea is to produce some plausible data similar to what an experiment would deliver

    # The data will be sampled from the analytical expresion on-the-fly
    ddcm_data = PoissonDDCMData(domain, Analytical=True, fluxex=stilde, gradex=etilde)

    # Solve the Data Driven problem: This problem should share the same
    # forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ')

    def myObjFunc(g, gdata, s, sdata):
        kappa = params['kappa']
        return 0.5 * fd.inner(g - gdata, g - gdata) * fd.dx + 0.5 * kappa * fd.inner(s - sdata, s - sdata) * fd.dx

    if formulation == 'Natural' and order_poly == 'Linear':
        ords = [0, 1]
    elif formulation == 'Natural' and order_poly == 'Quadratic':
        ords = [1, 2]
    elif formulation == 'Natural' and order_poly == 'Cubic':
        ords = [2, 3]
    elif formulation == 'Equal-Order' and order_poly == 'Linear':
        ords = [1, 1]
    elif formulation == 'Equal-Order' and order_poly == 'Quadratic':
        ords = [2, 2]
    elif formulation == 'Equal-Order' and order_poly == 'Cubic':
        ords = [3, 3]
    else:
        raise ValueError('Invalid formulation and polynomial order')
    ddcm_prob = DDCMProblem(ddcm_data, myObjFunc, None, 'Pure-Dirichlet', forcings, DDCMStab, orders=ords,
                            params=params)
    ddcm_prob.solve(0.0, ct.MY_DDCM_SOL)

    # Compute some error norms
    print('----------------------------------------------------------')
    print('|> Computing some error norms ')
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

    if formulation == 'Equal-Order':
        norms_output['Hdiv_s'] = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s, norm_type='Hdiv')
        norms_output['Hdiv_e'] = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g, norm_type='Hdiv')
        norms_output['Hdiv_m'] = fd.errornorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type='Hdiv')

    print('----------------------------------------------------------')
    print('|> Data Driven solution OK')

    elapsed_time = time.time() - ti
    print('|> End DDCM Computation | Total time= %1.1f s' % (elapsed_time))
    print('==========================================================')
    return norms_output

# ===============================================
# Codina's fomulation
# ===============================================
def main_ex02_(formulation: str = 'VMS-Primal', n: int = 20, order_poly: str = 'Linear'):

    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')
    
    print('----------------------------------------------------------')
    print('|> Reference Domain and solution begins')

    # Define the computational domain
    nsd, ls, ns = 2, [1.0, 1.0], [n, n]
    hsize = max(ls[0]/ns[0], ls[1]/ns[1])
    domain = ComputationalDomain(nsd, ls, ns, meshfile=None, flag_quads=False)

    # Get the characteristic mesh size
    h = fd.CellDiameter(domain.msh)
    
    # Set physical constants and formulation parameters
    if formulation == 'VMS-Primal':
        params = {'perm': fd.Constant(1.0), 'zeta': fd.Constant(1.0), 'kappa': fd.Constant(1.0),
                  'cu': fd.Constant(0.0),
                  'ce': fd.Constant(0.125), 
                  'cs': fd.Constant(0.125), 
                  'cl': fd.Constant(-0.0),
                  'cm': fd.Constant(-0.125), 
                  'Pprime': 'Identity',
                  'L0': h, 
                  'L1': fd.Constant(1.0)/h,
                  'formulation': formulation}
    elif formulation == 'VMS-Dual':
        params = {'perm': fd.Constant(1.0), 'zeta': fd.Constant(1.0), 'kappa': fd.Constant(1.0),
                  'cu': fd.Constant(0.125),
                  'ce': fd.Constant(1.25), 
                  'cs': fd.Constant(0.0), 
                  'cl': fd.Constant(-0.125),
                  'cm': fd.Constant(-0.0), 
                  'Pprime': 'Identity',
                  'L0': fd.Constant(1.0), 
                  'L1': fd.Constant(1.0),
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
        etn = fd.as_vector([eexact(x)[0]/met, eexact(x)[1]/met])
        mst = met - A*met**3
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

    ref_prob.solve(0.0, ct.MY_PRIMAL_DDCM_SOL if formulation == 'VMS-Primal' else ct.MY_DUAL_DDCM_SOL)

    print('----------------------------------------------------------')
    print('|> Reference domain and solution: Ok')

    # The data will be sampled from the analytical expression on-the-fly
    ddcm_data = PoissonDDCMData(domain, Analytical=True, fluxex=stilde, gradex=etilde)
  
    print('----------------------------------------------------------')
    print('|> Data Driven solution begins')

    def my_obj_func(g,gdata,s,sdata):
        kappa = params['kappa']
        return 0.5*fd.inner(g-gdata, g-gdata)*fd.dx + 0.5*kappa*fd.inner(s-sdata, s-sdata)*fd.dx

    if order_poly == 'Linear':
        ords = [1,1]
    elif order_poly == 'Quadratic':
        ords = [2,2]
    elif order_poly == 'Cubic':
        ords = [3,3]
    else:
        raise ValueError('Invalid formulation and polynomial order')
    
    ddcm_prob = DDCMProblem(ddcm_data, my_obj_func, None, 'Pure-Dirichlet', forcings, DDCMStabCod, orders=ords, params=params)
    ddcm_prob.solve(0.0,"my_cod_ddcmsol")

    # Compute some error norms
    print('----------------------------------------------------------')
    print('|> Computing some error norms ')
    x = fd.SpatialCoordinate(ddcm_data.domain.msh)

    norms_output = {key: None for key in ['h', 'L2u', 'H1u', 'L2l', 'H1l', 'L2s', 'L2e', 'L2m', 'Hdiv_s', 'Hdiv_m']}
    
    norms_output['h'] = hsize
    norms_output['L2u'] = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u , norm_type = 'L2')
    norms_output['L2l'] = fd.errornorm(lmbdexact(x), ddcm_prob.ddcm_solution.lmd , norm_type = 'L2')
    norms_output['L2s'] = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s , norm_type = 'L2')
    norms_output['L2e'] = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g , norm_type = 'L2')
    norms_output['L2m'] = fd.errornorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type = 'L2')
        
    if formulation == 'VMS-Primal' or formulation == 'VMS-Dual':
        norms_output['H1u'] = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u, norm_type = 'H1')
        norms_output['H1l'] = fd.errornorm(lmbdexact(x), ddcm_prob.ddcm_solution.lmd, norm_type = 'H1')
        norms_output['Hdiv_s'] = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s , norm_type = 'Hdiv')
        norms_output['Hdiv_e'] = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g , norm_type = 'Hdiv')
        norms_output['Hdiv_m'] = fd.errornorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type = 'Hdiv')
    else:
        raise ValueError('Formulation not recognized')

    print('----------------------------------------------------------')
    print('|> Data Driven solution: Ok')
    
    elapsed_time = time.time() - ti
    print('|> End DDCM Computation | Total time= %1.1f s' %elapsed_time)
    print('==========================================================')

    return norms_output

# ===============================================
# Helper functions
# ===============================================
def get_norm_list(formulation: str):
    if formulation == 'Natural':
        return ['h', 'L2u', 'H1u','L2l', 'H1l', 'L2s', 'L2e', 'L2m']
    elif formulation == 'Equal-Order':
        return ['h', 'L2u', 'H1u','L2l', 'H1l', 'L2s', 'L2e', 'L2m', 'Hdiv_s', 'Hdiv_e' ,'Hdiv_m']
    elif formulation == 'VMS-Primal' or formulation == 'VMS-Dual':
        return ['h', 'L2u', 'H1u', 'L2l', 'H1l', 'L2s', 'L2e', 'L2m', 'Hdiv_s', 'Hdiv_e', 'Hdiv_m']
    else:
        return ValueError("Invalid formulation")

# ===============================================
# Run
# ===============================================
main_selector = np.int32(sys.argv[1])

# Compare Gustavo's formulation with Primal and Dual Codina's formulation
def case_01(order_poly: str = 'Linear'):

    list_of_formulations = ['Equal-Order', 'VMS-Primal', 'VMS-Dual']
    norms = {key: [] for key in list_of_formulations}
    for form in norms.keys():
        norms[form] = get_norm_list(form)

    error = {}
    for form, norm_list in norms.items():
        error[form] = {}
        for norm in norm_list:
            error[form][norm] = []

    function_output = {}
    for N in [25, 50, 100, 200]:
        print('===================================================')
        print(f' \t Solving Equal-Order with N = {N}')
        print('===================================================')
        # Run main function
        function_output['Equal-Order'] = main_ex01_(formulation = 'Equal-Order', n = N, order_poly = order_poly)
        function_output['VMS-Primal']= main_ex02_(formulation = 'VMS-Primal', n = N, order_poly = order_poly)
        function_output['VMS-Dual'] = main_ex02_(formulation='VMS-Dual', n = N, order_poly = order_poly)

        # Store the errors
        for form, inner_dict in error.items():
            for norm in inner_dict.keys():
                error[form][norm].append(function_output[form][norm])

    # Save the results
    for form, inner_dict in error.items():
        results = DDCMSaveResults(errors=inner_dict, formulation=form)
        results.save_results(folder_name = ct.RESULTS_CONVERGENCE_COMPARISON, order_poly=order_poly)

# ===============================================
# Main
# ===============================================
if main_selector == 1:
    case_01(order_poly='Linear')


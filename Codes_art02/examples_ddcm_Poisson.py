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

# ===============================================
# Codina's fomulation
# ===============================================
def main_EX01_(formulation: str = 'VMS-PRIMAL', N: int = 20, order_poly: str = 'Linear'):

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
              'ce': fd.Constant(1.0), 
              'cs': fd.Constant(1.0), 
              'cl': fd.Constant(0.0), 
              'cm': fd.Constant(1.0), 
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

    ref_prob.solve(0.0, "myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: Ok')
    print('----------------------------------------------------------')

    # The data will be sampled from the analytical expresion on-the-fly
    ddcm_data = PoissonDDCMData(domain, Analytical=True, fluxex=stilde, gradex=etilde)
  
    print('----------------------------------------------------------')
    print('|> Data Driven solution begins')
    print('----------------------------------------------------------')

    def myObjFunc(g,gdata,s,sdata):
        kappa = params['kappa']
        return 0.5*fd.inner(g-gdata, g-gdata)*fd.dx + 0.5*kappa*fd.inner(s-sdata, s-sdata)*fd.dx

    
    if(formulation == 'VMS-PRIMAL' and order_poly == 'Linear'):
        ords = [1,1]
    elif(formulation == 'VMS-PRIMAL' and order_poly == 'Quadratic'):
        ords = [2,2]
    elif(formulation == 'VMS-PRIMAL' and order_poly == 'Cubic'):
        ords = [3,3]
    elif(formulation == 'VMS-DUAL' and order_poly == 'Linear'):
        ords = [0,1]
    else:
        raise ValueError('Invalid formulation and polynomial order')
    
    ddcm_prob = DDCMProblem(ddcm_data, myObjFunc, None, 'Pure-Dirichlet', forcings, DDCMStabCod, orders=ords, params=params)

    ddcm_prob.solve(0.0,"myddcmsol")

    # Compute some error norms
    print('----------------------------------------------------------')
    print('|> Computing some error norms ')
    print('----------------------------------------------------------')
    x = fd.SpatialCoordinate(ddcm_data.domain.msh)
    L2error_u = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u , norm_type = 'L2')
    H1error_u = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u , norm_type = 'H1')
    L2error_lmd = fd.errornorm(lmbdexact(x), ddcm_prob.ddcm_solution.lmd, norm_type = 'L2')
    H1error_lmd = fd.errornorm(lmbdexact(x), ddcm_prob.ddcm_solution.lmd, norm_type = 'H1')
    L2error_s = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s , norm_type = 'L2')
    L2error_e = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g , norm_type = 'L2')
    L2error_mu = fd.errornorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type = 'L2')

    print('----------------------------------------------------------')
    print('|> Data Driven solution: Ok')
    print('----------------------------------------------------------')
    
    elapsed_time = time.time() - ti
    print('----------------------------------------------------------')
    print('|> End DDCM Computation | Total time= %1.1f s' %(elapsed_time))
    print('----------------------------------------------------------')

    return hsize, L2error_u, H1error_u, L2error_lmd, H1error_lmd, L2error_s, L2error_e, L2error_mu

    

# ===============================================
# Codina's fomulation
# ===============================================
main_selector = np.int32(sys.argv[1])

if (main_selector == 1):
    output = main_EX01_(formulation = 'VMS-PRIMAL', N = 50, order_poly = 'Linear')
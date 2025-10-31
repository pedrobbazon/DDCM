#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on:         17th May, 2024
Last Modified on:   20th October, 2025

@authors: P. Balbao & R.F. Ausas
"""

# ===============================================
# General libraries
# ===============================================
import sys
import firedrake as fd
from firedrake.output import VTKFile

# ===============================================
# Our libraries
# ===============================================
from ddcm_classes import *
from PoissonSolvers import *
from DDCMSolvers import *
from save_results_ddcm import *
from ddcm_utils import *
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
    params = {'perm': fd.Constant(1.0), 'zeta': fd.Constant(1.0), 'difus': fd.Constant(1.0),
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


def main_ex0101_(formulation='Natural', n=20, order_poly='Linear'):
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
    params = {'perm': fd.Constant(1.0), 'zeta': fd.Constant(1.0), 'kappa': fd.Constant(1.0), 'difus': fd.Constant(1.0),
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
    return norms_output, domain, ddcm_prob.ddcm_solution.s, ddcm_prob.ddcm_solution.g

def main_ex0202_(formulation, hvalue, order_poly, kwargs):

    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')
    
    print('----------------------------------------------------------')
    print('|> Reference Domain and solution begins')

    # Get the characteristic mesh size
    Lx, Ly, hsize = 1.0, 1.0, hvalue
    nsd, ls, ns = 2, [1.0, 1.0], [int(Lx/hvalue), int(Ly/hvalue)]
    hsize = max(ls[0] / ns[0], ls[1] / ns[1])
    domain = ComputationalDomain(nsd, ls, ns, meshfile=None, flag_quads=False)
    h = fd.CellDiameter(domain.msh)
    
    # Set physical constants and formulation parameters
    if formulation == 'VMS-Primal':
        params = {'perm': fd.Constant(1.0), 'zeta': fd.Constant(1.0), 'kappa': fd.Constant(1.0), 'difus': fd.Constant(1.0),
                  'cu': fd.Constant(kwargs.get('cu', 0.0)),
                  'ce': fd.Constant(kwargs.get('ce', 0.125)),
                  'cs': fd.Constant(kwargs.get('cs', 0.125)),
                  'cl': fd.Constant(kwargs.get('cl', 0.0)),
                  'cm': fd.Constant(kwargs.get('cm', 0.125)),
                  'Pprime': kwargs.get('Pprime', 'Identity'),
                  'L0': kwargs.get('L0', h),
                  'L1': kwargs.get('L1',  fd.Constant(1.0) / h),
                  'formulation': kwargs.get('formulation','VMS-Primal')}
    elif formulation == 'VMS-Dual':
        params = {'perm': fd.Constant(1.0), 'zeta': fd.Constant(1.0), 'kappa': fd.Constant(1.0), 'difus': fd.Constant(1.0),
                  'cu': fd.Constant(kwargs.get('cu', 0.25)),
                  'ce': fd.Constant(kwargs.get('ce', 0.25)),
                  'cs': fd.Constant(kwargs.get('cs', 0.0)),
                  'cl': fd.Constant(kwargs.get('cl', 0.25)),
                  'cm': fd.Constant(kwargs.get('cm', 0.25)),
                  'Pprime': kwargs.get('Pprime', 'Identity'),
                  'L0': kwargs.get('L0', fd.Constant(1.0)),
                  'L1': kwargs.get('L1',fd.Constant(1.0)),
                  'formulation': kwargs.get('formulation', 'VMS-Dual')}

    # Propose an manufactured potential field and its associated gradient and also the exact flux
    def uexact(x):
        return fd.cos(fd.pi*x[0])*fd.cos(fd.pi*x[1])
    def eexact(x):
        return fd.grad(uexact(x))
    
    def eexactnp(x):
        return [-np.pi*np.sin(np.pi*x[:,0])*np.cos(np.pi*x[:,1]), -np.pi*np.cos(np.pi*x[:,0])*np.sin(np.pi*x[:,1])]

    def lmbdexact(x): 
        A = fd.Constant(0.05)   # Increase A slightly for more variation
        ome = fd.Constant(2.0)  # Frequency
        phi = fd.Constant(0.0)   # Phase angle
        return  - (A / (ome * fd.pi)) * (
            fd.cos(ome * fd.pi * x[0] + phi) + fd.cos(ome * fd.pi * x[1] + phi)
        )
    
    def gradlmbdexactnp(x): 
        A = 0.05   # Increase A slightly for more variation
        ome = 2.0  # Frequency
        phi = 0.0   # Phase angle
        aux = (A / (ome * np.pi))
        return [ aux*ome*np.pi* np.sin(ome * np.pi * x[0] + phi), aux*ome*np.pi * np.sin(ome * np.pi * x[1] + phi)]

    # Propose a flux-gradient relationship based on the dataset  
    def etildeexact(x):
        A = fd.Constant(0.05)        # Amplitude
        ome = fd.Constant(4.0)       # Frequency
        phi = fd.Constant(0.0)       # Phase angle
        return fd.as_vector([eexact(x)[0] + A*fd.sin(ome*fd.pi*x[0] + phi),
                             eexact(x)[1] + A*fd.sin(ome*fd.pi*x[1] + phi)])
    
    def etildeexactnp(x):
        A = 0.05        # Amplitude
        ome = 4.0       # Frequency
        phi = 0.0       # Phase angle
        return [eexactnp(x)[0] + A*np.sin(ome*np.pi*x[:,0] + phi), eexactnp(x)[1] + A*np.sin(ome*np.pi*x[:,1] + phi)]

    def sexact(x):
        A = fd.Constant(0.025)  # Amplitude
        eps = fd.Constant(1.0e-10)
        met = fd.sqrt(eexact(x)[0]**2 + eexact(x)[1]**2) + eps
        etn = fd.as_vector([eexact(x)[0]/met, eexact(x)[1]/met])
        mst = met - A*met**3
        return fd.as_vector([-mst*etn[0], -mst*etn[1]])
    
    def sexactnp(x):
        A = 0.025  # Amplitude
        eps = 1.0e-10
        met = np.sqrt(eexactnp(x)[0]**2 + eexactnp(x)[1]**2) + eps
        etn = [eexactnp(x)[0]/met, eexactnp(x)[1]/met]
        mst = met - A*met**3
        return [-mst*etn[0], -mst*etn[1]]
    
    # by eq.(22)
    def muexact(x):
        return etildeexact(x) - eexact(x)
    
    # by eq.(21)
    def fdual(x):
        zeta = params['zeta']
        return zeta*lmbdexact(x) + fd.div(muexact(x))
    # by eq.(23)
    def stildeexact(x):
        kappa = params['kappa']
        return sexact(x) - (1/kappa)*fd.grad(lmbdexact(x))
    
    def stildeexactnp(x):
        kappa = 1.0
        return [sexactnp(x)[0] - (1/kappa)* gradlmbdexactnp(x[:,0])[0], sexactnp(x)[1] - (1/kappa)* gradlmbdexactnp(x[:,1])[1] ]
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
                   'exact-mu': muexact, 'exact-etilde': etildeexact, 'exact-stilde': stildeexact}
    ref_prob = PoissonStdProblem(domain, params_anal, 'Irrelevant', forcings, None, PoissonAnalytical, orders=[3,3])

    ref_prob.solve(0.0, 'standard_solution')

    print('----------------------------------------------------------')
    print('|> Reference domain and solution: Ok')
    
    x = fd.SpatialCoordinate(domain.msh)
    if(False):
        ddcm_data = PoissonDDCMData(domain, Analytical=False)
        x = fd.SpatialCoordinate(ddcm_data.domain.msh)
        gradu_vals = fd.project(etildeexact(x), domain.VDG0).dat.data
        flux_vals = fd.project(stildeexact(x), domain.VDG0).dat.data
        ddcm_data.SetSampledData(gradu_vals, flux_vals, add_origin=False, data_file=None)
        refsol = DDCMSolution()
        refsol.s = stildeexact(x)
        refsol.g = etildeexact(x)
        ddcm_data.Distribute(refsol, flag_project='Project-VDG0', flag_plot=True)
    elif(False):
        one_sample_per_element = False
        if(one_sample_per_element == True): # One point per element centroid
            X = interpolate(domain.msh.coordinates, domain.VDG0)
            xs = fd.assemble(X).dat.data
        else: # Sample points one a regular grid
            Ls = np.array([Lx, Ly])
            Nsamples = 500
            Nsampx, Nsampy  = Nsamples, Nsamples
            Dx, Dy = Ls[0] / Nsampx, Ls[1] / Nsampy
            xls = np.linspace(0.5*Dx, Ls[0]-0.5*Dx, Nsampx)
            yls = np.linspace(0.5*Dy, Ls[1]-0.5*Dy, Nsampy)
            xlsg, ylsg = np.meshgrid(xls, yls)
            xs = np.dstack([xlsg, ylsg]).reshape(Nsampx*Nsampy, 2)

        # Use the numpy version of the functions eexact and sexact for this:
        gradu_samples = np.array(etildeexactnp(xs)).T
        flux_samples  = np.array(stildeexactnp(xs)).T
        ddcm_data = PoissonDDCMData(domain)
        ddcm_data.SetSampledData(gradu_samples, flux_samples, add_origin=True, data_file='myDATA.txt')
        ddcm_data.DistributeCoordBased(xs, tt=Nsamples ,flag_plot=True)
    else:
        ddcm_data = PoissonDDCMData(domain, Analytical=True, fluxex=stildeexact, gradex=etildeexact)
  
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
    
    ddcm_prob = DDCMProblem(ddcm_data, my_obj_func, None, 'Pure-Neumann', forcings, DDCMStabCod, orders=ords, params=params)
    ddcm_prob.solve(0.0, ct.MY_PRIMAL_DDCM_SOL if formulation == 'VMS-Primal' else ct.MY_DUAL_DDCM_SOL)

    # Compute some error norms
    print('----------------------------------------------------------')
    print('|> Computing some error norms ')
    #x = fd.SpatialCoordinate(ddcm_data.domain.msh)

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
        params = {'perm': fd.Constant(1.0), 'zeta': fd.Constant(1.0), 'kappa': fd.Constant(1.0), 'difus': fd.Constant(1.0),
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
        params = {'perm': fd.Constant(1.0), 'zeta': fd.Constant(1.0), 'kappa': fd.Constant(1.0), 'difus': fd.Constant(1.0),
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


# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================

def main_ex03_(formulation, hvalue, order_poly, kwargs):
    ti = time.time()

    print('==========================================================')
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')

    print('----------------------------------------------------------')
    print('|> Reference Domain and solution begins')

    # Create (or import) the computational domain/mesh and Compute a reference solution
    Lx, Ly, hsize = 2.0, 1.0, hvalue
    meshfile, tags = create_mesh_internal_boundary_layer(Lx, Ly, hsize, flag_refine_interface=False)
    domain = ComputationalDomain(nsd=2, Lsides=[1.0, 1.0], Ndivs=[1, 1], meshfile=meshfile, flag_quads=False)
    h = fd.CellDiameter(domain.msh)
    x = fd.SpatialCoordinate(domain.msh)

    # These would be the ones corresponding to the analytical solution we propose
    params = {'perm': fd.Constant(1.0), 'zeta': fd.Constant(100.0),
              'theta': fd.Constant(kwargs.get('theta', 0.0)), 'kappa': fd.Constant(1.0), 'difus': fd.Constant(1.0),
              'alpha': fd.Constant(kwargs.get('alpha', 0.0)), 'beta': fd.Constant(kwargs.get('beta', 0.0)),
              'gamma': fd.Constant(kwargs.get('gamma', 0.0)), 'eta': fd.Constant(kwargs.get('eta', 0.0)),
              'ls': h, 'lmu': h, 'formulation': formulation}

    # ORTHOGONAL PROJECTION: DISCONTINUOUS SOURCE TERM
    Q = fd.FunctionSpace(domain.msh, "DG", degree=0)
    source = fd.TrialFunction(Q)
    vp = fd.TestFunction(Q)
    ap = source * vp * fd.dx
    sourceL, sourceR = 0.0, params['zeta']
    Lp = fd.Constant(sourceL) * vp * fd.dx(tags['leftdom_marker'])
    Lp += fd.Constant(sourceR) * vp * fd.dx(tags['rightdom_marker'])
    sourceh = fd.Function(Q, name='sigma')
    fd.solve(ap == Lp, sourceh, bcs=[], solver_parameters=petsc_opt1)
    VTKFile("discontinuous_source.pvd").write(sourceh)

    # Propose an manufactured potential field and its associated gradient and also the exact flux
    def uexact(x):
        deltar = fd.sqrt(params["difus"] / params["zeta"])
        u_exp_left = 0.5 * fd.exp((x[0] - 1) / deltar)
        u_exp_right = 1 - 0.5 * fd.exp(-(x[0] - 1) / deltar)
        return fd.conditional(x[0] < Lx / 2, u_exp_left, u_exp_right)

    def eexact(x):
        return fd.grad(uexact(x))

    def sexact(x):
        return - params['difus'] * eexact(x)

    def mysource(x):
        return sourceh

    def mysurfaceflux(x):
        return fd.as_vector([0.0, 0.0])

    def mybcvals(x):
        return uexact(x)

    def mylmbd_boundary_vals(x):
        return fd.Constant(0.0)

    def mymu_boundary_vals(x):
        return fd.as_vector([0.0, 0.0])

    def mydual_source(x):
        return fd.Constant(0.0)

    forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals, mylmbd_boundary_vals, mymu_boundary_vals,
                                  mydual_source)

    # This will only be called for visualization purposes
    params_anal = {'exact-potential': uexact, 'exact-flux': sexact, 'exact-lmbd': mylmbd_boundary_vals, 'exact-mu':
        mymu_boundary_vals}
    ref_prob = PoissonStdProblem(domain, params_anal, 'Pure-Dirichlet', forcings, None, PoissonAnalytical,
                                 orders=[3, 3])
    ref_prob.solve(0.0, ct.MY_REF_SOL)

    print('----------------------------------------------------------')
    print('|> Reference domain and solution: OK')

    # Read or sample some synthetic data from a standard solution:
    # The idea is to produce some plausible data similar to what an experiment would deliver

    # The data will be sampled from the analytical expresion on-the-fly
    ddcm_data = PoissonDDCMData(domain, Analytical=True, fluxex=sexact, gradex=eexact)

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
    ddcm_prob = DDCMProblem(ddcm_data, myObjFunc, None, 'Mixed-Dirichlet-Neumann', forcings, DDCMStab, orders=ords,
                            params=params)
    ddcm_prob.solve(0.0, ct.MY_DDCM_SOL)

    # Compute some error norms
    print('----------------------------------------------------------')
    print('|> Computing some error norms ')
    x = fd.SpatialCoordinate(ddcm_data.domain.msh)

    norms_output = {key: None for key in ['h', 'L2u', 'H1u','L2l', 'H1l', 'L2s', 'L2e','L2m','Hdiv_s', 'Hdiv_e','Hdiv_m']}

    norms_output['h'] = hsize
    norms_output['L2u'] = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u, norm_type='L2')
    norms_output['H1u'] = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u, norm_type='H1')
    norms_output['L2l'] = fd.errornorm(fd.Constant(0.0), ddcm_prob.ddcm_solution.lmd, norm_type='L2')
    norms_output['H1l'] = fd.errornorm(fd.Constant(0.0), ddcm_prob.ddcm_solution.lmd, norm_type='H1')
    norms_output['L2s'] = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s, norm_type='L2')
    norms_output['L2e'] = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g, norm_type='L2')
    norms_output['L2m'] = fd.errornorm(fd.as_vector([0.0,0.0]), ddcm_prob.ddcm_solution.mu, norm_type='L2')

    if formulation == 'Equal-Order':
        norms_output['Hdiv_s'] = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s, norm_type='Hdiv')
        norms_output['Hdiv_e'] = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g, norm_type='Hdiv')
        norms_output['Hdiv_m'] = fd.errornorm(fd.as_vector([0.0, 0.0]), ddcm_prob.ddcm_solution.mu, norm_type='Hdiv')

    print('----------------------------------------------------------')
    print('|> Data Driven solution OK')

    elapsed_time = time.time() - ti
    print('|> End DDCM Computation | Total time= %1.1f s' % (elapsed_time))
    print('==========================================================')
    return norms_output


def main_ex06_(formulation, hvalue, order_poly, kwargs):
    ti = time.time()

    print('==========================================================')
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')

    print('----------------------------------------------------------')
    print('|> Reference Domain and solution begins')

    # Create (or import) the computational domain/mesh and Compute a reference solution
    Lx, Ly, hsize = 2.0, 1.0, hvalue
    meshfile, tags = create_mesh_internal_boundary_layer(Lx, Ly, hsize, flag_refine_interface=False)
    domain = ComputationalDomain(nsd=2, Lsides=[1.0, 1.0], Ndivs=[1, 1], meshfile=meshfile, flag_quads=False)
    h = fd.CellDiameter(domain.msh)
    x = fd.SpatialCoordinate(domain.msh)

    # These would be the ones corresponding to the analytical solution we propose
    params = {'perm': fd.Constant(1.0), 'zeta': fd.Constant(100.0),
              'theta': fd.Constant(kwargs.get('theta', 0.0)), 'kappa': fd.Constant(1.0), 'difus': fd.Constant(0.1),
              'alpha': fd.Constant(kwargs.get('alpha', 0.0)), 'beta': fd.Constant(kwargs.get('beta', 0.0)),
              'gamma': fd.Constant(kwargs.get('gamma', 0.0)), 'eta': fd.Constant(kwargs.get('eta', 0.0)),
              'ls': h, 'lmu': h, 'formulation': formulation}

    # ORTHOGONAL PROJECTION: DISCONTINUOUS SOURCE TERM
    Q = fd.FunctionSpace(domain.msh, "DG", degree=0)
    source = fd.TrialFunction(Q)
    vp = fd.TestFunction(Q)
    ap = source * vp * fd.dx
    sourceL, sourceR = 0.0, params['zeta']
    Lp = fd.Constant(sourceL) * vp * fd.dx(tags['leftdom_marker'])
    Lp += fd.Constant(sourceR) * vp * fd.dx(tags['rightdom_marker'])
    sourceh = fd.Function(Q, name='sigma')
    fd.solve(ap == Lp, sourceh, bcs=[], solver_parameters=petsc_opt1)
    VTKFile("discontinuous_source.pvd").write(sourceh)

    # Propose a manufactured potential field and its associated gradient and also the exact flux
    def uexact(x):
        deltar = fd.sqrt(params["difus"] / params["zeta"])
        u_exp_left = 0.5 * fd.exp((x[0] - 1) / deltar)
        u_exp_right = 1 - 0.5 * fd.exp(-(x[0] - 1) / deltar)
        return fd.conditional(x[0] < Lx / 2, u_exp_left, u_exp_right)

    def eexact(x):
        return fd.grad(uexact(x))

    def sexact(x):
        return - params['difus'] * eexact(x)

    def mysource(x):
        return sourceh
    
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

    def muexact(x):
        return etilde(x) - eexact(x)

    def mysurfaceflux(x):
        return sexact(x)
    
    def fdual(x):
        zeta = params['zeta']
        return zeta * lmbdexact(x) + fd.div(muexact(x))
    
    def stilde(x):
        kappa = params['kappa']
        return sexact(x) - (1 / kappa) * fd.grad(lmbdexact(x))

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
    ref_prob = PoissonStdProblem(domain, params_anal, 'Pure-Dirichlet', forcings, None, PoissonAnalytical,
                                 orders=[3, 3])
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
    ddcm_prob = DDCMProblem(ddcm_data, myObjFunc, None, 'Mixed-Dirichlet-Neumann', forcings, DDCMStab, orders=ords,
                            params=params)
    ddcm_prob.solve(0.0, ct.MY_DDCM_SOL)

    # Compute some error norms
    print('----------------------------------------------------------')
    print('|> Computing some error norms ')
    x = fd.SpatialCoordinate(ddcm_data.domain.msh)

    norms_output = {key: None for key in ['h', 'L2u', 'H1u','L2l', 'H1l', 'L2s', 'L2e','L2m','Hdiv_s', 'Hdiv_e','Hdiv_m']}

    norms_output['h'] = hsize
    norms_output['L2u'] = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u, norm_type='L2')
    norms_output['H1u'] = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u, norm_type='H1')
    norms_output['L2l'] = fd.errornorm(lmbdexact(x), ddcm_prob.ddcm_solution.lmd, norm_type='L2')
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


def main_ex04_(formulation, hvalue, order_poly, kwargs):

    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')

    print('----------------------------------------------------------')
    print('|> Reference Domain and solution begins')

    # Define the computational domain
    Lx, Ly, hsize = 2.0, 1.0, hvalue
    meshfile, tags = create_mesh_internal_boundary_layer(Lx, Ly, hsize, flag_refine_interface = False)
    domain = ComputationalDomain(nsd=2, Lsides=[1.0,1.0], Ndivs=[1,1], meshfile = meshfile, flag_quads=False)
    h = fd.CellDiameter(domain.msh)
    x = fd.SpatialCoordinate(domain.msh)

    # Set physical constants and formulation parameters
    if formulation == 'VMS-Primal':
        params = {'perm': fd.Constant(1.0), 'zeta': fd.Constant(100.0), 'kappa': fd.Constant(1.0),  'difus': fd.Constant(1.0),
                  'cu': fd.Constant(kwargs.get('cu', 0.0)),
                  'ce': fd.Constant(kwargs.get('ce', 0.125)),
                  'cs': fd.Constant(kwargs.get('cs', 0.125)),
                  'cl': fd.Constant(kwargs.get('cl', 0.0)),
                  'cm': fd.Constant(kwargs.get('cm', 0.125)),
                  'Pprime': kwargs.get('Pprime', 'Identity'),
                  'L0': kwargs.get('L0', h),
                  'L1': kwargs.get('L1',  fd.Constant(1.0) / h),
                  'formulation': kwargs.get('formulation','VMS-Primal')}
    elif formulation == 'VMS-Dual':
        params = {'perm': fd.Constant(1.0), 'zeta': fd.Constant(100.0), 'kappa': fd.Constant(1.0), 'difus': fd.Constant(1.0),
                  'cu': fd.Constant(kwargs.get('cu', 0.125)),
                  'ce': fd.Constant(kwargs.get('ce', 0.125)),
                  'cs': fd.Constant(kwargs.get('cs', 0.0)),
                  'cl': fd.Constant(kwargs.get('cl', 0.125)),
                  'cm': fd.Constant(kwargs.get('cm', 0.0)),
                  'Pprime': kwargs.get('Pprime', 'Identity'),
                  'L0': kwargs.get('L0', fd.Constant(1.0)),
                  'L1': kwargs.get('L1',fd.Constant(1.0)),
                  'formulation': kwargs.get('formulation', 'VMS-Dual')}

    # ORTHOGONAL PROJECTION: DISCONTINUOUS SOURCE TERM
    Q = fd.FunctionSpace(domain.msh, "DG", degree=0)
    source = fd.TrialFunction(Q)
    vp = fd.TestFunction(Q)
    ap = source * vp * fd.dx
    sourceL, sourceR = 0.0, params['zeta']
    Lp = fd.Constant(sourceL) * vp * fd.dx(tags['leftdom_marker'])
    Lp += fd.Constant(sourceR) * vp * fd.dx(tags['rightdom_marker'])
    sourceh = fd.Function(Q, name='sigma')
    fd.solve(ap == Lp, sourceh, bcs=[], solver_parameters = petsc_opt1)
    VTKFile("discontinuous_source.pvd").write(sourceh)

    # Propose a manufactured potential field and its associated gradient and also the exact flux
    def uexact(x):
        deltar = fd.sqrt(params["difus"] / params["zeta"])
        u_exp_left = 0.5 * fd.exp((x[0] - 1) / deltar)
        u_exp_right = 1 - 0.5 * fd.exp(-(x[0] - 1) / deltar)
        return fd.conditional(x[0] < Lx / 2, u_exp_left, u_exp_right)

    def eexact(x):
        return fd.grad(uexact(x))

    def sexact(x):
        return - params['difus'] * eexact(x)

    def mysource(x):
        return sourceh

    def mysurfaceflux(x):
        return fd.as_vector([0.0,0.0])

    def mybcvals(x):
        return uexact(x)

    def mylmbd_boundary_vals(x):
        return fd.Constant(0.0)

    def mymu_boundary_vals(x):
        return fd.as_vector([0.0,0.0])

    def mydual_source(x):
        return fd.Constant(0.0)


    forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals, mylmbd_boundary_vals, mymu_boundary_vals,
                                  mydual_source)

    # This will only be called for visualization purposes
    params_anal = {'exact-potential': uexact, 'exact-flux': sexact, 'exact-lmbd': mylmbd_boundary_vals, 'exact-mu':
                   mymu_boundary_vals}
    ref_prob = PoissonStdProblem(domain, params_anal, 'Pure-Dirichlet', forcings, None, PoissonAnalytical, orders=[3, 3])
    ref_prob.solve(0.0, ct.MY_PRIMAL_DDCM_SOL if formulation == 'VMS-Primal' else ct.MY_DUAL_DDCM_SOL)

    print('----------------------------------------------------------')
    print('|> Reference domain and solution: Ok')

    # The data will be sampled from the analytical expression on-the-fly
    ddcm_data = PoissonDDCMData(domain, Analytical=True, fluxex=sexact, gradex=eexact)

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins')

    def my_obj_func(g, gdata, s, sdata):
        kappa = params['kappa']
        return 0.5 * fd.inner(g - gdata, g - gdata) * fd.dx + 0.5 * kappa * fd.inner(s - sdata, s - sdata) * fd.dx

    if order_poly == 'Linear':
        ords = [1, 1]
    elif order_poly == 'Quadratic':
        ords = [2, 2]
    elif order_poly == 'Cubic':
        ords = [3, 3]
    else:
        raise ValueError('Invalid formulation and polynomial order')

    ddcm_prob = DDCMProblem(ddcm_data, my_obj_func, None, 'Pure-Dirichlet', forcings, DDCMStabCod, orders=ords,
                            params=params)
    ddcm_prob.solve(0.0, "my_cod_ddcmsol")

    # Compute some error norms
    print('----------------------------------------------------------')
    print('|> Computing some error norms ')
    x = fd.SpatialCoordinate(ddcm_data.domain.msh)

    norms_output = {key: None for key in ['h', 'L2u', 'H1u','L2l', 'H1l' 'L2s', 'L2e','L2m', 'Hdiv_s', 'Hdiv_e','Hdiv_m']}

    norms_output['h'] = hsize
    norms_output['L2u'] = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u, norm_type='L2')
    norms_output['L2l'] = fd.errornorm(fd.Constant(0.0), ddcm_prob.ddcm_solution.lmd, norm_type='L2')
    norms_output['L2s'] = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s, norm_type='L2')
    norms_output['L2e'] = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g, norm_type='L2')
    norms_output['L2m'] = fd.errornorm(fd.as_vector([0.0,0.0]), ddcm_prob.ddcm_solution.mu, norm_type='L2')

    if formulation == 'VMS-Primal':
        norms_output['H1u'] = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u, norm_type='H1')
        norms_output['H1l'] = fd.errornorm(fd.Constant(0.0),ddcm_prob.ddcm_solution.lmd, norm_type='H1')
        norms_output['Hdiv_s'] = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s, norm_type='Hdiv')
        norms_output['Hdiv_e'] = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g, norm_type='Hdiv')
        norms_output['Hdiv_m'] = fd.errornorm(fd.as_vector([0.0,0.0]), ddcm_prob.ddcm_solution.mu,norm_type='Hdiv')

    elif formulation == 'VMS-Dual':
        norms_output['Hdiv_s'] = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s, norm_type='Hdiv')
        norms_output['Hdiv_e'] = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g, norm_type='Hdiv')
        norms_output['Hdiv_m'] = fd.errornorm(fd.as_vector([0.0, 0.0]), ddcm_prob.ddcm_solution.mu, norm_type='Hdiv')
    else:
        raise ValueError('Formulation not recognized')

    print('----------------------------------------------------------')
    print('|> Data Driven solution: Ok')

    elapsed_time = time.time() - ti
    print('|> End DDCM Computation | Total time= %1.1f s' % elapsed_time)
    print('==========================================================')

    return norms_output

def main_ex05_(formulation, hvalue, order_poly, kwargs):

    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')

    print('----------------------------------------------------------')
    print('|> Reference Domain and solution begins')

    # Define the computational domain
    Lx, Ly, hsize = 2.0, 1.0, hvalue
    meshfile, tags = create_mesh_internal_boundary_layer(Lx, Ly, hsize, flag_refine_interface = False)
    domain = ComputationalDomain(nsd=2, Lsides=[1.0,1.0], Ndivs=[1,1], meshfile = meshfile, flag_quads=False)
    h = fd.CellDiameter(domain.msh)
    x = fd.SpatialCoordinate(domain.msh)

    # Set physical constants and formulation parameters
    if formulation == 'VMS-Primal':
        params = {'perm': fd.Constant(1.0), 'zeta': fd.Constant(100.0), 'kappa': fd.Constant(1.0), 'difus': fd.Constant(1.0),
                  'cu': fd.Constant(kwargs.get('cu', 0.0)),
                  'ce': fd.Constant(kwargs.get('ce', 0.125)),
                  'cs': fd.Constant(kwargs.get('cs', 0.125)),
                  'cl': fd.Constant(kwargs.get('cl', 0.0)),
                  'cm': fd.Constant(kwargs.get('cm', 0.125)),
                  'Pprime': kwargs.get('Pprime', 'Identity'),
                  'L0': kwargs.get('L0', h),
                  'L1': kwargs.get('L1',  fd.Constant(1.0) / h),
                  'formulation': kwargs.get('formulation','VMS-Primal')}
    elif formulation == 'VMS-Dual':
        params = {'perm': fd.Constant(1.0), 'zeta': fd.Constant(100.0), 'kappa': fd.Constant(1.0), 'difus': fd.Constant(1.0),
                  'cu': fd.Constant(kwargs.get('cu', 0.25)),
                  'ce': fd.Constant(kwargs.get('ce', 0.25)),
                  'cs': fd.Constant(kwargs.get('cs', 0.0)),
                  'cl': fd.Constant(kwargs.get('cl', 0.25)),
                  'cm': fd.Constant(kwargs.get('cm', 0.25)),
                  'Pprime': kwargs.get('Pprime', 'Identity'),
                  'L0': kwargs.get('L0', fd.Constant(1.0)),
                  'L1': kwargs.get('L1',fd.Constant(1.0)),
                  'formulation': kwargs.get('formulation', 'VMS-Dual')}

    # ORTHOGONAL PROJECTION: DISCONTINUOUS SOURCE TERM
    Q = fd.FunctionSpace(domain.msh, "DG", degree=0)
    source = fd.TrialFunction(Q)
    vp = fd.TestFunction(Q)
    ap = source * vp * fd.dx
    sourceL, sourceR = 0.0, params['zeta']
    Lp = fd.Constant(sourceL) * vp * fd.dx(tags['leftdom_marker'])
    Lp += fd.Constant(sourceR) * vp * fd.dx(tags['rightdom_marker'])
    sourceh = fd.Function(Q, name='sigma')
    fd.solve(ap == Lp, sourceh, bcs=[], solver_parameters = petsc_opt1)
    VTKFile("discontinuous_source.pvd").write(sourceh)

    # Propose a manufactured potential field and its associated gradient and also the exact flux
    def uexact(x):
        deltar = fd.sqrt(params["difus"] / params["zeta"])
        u_exp_left = 0.5 * fd.exp((x[0] - 1) / deltar)
        u_exp_right = 1 - 0.5 * fd.exp(-(x[0] - 1) / deltar)
        return fd.conditional(x[0] < Lx / 2, u_exp_left, u_exp_right)

    def eexact(x):
        return fd.grad(uexact(x))

    def sexact(x):
        return - params['difus'] * eexact(x)

    def mysource(x):
        return sourceh
    
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

    def muexact(x):
        return etilde(x) - eexact(x)

    def mysurfaceflux(x):
        return sexact(x)
    
    def fdual(x):
        zeta = params['zeta']
        return zeta * lmbdexact(x) + fd.div(muexact(x))
    
    def stilde(x):
        kappa = params['kappa']
        return sexact(x) - (1 / kappa) * fd.grad(lmbdexact(x))

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
    ref_prob = PoissonStdProblem(domain, params_anal, 'Pure-Dirichlet', forcings, None, PoissonAnalytical, orders=[3, 3])
    ref_prob.solve(0.0, ct.MY_PRIMAL_DDCM_SOL if formulation == 'VMS-Primal' else ct.MY_DUAL_DDCM_SOL)

    print('----------------------------------------------------------')
    print('|> Reference domain and solution: Ok')

    # The data will be sampled from the analytical expression on-the-fly
    ddcm_data = PoissonDDCMData(domain, Analytical=True, fluxex=stilde, gradex=etilde)

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins')

    def my_obj_func(g, gdata, s, sdata):
        kappa = params['kappa']
        return 0.5 * fd.inner(g - gdata, g - gdata) * fd.dx + 0.5 * kappa * fd.inner(s - sdata, s - sdata) * fd.dx

    if order_poly == 'Linear':
        ords = [1, 1]
    elif order_poly == 'Quadratic':
        ords = [2, 2]
    elif order_poly == 'Cubic':
        ords = [3, 3]
    else:
        raise ValueError('Invalid formulation and polynomial order')

    ddcm_prob = DDCMProblem(ddcm_data, my_obj_func, None, 'Pure-Neumann', forcings, DDCMStabCod, orders=ords,
                            params=params)
    ddcm_prob.solve(0.0, "my_cod_ddcmsol")

    # Compute some error norms
    print('----------------------------------------------------------')
    print('|> Computing some error norms ')
    x = fd.SpatialCoordinate(ddcm_data.domain.msh)

    norms_output = {key: None for key in ['h', 'L2u', 'H1u','L2l', 'H1l' 'L2s', 'L2e','L2m', 'Hdiv_s', 'Hdiv_e','Hdiv_m']}

    norms_output['h'] = hsize
    norms_output['L2u'] = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u, norm_type='L2')
    norms_output['L2l'] = fd.errornorm(lmbdexact(x), ddcm_prob.ddcm_solution.lmd, norm_type='L2')
    norms_output['L2s'] = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s, norm_type='L2')
    norms_output['L2e'] = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g, norm_type='L2')
    norms_output['L2m'] = fd.errornorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type='L2')

    if formulation == 'VMS-Primal':
        norms_output['H1u'] = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u, norm_type='H1')
        norms_output['H1l'] = fd.errornorm(lmbdexact(x),ddcm_prob.ddcm_solution.lmd, norm_type='H1')
        norms_output['Hdiv_s'] = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s, norm_type='Hdiv')
        norms_output['Hdiv_e'] = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g, norm_type='Hdiv')
        norms_output['Hdiv_m'] = fd.errornorm(muexact(x), ddcm_prob.ddcm_solution.mu,norm_type='Hdiv')

    elif formulation == 'VMS-Dual':
        norms_output['Hdiv_s'] = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s, norm_type='Hdiv')
        norms_output['Hdiv_e'] = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g, norm_type='Hdiv')
        norms_output['Hdiv_m'] = fd.errornorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type='Hdiv')
    else:
        raise ValueError('Formulation not recognized')

    print('----------------------------------------------------------')
    print('|> Data Driven solution: Ok')

    elapsed_time = time.time() - ti
    print('|> End DDCM Computation | Total time= %1.1f s' % elapsed_time)
    print('==========================================================')

    return norms_output


def main_ex07_(formulation='Natural', hsize=0.1, case_corner=0, order_poly = 'Linear', **kwargs):
    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')

    print('----------------------------------------------------------')
    print('|::| Reference Domain and solution begins ...')
    print('---------------------------------------------')

    #  Create (or import) the computational domain/mesh and Compute a reference solution
    lc = hsize  # characteristic length of the mesh
    flag_singularity_refinement = True

    mymeshfile, tags = CreateMeshReentrantCorner('reentrant', case_corner, lc, flag_singularity_refinement)
    if (case_corner == 0):
        omega = 1.75 * fd.pi
    elif (case_corner == 1):
        omega = 1.5 * fd.pi
    elif (case_corner == 2):
        omega = 1.25 * fd.pi
    elif (case_corner == 3):
        omega = 0.5 * fd.pi

    nsd, Ls, Ns = 2, [1.0, 1.0], [100, 100]
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=mymeshfile, flag_quads=False)
    h = fd.CellDiameter(domain.msh)
    x = fd.SpatialCoordinate(domain.msh)

    # These would be the ones corresponding to the analytical solution we propose
    params = {'perm': fd.Constant(1.0), 'zeta': fd.Constant(1.0),
              'theta': fd.Constant(kwargs.get('theta', 0.0)), 'kappa': fd.Constant(1.0),
              'alpha': fd.Constant(kwargs.get('alpha', 0.0)), 'beta': fd.Constant(kwargs.get('beta', 0.0)),
              'gamma': fd.Constant(kwargs.get('gamma', 0.0)), 'eta': fd.Constant(kwargs.get('eta', 0.0)),
              'ls': h, 'lmu': h, 'formulation': formulation}

    # We define the analytical expression for the potential field and its gradient
    def uexact(x):
        angle = fd.atan2(x[1] + 1e-12, x[0] + 1e-12)
        theta = fd.conditional(fd.gt(angle, 0), angle, angle + 2 * fd.pi)
        r = fd.sqrt(x[0] ** 2 + x[1] ** 2) + 1e-12
        alpha = fd.pi / omega
        return (r ** alpha) * fd.sin(alpha * theta)

    def eexact(x):
        return fd.grad(uexact(x))

    # Propose an analytical expression for lambda
    def lmbdexact(x):
        A = fd.Constant(0.05)  # Increase A slightly for more variation
        ome = fd.Constant(2.0)  # Frequency
        phi = fd.Constant(0.0)  # Phase angle
        return - (A / (ome * fd.pi)) * (fd.cos(ome * fd.pi * x[0] + phi) + fd.cos(ome * fd.pi * x[1] + phi))

    # Propose a flux-gradient relationship based on the dataset
    def etilde(x):
        A = fd.Constant(0.05)  # Amplitude
        ome = fd.Constant(4.0)  # Frequency
        phi = fd.Constant(0.0)  # Phase angle
        return fd.as_vector([eexact(x)[0] + A * fd.sin(ome * fd.pi * x[0] + phi),
                             eexact(x)[1] + A * fd.sin(ome * fd.pi * x[1] + phi)])

    def sexact(x):
        A = fd.Constant(0.0)  # 0.025 Amplitude
        eps = fd.Constant(1.0e-10)
        met = fd.sqrt(eexact(x)[0] ** 2 + eexact(x)[1] ** 2) + eps
        etn = fd.as_vector([eexact(x)[0] / met, eexact(x)[1] / met])
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

    ref_prob.solve(0.0, "myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: OK')
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

    if (formulation == 'Natural'):
        ords = [0, 1]
    elif (formulation == 'Equal-Order'):
        ords = [1, 1]
    else:
        raise ValueError(f"Unknown {formulation}")
    ddcm_prob = DDCMProblem(ddcm_data, myObjFunc, None, 'Pure-Dirichlet', forcings, DDCMStab, orders=ords,
                            params=params)

    ddcm_prob.solve(0.0, f"myddcmsol_{formulation}")

    # Compute some error norms
    norms_output = {key: None for key in
                    ['h', 'L2u', 'H1u', 'L2l', 'H1l' 'L2s', 'L2e', 'L2m', 'Hdiv_s', 'Hdiv_e', 'Hdiv_m',
                     'H1semi_u', 'H1semi_l', 'Hdivsemi_s', 'Hdivsemi_e', 'Hdivsemi_m']}

    norms_output['h'] = hsize
    norms_output['L2u'] = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u, norm_type='L2')
    norms_output['H1u'] = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u, norm_type='H1')
    norms_output['H1semi_u'] = error_seminorm(uexact(x), ddcm_prob.ddcm_solution.u, norm_type='H1')

    norms_output['L2l'] = fd.errornorm(lmbdexact(x), ddcm_prob.ddcm_solution.lmd, norm_type='L2')
    norms_output['H1l'] = fd.errornorm(lmbdexact(x), ddcm_prob.ddcm_solution.lmd, norm_type='H1')
    norms_output['H1semi_l'] = error_seminorm(lmbdexact(x), ddcm_prob.ddcm_solution.lmd, norm_type='H1')

    norms_output['L2s'] = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s, norm_type='L2')
    norms_output['L2e'] = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g, norm_type='L2')
    norms_output['L2m'] = fd.errornorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type='L2')

    if formulation == 'Equal-Order':
        norms_output['Hdiv_s'] = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s, norm_type='Hdiv')
        norms_output['Hdivsemi_s'] = error_seminorm(sexact(x), ddcm_prob.ddcm_solution.s, norm_type='Hdiv')

        norms_output['Hdiv_e'] = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g, norm_type='Hdiv')
        norms_output['Hdivsemi_e'] = error_seminorm(eexact(x), ddcm_prob.ddcm_solution.g, norm_type='Hdiv')

        norms_output['Hdiv_m'] = fd.errornorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type='Hdiv')
        norms_output['Hdivsemi_m'] = error_seminorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type='Hdiv')

    print('----------------------------------------------------------')
    print('|> Data Driven solution OK')
    print('----------------------------------------------------------')

    elapsed_time = time.time() - ti
    print('----------------------------------------------------------')
    print('|> End DDCM Computation | Total time= %1.1f s' % (elapsed_time))
    print('----------------------------------------------------------')


    return norms_output


def main_ex08_(formulation, hsize=0.1, case_corner=0,  order_poly = 'Linear', **kwargs):
    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')

    print('----------------------------------------------------------')
    print('|::| Reference Domain and solution begins ...')
    print('---------------------------------------------')

    #  Create (or import) the computational domain/mesh and Compute a reference solution
    lc = hsize  # characteristic length of the mesh
    flag_singularity_refinement = True

    mymeshfile, tags = CreateMeshReentrantCorner('reentrant', case_corner, lc, flag_singularity_refinement)
    if (case_corner == 0):
        omega = 1.75 * fd.pi
    elif (case_corner == 1):
        omega = 1.5 * fd.pi
    elif (case_corner == 2):
        omega = 1.25 * fd.pi
    elif (case_corner == 3):
        omega = 0.5 * fd.pi

    nsd, Ls, Ns = 2, [1.0, 1.0], [100, 100]
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=mymeshfile, flag_quads=False)

    # These would be the ones corresponding to the analytical solution we propose
    h = fd.CellDiameter(domain.msh)
    x = fd.SpatialCoordinate(domain.msh)

    # Set physical constants and formulation parameters
    if formulation == 'VMS-Primal':
        params = {'perm': fd.Constant(1.0), 'zeta': fd.Constant(1.0), 'kappa': fd.Constant(1.0),
                  'difus': fd.Constant(1.0),
                  'cu': fd.Constant(kwargs.get('cu', 0.0)),
                  'ce': fd.Constant(kwargs.get('ce', 0.125)),
                  'cs': fd.Constant(kwargs.get('cs', 0.125)),
                  'cl': fd.Constant(kwargs.get('cl', 0.0)),
                  'cm': fd.Constant(kwargs.get('cm', 0.125)),
                  'Pprime': kwargs.get('Pprime', 'Identity'),
                  'L0': kwargs.get('L0', h),
                  'L1': kwargs.get('L1', fd.Constant(1.0) / h),
                  'formulation': kwargs.get('formulation', 'VMS-Primal')}
    elif formulation == 'VMS-Dual':
        params = {'perm': fd.Constant(1.0), 'zeta': fd.Constant(1.0), 'kappa': fd.Constant(1.0),
                  'difus': fd.Constant(1.0),
                  'cu': fd.Constant(kwargs.get('cu', 0.25)),
                  'ce': fd.Constant(kwargs.get('ce', 0.25)),
                  'cs': fd.Constant(kwargs.get('cs', 0.0)),
                  'cl': fd.Constant(kwargs.get('cl', 0.25)),
                  'cm': fd.Constant(kwargs.get('cm', 0.25)),
                  'Pprime': kwargs.get('Pprime', 'Identity'),
                  'L0': kwargs.get('L0', fd.Constant(1.0)),
                  'L1': kwargs.get('L1', fd.Constant(1.0)),
                  'formulation': kwargs.get('formulation', 'VMS-Dual')}

    # We define the analytical expression for the potential field and its gradient
    def uexact(x):
        angle = fd.atan2(x[1] + 1e-12, x[0] + 1e-12)
        theta = fd.conditional(fd.gt(angle, 0), angle, angle + 2 * fd.pi)
        r = fd.sqrt(x[0] ** 2 + x[1] ** 2) + 1e-12
        alpha = fd.pi / omega
        return (r ** alpha) * fd.sin(alpha * theta)

    def eexact(x):
        return fd.grad(uexact(x))

    # Propose an analytical expression for lambda
    def lmbdexact(x):
        A = fd.Constant(0.05)  # Increase A slightly for more variation
        ome = fd.Constant(2.0)  # Frequency
        phi = fd.Constant(0.0)  # Phase angle
        return - (A / (ome * fd.pi)) * (fd.cos(ome * fd.pi * x[0] + phi) + fd.cos(ome * fd.pi * x[1] + phi))

    # Propose a flux-gradient relationship based on the dataset
    def etilde(x):
        A = fd.Constant(0.05)  # Amplitude
        ome = fd.Constant(4.0)  # Frequency
        phi = fd.Constant(0.0)  # Phase angle
        return fd.as_vector([eexact(x)[0] + A * fd.sin(ome * fd.pi * x[0] + phi),
                             eexact(x)[1] + A * fd.sin(ome * fd.pi * x[1] + phi)])

    def sexact(x):
        A = fd.Constant(0.0)  # 0.025 Amplitude
        eps = fd.Constant(1.0e-10)
        met = fd.sqrt(eexact(x)[0] ** 2 + eexact(x)[1] ** 2) + eps
        etn = fd.as_vector([eexact(x)[0] / met, eexact(x)[1] / met])
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
    ref_prob = PoissonStdProblem(domain, params_anal, 'Irrelevant', forcings, None, PoissonAnalytical,
                                 orders=[3, 3])

    ref_prob.solve(0.0, "myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: OK')
    print('----------------------------------------------------------')

    # Read or sample some synthetic data from a standard solution:
    # The idea is to produce some plausible data similar to what an experiment would deliver

    # The data will be sampled from the analytical expresion on-the-fly
    ddcm_data = PoissonDDCMData(domain, Analytical=True, fluxex=stilde, gradex=etilde)

    # Solve the Data Driven problem: This problem should share the same
    # forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins')

    def my_obj_func(g, gdata, s, sdata):
        kappa = params['kappa']
        return 0.5 * fd.inner(g - gdata, g - gdata) * fd.dx + 0.5 * kappa * fd.inner(s - sdata, s - sdata) * fd.dx

    ords = [1, 1]
    ddcm_prob = DDCMProblem(ddcm_data, my_obj_func, None, 'Pure-Dirichlet', forcings, DDCMStabCod, orders=ords,
                            params=params)
    ddcm_prob.solve(0.0, f"mycodddcmsol_{formulation}")

    # Compute some error norms
    print('----------------------------------------------------------')
    print('|> Computing some error norms ')
    x = fd.SpatialCoordinate(ddcm_data.domain.msh)

    # Compute some error norms
    norms_output = {key: None for key in
                    ['h', 'L2u', 'H1u', 'L2l', 'H1l' 'L2s', 'L2e', 'L2m', 'Hdiv_s', 'Hdiv_e', 'Hdiv_m',
                     'H1semi_u', 'H1semi_l', 'Hdivsemi_s', 'Hdivsemi_e', 'Hdivsemi_m']}

    norms_output['h'] = hsize
    norms_output['L2u'] = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u, norm_type='L2')
    norms_output['L2l'] = fd.errornorm(lmbdexact(x), ddcm_prob.ddcm_solution.lmd, norm_type='L2')
    norms_output['L2s'] = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s, norm_type='L2')
    norms_output['L2e'] = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g, norm_type='L2')
    norms_output['L2m'] = fd.errornorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type='L2')

    if formulation == 'VMS-Primal':
        norms_output['H1u'] = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u, norm_type='H1')
        norms_output['H1semi_u'] = error_seminorm(uexact(x), ddcm_prob.ddcm_solution.u, norm_type='H1')

        norms_output['H1l'] = fd.errornorm(lmbdexact(x), ddcm_prob.ddcm_solution.lmd, norm_type='H1')
        norms_output['H1semil'] = error_seminorm(lmbdexact(x), ddcm_prob.ddcm_solution.lmd, norm_type='H1')

        norms_output['Hdiv_s'] = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s, norm_type='Hdiv')
        norms_output['Hdivsemi_s'] = error_seminorm(sexact(x), ddcm_prob.ddcm_solution.s, norm_type='Hdiv')

        norms_output['Hdiv_e'] = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g, norm_type='Hdiv')
        norms_output['Hdivsemi_e'] = error_seminorm(eexact(x), ddcm_prob.ddcm_solution.g, norm_type='Hdiv')

        norms_output['Hdiv_m'] = fd.errornorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type='Hdiv')
        norms_output['Hdivsemi_m'] =error_seminorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type='Hdiv')

    elif formulation == 'VMS-Dual':
        norms_output['Hdiv_s'] = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s, norm_type='Hdiv')
        norms_output['Hdivsemi_s'] = error_seminorm(sexact(x), ddcm_prob.ddcm_solution.s, norm_type='Hdiv')

        norms_output['Hdiv_e'] = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g, norm_type='Hdiv')
        norms_output['Hdivsemi_e'] = error_seminorm(eexact(x), ddcm_prob.ddcm_solution.g, norm_type='Hdiv')

        norms_output['Hdiv_m'] = fd.errornorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type='Hdiv')
        norms_output['Hdivsemi_m'] = error_seminorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type='Hdiv')
    else:
        raise ValueError('Formulation not recognized')

    print('----------------------------------------------------------')
    print('|> Data Driven solution: Ok')

    elapsed_time = time.time() - ti
    print('|> End DDCM Computation | Total time= %1.1f s' % elapsed_time)
    print('==========================================================')

    return norms_output



# ===============================================
# Helper functions
# ===============================================
def get_norm_list(formulation: str):
    if formulation == 'Natural':
        return ['h', 'L2u', 'H1u','L2l', 'H1l', 'L2s', 'L2e','L2m']
    elif formulation == 'Equal-Order-uns' or formulation == 'Equal-Order-fully':
        return ['h', 'L2u', 'H1u','L2l', 'H1l', 'L2s', 'L2e', 'L2m', 'Hdiv_s', 'Hdiv_e', 'Hdiv_m']
    elif formulation == 'Primal-ASGS' or formulation == 'Primal-OSGS':
        return ['h', 'L2u', 'H1u','L2l', 'H1l', 'L2s', 'L2e', 'L2m', 'Hdiv_s', 'Hdiv_e', 'Hdiv_m']
    elif formulation == 'Dual-ASGS' or formulation == 'Dual-OSGS':
        return ['h', 'L2u', 'L2l', 'L2s', 'L2e', 'L2m', 'Hdiv_s', 'Hdiv_e', 'Hdiv_m']
    else:
        return ValueError("Invalid formulation")

def get_stab_params(formulation):
    if formulation == 'Natural':
        return {}
    elif formulation == 'Equal-Order-uns':
        return {}
    elif formulation == 'Equal-Order-fully':
        return {'theta': 0.5, 'alpha': 0.125, 'beta': 0.5, 'gamma': 0.125, 'eta': 0.5}
    elif formulation == 'Primal-ASGS' or formulation == 'Dual-ASGS':
        return {'Pprime': 'Identity'}
    elif formulation == 'Primal-OSGS' or formulation == 'Dual-OSGS':
        return {'Pprime': 'OSS'}
    else:
        raise ValueError("Invalid formulation")


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


# Compare Gustavo's formulation with Primal and Dual Codina's formulation
def case_02(order_poly: str = 'Linear'):

    #list_of_formulations = ['Natural','Equal-Order-uns','Equal-Order-fully',
    #                        'Primal-ASGS', 'Primal-OSGS','Dual-ASGS','Dual-OSGS']
    #list_of_formulations = ['Natural','Equal-Order-uns','Equal-Order-fully','Primal-ASGS','Dual-ASGS']
    #list_of_formulations = ['Primal-ASGS', 'Dual-ASGS']
    #list_of_formulations = ['Dual-ASGS']
    list_of_formulations = ['Primal-ASGS']
    
    norms = {key: [] for key in list_of_formulations}

    for form in norms.keys():
        norms[form] = get_norm_list(form)

    error = {}
    for form, norm_list in norms.items():
        error[form] = {}
        for norm in norm_list:
            error[form][norm] = []

    function_output = {}
    for h in [0.05, 0.025, 0.0125]:
        print('===================================================')
        print(f' \t Solving h = {h}')
        print('===================================================')
        # Run main function
        #function_output['Natural'] = main_ex06_(formulation = 'Natural', hvalue = h, order_poly = order_poly,
        #                                        kwargs ={})

        #function_output['Equal-Order-uns']= main_ex06_(formulation = 'Equal-Order', hvalue = h, order_poly = order_poly,
        #                                               kwargs = {})

        #function_output['Equal-Order-fully'] = main_ex06_(formulation='Equal-Order', hvalue=h, order_poly=order_poly,
        #                                                  kwargs=get_stab_params('Equal-Order-fully'))

        #function_output['Primal-ASGS'] = main_ex05_(formulation='VMS-Primal', hvalue = h, order_poly = order_poly,
        #                                            kwargs = get_stab_params('Primal-ASGS'))

        #function_output['Primal-OSGS'] = main_ex04_(formulation='VMS-Primal', hvalue=h, order_poly=order_poly,
        #                                            kwargs=get_stab_params('Primal-OSGS'))

        function_output['Dual-ASGS'] = main_ex05_(formulation='VMS-Dual', hvalue=h, order_poly=order_poly,
                                                  kwargs=get_stab_params('Dual-ASGS'))

        #function_output['Dual-OSGS'] = main_ex04_(formulation='VMS-Dual', hvalue=h, order_poly=order_poly,
        #                                            kwargs=get_stab_params('Dual-OSGS'))

        # Store the errors
        for form, inner_dict in error.items():
            for norm in inner_dict.keys():
                error[form][norm].append(function_output[form][norm])

    # Save the results
    for form, inner_dict in error.items():
        results = DDCMSaveResults(errors=inner_dict, formulation=form)
        results.save_results(folder_name = 'Convergence_new', order_poly=order_poly)

# Compare Gustavo's formulation with Primal and Dual Codina's formulation
def case_03(order_poly: str = 'Linear'):
 
    list_of_formulations = ['Primal-ASGS', 'Dual-ASGS']
    
    norms = {key: [] for key in list_of_formulations}

    for form in norms.keys():
        norms[form] = get_norm_list(form)

    error = {}
    for form, norm_list in norms.items():
        error[form] = {}
        for norm in norm_list:
            error[form][norm] = []

    function_output = {}
    for h in [0.025]:
        print('===================================================')
        print(f' \t Solving h = {h}')
        print('===================================================')
        # Run main function
        #function_output['Natural'], domain, stilde, etilde = main_ex0101_(formulation = 'Natural', order_poly = order_poly)
        #
        #function_output['Primal-ASGS'] = main_ex0202_(formulation='VMS-Primal', hvalue=h,
        #                                              order_poly = order_poly, kwargs=get_stab_params('Primal-ASGS'))

        function_output['Dual-ASGS'] = main_ex0202_(formulation='VMS-Dual', hvalue=h, 
                                                    order_poly=order_poly, kwargs=get_stab_params('Dual-ASGS'))

        # Store the errors
        for form, inner_dict in error.items():
            for norm in inner_dict.keys():
                error[form][norm].append(function_output[form][norm])

    # Save the results
    for form, inner_dict in error.items():
        results = DDCMSaveResults(errors=inner_dict, formulation=form)
        results.save_results(folder_name = 'Convergence_new', order_poly=order_poly)


# Compare Gustavo's formulation with Primal and Dual Codina's formulation
def case_04(order_poly = 'Linear'):
    list_of_formulations = ['Natural', 'Equal-Order-uns', 'Equal-Order-fully', 'Primal-ASGS', 'Dual-ASGS']
    norms = {key: [] for key in list_of_formulations}

    for form in norms.keys():
        norms[form] = get_norm_list(form)

    error = {}
    for form, norm_list in norms.items():
        error[form] = {}
        for norm in norm_list:
            error[form][norm] = []

    function_output = {}
    for h in [0.2, 0.1, 0.05, 0.025]:
        print('===================================================')
        print(f' \t Solving h = {h}')
        print('===================================================')
        # Run main function
        function_output['Natural'] = main_ex07_(formulation='Natural', hsize=h,case_corner=0,  order_poly = 'Linear',
                                                kwargs={})

        function_output['Equal-Order-uns'] = main_ex07_(formulation='Equal-Order', hsize=h,case_corner=0, order_poly = 'Linear',
                                                        kwargs={})

        function_output['Equal-Order-fully'] = main_ex07_(formulation='Equal-Order', hsize=h, case_corner=0,  order_poly = 'Linear',
                                                          kwargs=get_stab_params('Equal-Order-fully'))

        function_output['Primal-ASGS'] = main_ex08_(formulation='VMS-Primal', hsize=h,case_corner=0,  order_poly = 'Linear',
                                                    kwargs=get_stab_params('Primal-ASGS'))

        function_output['Dual-ASGS'] = main_ex08_(formulation='VMS-Dual', hsize=h,case_corner=0,  order_poly = 'Linear',
                                                  kwargs=get_stab_params('Dual-ASGS'))


        # Store the errors
        for form, inner_dict in error.items():
            for norm in inner_dict.keys():
                error[form][norm].append(function_output[form][norm])

    # Save the results
    for form, inner_dict in error.items():
        results = DDCMSaveResults(errors=inner_dict, formulation=form)
        results.save_results(folder_name='Convergence_reentrant', order_poly=order_poly)

# Compare Gustavo's formulation with Primal and Dual Codina's formulation
def case_05(order_poly = 'Linear'):

    h = 0.025
    # Run main function
    _ = main_ex07_(formulation='Natural', hsize=h,case_corner=0,  order_poly = 'Linear',
                                            kwargs={})

    _ = main_ex07_(formulation='Equal-Order', hsize=h,case_corner=0, order_poly = 'Linear',
                                                    kwargs={})

    _ = main_ex07_(formulation='Equal-Order', hsize=h, case_corner=0,  order_poly = 'Linear',
                                                      kwargs=get_stab_params('Equal-Order-fully'))

    _ = main_ex08_(formulation='VMS-Primal', hsize=h,case_corner=0,  order_poly = 'Linear',
                                                kwargs=get_stab_params('Primal-ASGS'))

    _ = main_ex08_(formulation='VMS-Dual', hsize=h,case_corner=0,  order_poly = 'Linear',
                                              kwargs=get_stab_params('Dual-ASGS'))



# ===============================================
# Main
# ===============================================
if main_selector == 1:
    case_01(order_poly='Linear')

elif main_selector == 2:
    case_02(order_poly='Quadratic')

elif main_selector == 3:
    case_03(order_poly='Linear')

elif main_selector == 4:
    case_04(order_poly='Linear')

elif main_selector == 5:
    case_05(order_poly='Linear')


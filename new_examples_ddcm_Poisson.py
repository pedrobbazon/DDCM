#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on:         24th March, 2025
Last Modified on:   24th March, 2025

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

# On the effect of stabilization parameters and polynomial degree in the natural/equal-order formulation
def main_EX1_(formulation='Natural', N=20, Nsamples=3, **kwargs):

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
              'theta': fd.Constant(kwargs.get('theta', 0.0)), 	'kappa': fd.Constant(1.0), 
              'alpha': fd.Constant(kwargs.get('alpha', 0.0)), 	'beta': fd.Constant(kwargs.get('beta', 0.0)), 
              'gamma': fd.Constant(kwargs.get('gamma', 0.0)), 	'eta': fd.Constant(kwargs.get('eta', 0.0)),
              'ls': h, 'lmu': h, 'formulation': formulation} 
    
    #________________________________
    def mysource(x):
        return fd.Constant(1.0)
    def mysurfaceflux(x):
        return None
    def mybcvals(x):
        return fd.Constant(0.123456789)
    def mydual_source(x):
        return fd.Constant(0.0)
    def mylmbd_boundary_vals(x):
        return fd.Constant(0.0)
    forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals,lmbd_boundary_vals=mylmbd_boundary_vals, dual_volumetric_source=mydual_source)
    #________________________________

    params_ref = {'eta': fd.Constant(1.e-12), 'mu_over_kappa': fd.Constant(1.0), 'beta_x_rho': fd.Constant(10.0)}
    ref_prob = PoissonStdProblem(domain, params_ref, 'Pure-Dirichlet', forcings, FLM_ForcheimerReg, PoissonDualMixedHdiv, orders=[2,1])

    ref_prob.solve(0.0, "myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: OK')
    print('----------------------------------------------------------')

    # 2. Read or sample some synthetic data from a standard solution: 
    #    The idea is to produce some plausible data similar to what an experiment would deliver

    # The data will be sampled from the analytical expresion

    one_sample_per_element = False
    if(one_sample_per_element == True): # One point per element centroid
        X = interpolate(domain.msh.coordinates, domain.VDG0)
        xs = fd.assemble(X).dat.data
    else: # Sample points one a regular grid
        Nsampx, Nsampy  = Nsamples, Nsamples
        Dx, Dy = Ls[0] / Nsampx, Ls[1] / Nsampy
        xls = np.linspace(0.5*Dx, Ls[0]-0.5*Dx, Nsampx)
        yls = np.linspace(0.5*Dy, Ls[1]-0.5*Dy, Nsampy)
        xlsg, ylsg = np.meshgrid(xls, yls)
        xs = np.dstack([xlsg, ylsg]).reshape(Nsampx*Nsampy, 2)

    #- Pay attention: this is a concious decision
    # and it may change in other situations
    gproj = fd.project(ref_prob.std_solution.g, domain.VDG0, name='Reference.Grad(u)')
    gradu_sampled_vals = np.array(gproj.at(xs))
    flux_sampled_vals = np.array(ref_prob.std_solution.s.at(xs))
    #-------------------------------------------  
        
    # Apply any manipulation/transformation you need to define
    # the synthetic data (rotations, scalings, removing outliers, etc)
    gradu_samples = gradu_sampled_vals
    flux_samples  = flux_sampled_vals

    ddcm_data = PoissonDDCMData(domain)
    ddcm_data.SetSampledData(gradu_samples, flux_samples, add_origin=True, data_file='myDATA.txt')
    
    ddcm_data.DistributeCoordBased(xs, tt=Nsamples ,flag_plot=True)

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution: %d data points' %(ddcm_data.sampled_grad.shape[0]))
    print('----------------------------------------------------------')

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
    ddcm_prob = DDCMProblem(ddcm_data, myObjFunc, None, 'Pure-Dirichlet', forcings, DDCMStab, orders=ords, params=params)

    ddcm_prob.solve(Nsamples, "myddcmsol")

    print('----------------------------------------------------------')
    print('|> Data Driven solution OK')
    print('----------------------------------------------------------')
    
    elapsed_time = time.time() - ti
    print('----------------------------------------------------------')
    print('|> End DDCM Computation | Total time= %1.1f s' %(elapsed_time))
    print('----------------------------------------------------------')

    return 0
    
# On the effect of stabilization parameters and polynomial degree in the natural/equal-order formulation
def main_EX2_(formulation='Natural', N=20, Nsamples=3, higher_order_poly = False, **kwargs):

    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')
    
    print('----------------------------------------------------------')
    print('|::| Reference Domain and solution begins ...')
    print('---------------------------------------------')

    # 1. Create (or import) the computational domain/mesh and Compute a reference solution

    nsd, Ls, Ns = 2, [1.0, 1.0], [N, N]
    hsize = max(Ls[0]/Ns[0], Ls[1]/Ns[1])
    mymeshfile, tags = CreateMeshSquare('squaremesh', hsize)
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=mymeshfile, flag_quads=False)
    #________________________________
    # These would be the ones corresponding to the analytical solution we propose

    h = fd.CellDiameter(domain.msh)

    params = {'perm': fd.Constant(1.0), 'zeda': fd.Constant(1.0), 
              'theta': fd.Constant(kwargs.get('theta', 0.0)), 	'kappa': fd.Constant(1.0), 
              'alpha': fd.Constant(kwargs.get('alpha', 0.0)), 	'beta': fd.Constant(kwargs.get('beta', 0.0)), 
              'gamma': fd.Constant(kwargs.get('gamma', 0.0)), 	'eta': fd.Constant(kwargs.get('eta', 0.0)),
              'ls': h, 'lmu': h, 'formulation': formulation} 
    
    
    # Propose an manufactured potential field and its associated gradient and also the exact flux
    #def uexact(x):
    #    return fd.cos(fd.pi*x[0])*fd.cos(fd.pi*x[1])
    
    def uexact(x):
        return x[0]*(x[0] - Ls[0])*x[1]*(x[1] - Ls[1])

    #def eexactnp(x):
    #    return [-np.pi*np.sin(np.pi*x[:,0])*np.cos(np.pi*x[:,1]), -np.pi*np.cos(np.pi*x[:,0])*np.sin(np.pi*x[:,1])]
    def eexactnp(x):
        return [(2*x[:,0] - Ls[0])*x[:,1]*(x[:,1] - Ls[1]), (2*x[:,1] - Ls[1])*x[:,0]*(x[:,0] - Ls[0])]

    def eexact(x):
        return fd.grad(uexact(x))

    def lmbdexact(x): 
        return fd.Constant(0.0)
    
    def sexact(x):
        return -fd.grad(uexact(x))
    
    #def sexactnp(x):
    #    return [np.pi*np.sin(np.pi*x[:,0])*np.cos(np.pi*x[:,1]), np.pi*np.cos(np.pi*x[:,0])*np.sin(np.pi*x[:,1])]
    def sexactnp(x):
        return [-(2*x[:,0] - Ls[0])*x[:,1]*(x[:,1] - Ls[1]), -(2*x[:,1] - Ls[1])*x[:,0]*(x[:,0] - Ls[0])]
    
    def muexact(x):
        return fd.as_vector([fd.Constant(0.0), fd.Constant(0.0)])
    def fdual(x):
        return fd.Constant(0.0)
    def qexact(x):
        zeda = params['zeda']
        return zeda*uexact(x) + fd.div(sexact(x))
            
    forcings = PoissonForcingData(qexact, sexact, uexact, lmbdexact, muexact, fdual)
    #________________________________

    # This will only be called for visualization purposes
    params_anal = {'exact-potential': uexact, 'exact-flux': sexact, 'exact-grad': eexact, 'exact-lmbd' : lmbdexact, 
                   'exact-mu': muexact, 'exact-etilde': None, 'exact-stilde': None}
    ref_prob = PoissonStdProblem(domain, params_anal, 'Irrelevant', forcings, None, PoissonAnalytical, orders=[3,3])

    ref_prob.solve(0.0, "myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: OK')
    print('----------------------------------------------------------')

    # 2. Read or sample some synthetic data from a standard solution: 
    #    The idea is to produce some plausible data similar to what an experiment would deliver

    # The data will be sampled from the analytical expresion

    one_sample_per_element = False
    if(one_sample_per_element == True): # One point per element centroid
        X = interpolate(domain.msh.coordinates, domain.VDG0)
        xs = fd.assemble(X).dat.data
    else: # Sample points one a regular grid
        Nsampx, Nsampy  = Nsamples, Nsamples
        Dx, Dy = Ls[0] / Nsampx, Ls[1] / Nsampy
        xls = np.linspace(0.5*Dx, Ls[0]-0.5*Dx, Nsampx)
        yls = np.linspace(0.5*Dy, Ls[1]-0.5*Dy, Nsampy)
        xlsg, ylsg = np.meshgrid(xls, yls)
        xs = np.dstack([xlsg, ylsg]).reshape(Nsampx*Nsampy, 2)

    # Use the numpy version of the functions eexact and sexact for this:
    gradu_samples = np.array(eexactnp(xs)).T
    flux_samples  = np.array(sexactnp(xs)).T
    ddcm_data = PoissonDDCMData(domain)
    ddcm_data.SetSampledData(gradu_samples, flux_samples, add_origin=True, data_file='myDATA.txt')
    ddcm_data.DistributeCoordBased(xs, tt=Nsamples ,flag_plot=True)

    # 3. Solve the Data Driven problem: This problem should share the same 
    #    forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g,gdata,s,sdata):
        kappa = params['kappa']
        return 0.5*fd.inner(g-gdata, g-gdata)*fd.dx + 0.5*kappa*fd.inner(s-sdata, s-sdata)*fd.dx

    if(formulation == 'Natural' and higher_order_poly == False):
        ords = [0,1]
    elif(formulation == 'Natural' and higher_order_poly == True):
        ords = [1,2]
    elif(formulation == 'Equal-Order' and higher_order_poly == False):
        ords = [1,1]
    elif(formulation == 'Equal-Order' and higher_order_poly == True):
        ords = [2,2]
    else:
        raise ValueError('Invalid formulation and polynomial order')
    
    ddcm_prob = DDCMProblem(ddcm_data, myObjFunc, None, 'Pure-Dirichlet', forcings, DDCMStab, orders=ords, params=params)
    ddcm_prob.solve(tt=Nsamples,dir_outs="myddcmsol")

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

def plotErrors(hhs=[], L2u=[], H1u=[], L2l=[], H1l=[], EsL2=[], EeL2=[], EmL2=[], EsHdiv=[], EeHdiv=[],
               EmHdiv=[], formulation='Natural', flag_save_fig=False, fig_names = [], flag_plot_fig = True):
    fig, ax = plt.subplots(figsize=(8,6))
    if L2u:
        ax.loglog(hhs,L2u,'-or',label='$\\|u-u_h\\|_{L^2}$')
    if H1u:
        ax.loglog(hhs,H1u,'-ob',label='$\\|u-u_h\\|_{H^1}$')
    if L2l:
        ax.loglog(hhs,L2l,'-og',label='$\\|\\lambda-\\lambda_h\\|_{L^2}$')
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
        
    
def computeConvergenceRates(hhs, errors):
    """Compute convergence rates for each error type."""
    conv_rates = {}
    for key, value in errors.items():
        if np.all(np.isnan(value)):
            conv_rates[key] = [np.nan] 
        else:
            p = (np.log(value[-1] / value[-2]) / np.log(hhs[-1] / hhs[-2]))
            conv_rates[key] = [p]  
    return conv_rates

from tabulate import tabulate
def printConvergenceRates(conv_rates):
    headers = ["Norm", "Convergence rate"]
    table_data = [[norm] + [f"{rate:.4}" for rate in val] for norm, val in conv_rates.items()]
    print("Estimated Convergence Rates:")
    print(tabulate(table_data, headers=headers, tablefmt="github"))
    
   
# ===== RUN ===== #
import sys

main_selector = np.int32(sys.argv[1])

if(main_selector == 2):
    
    # Configuration
    #formulation = 'Natural'  
    formulation = 'Equal-Order'
    
    if formulation == 'Natural':
        stab_params = {None: None}
        fig_val_list = ['u', 'l', 's_e_mu']
        num_plots = 3
    elif formulation == 'Equal-Order':
        # stab_params = {'theta': 0.5, 'alpha': 0.125, 'beta': 0.5, 'gamma': 0.125, 'eta': 0.5, 
        #                'No stabilization': None,
        #                'Fully stabilized':{'theta': 0.5, 'alpha': 0.125, 'beta': 0.5, 'gamma': 0.125, 'eta': 0.5}, 
        #                'Partially stabilized':{'alpha': 0.125, 'beta': 0.5, 'eta': 0.5}
        #                }
        stab_params = {'Fully stabilized':{'theta': 0.5, 'alpha': 0.125, 'beta': 0.5, 'gamma': 0.125, 'eta': 0.5}}
        #stab_params = {'Fully stabilized':{'theta': 0.5, 'alpha': 0.125, 'beta': 0.5, 'gamma': 0.125, 'eta': 0.5}, 
        #               'Partially stabilized':{'theta': 0.5, 'alpha': 0.125, 'beta': 0.5, 'eta': 0.5}
        #              }
        fig_val_list = ['uL2', 'uH1', 'lL2', 'lH1', 'sL2', 'sHdiv', 'eL2', 'eHdiv', 'muL2', 'muHdiv']
        num_plots = 10
    else:
        raise ValueError('Invalid formulation')

    # Line styles for plotting
    line_styles = {'theta': '-o', 'alpha': '--s', 'beta': '-.^', 'gamma': ':p', 'eta': '--d', 
                   'No stabilization': '-.h', 
                   'Fully stabilized': ':v',
                   'Partially stabilized': '-<'
                   }
    
    # Initialize all plots
    def initializePlots(num_plots, formulation):
        """Initialize all plots with labels, titles, and grids."""
        for fig_num in range(1, num_plots + 1):
            plt.figure(fig_num, figsize=(8, 6))
            plt.xlabel('$h$', fontsize=14)
            plt.ylabel('Error', fontsize=14)
            # plt.title(formulation, fontsize=14)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            plt.tick_params(axis='both', which='major', labelsize=14)
            plt.tick_params(axis='both', which='minor', labelsize=14)

    # Plot lines on a specific figure
    def plotLines(fig_num, hhs, data, label, line_type='-o'):
        """Plot lines on an existing figure."""
        plt.figure(fig_num)
        plt.loglog(hhs, data, line_type, linewidth=2, label=label)

    # Main loop
    """Main logic for plotting and saving figures."""
    initializePlots(num_plots, formulation)

    for param, val in stab_params.items():
        kwargs = {} if formulation == 'Natural' or param == 'No stabilization' else {param: val}
        print(f'==== Solving for {kwargs} ====')
        Nhs = [25, 50, 100, 200]
        Nsamples = [16,32,64,128,256,512,1024,2048]
        for N in Nsamples:
            print('||------------------------------------------------')
            print('||> Running mesh convergence for Nsamples=', N)
            hhs, errors = [], {key: [] for key in ['L2u', 'H1u', 'L2l', 'H1l', 'EsL2', 'EeL2', 'EmL2', 'EsHdiv', 'EeHdiv', 'EmHdiv']}
            for k, Nh in enumerate(Nhs):
                output = main_EX2_(formulation, Nh, N, **kwargs)
                hhs.append(output[0])
                for i, key in enumerate(errors.keys()):
                    errors[key].append(output[i + 1])
            print('||> Done Nsamples=', N)
            print('||------------------------------------------------')
            # Compute convergence rates
            conv_rates = computeConvergenceRates(hhs, errors)
            printConvergenceRates(conv_rates)

            # Plot data
            if formulation == 'Natural':
                plot_data = [
                    (1, [errors['L2u'], errors['H1u']], ['$\\|u - u_h\\|_{L^2}$', '$\\|u - u_h\\|_{H^1}$'], ['-o', '-o']),
                    (2, [errors['L2l'], errors['H1l']], ['$\\|\\lambda - \\lambda_h\\|_{L^2}$', '$\\|\\lambda - \\lambda_h\\|_{H^1}$'], ['--s', '--s']),
                    (3, [errors['EsL2'], errors['EeL2'], errors['EmL2']], 
                    ['$\\|\\mathbf{s} - \\mathbf{s}_h\\|_{L^2}$', '$\\|\\mathbf{e} - \\mathbf{e}_h\\|_{L^2}$', '$\\|\\boldsymbol{\\mu} - \\boldsymbol{\\mu}_h\\|_{L^2}$'], 
                    ['-.^', ':p', '--d'])
                ]
            else:
                plot_data = [
                    (i + 1, [errors[key]], 
                        [f'{label} $\, \\{param}={val}$' if param not in ['No stabilization', 'Fully stabilized', 'Partially stabilized']
                        else f'{label} $\,\\alpha=0.125, \\beta=0.5, \\eta=0.5$' if param == 'Partially stabilized' 
                        else f'{label} $\,$ {param}'], 
                        [line_styles[param]]
                    )
                    for i, (key, label) in enumerate(zip(
                        ['L2u', 'H1u', 'L2l', 'H1l', 'EsL2', 'EsHdiv', 'EeL2', 'EeHdiv', 'EmL2', 'EmHdiv'],
                        ['$\\|u - u_h\\|_{L^2}$', '$\\|u - u_h\\|_{H^1}$', '$\\|\\lambda - \\lambda_h\\|_{L^2}$',
                        '$\\|\\lambda - \\lambda_h\\|_{H^1}$', '$\\|\\mathbf{s} - \\mathbf{s}_h\\|_{L^2}$',
                        '$\\|\\mathbf{s} - \\mathbf{s}_h\\|_{Hdiv}$', '$\\|\\mathbf{e} - \\mathbf{e}_h\\|_{L^2}$',
                        '$\\|\\mathbf{e} - \\mathbf{e}_h\\|_{Hdiv}$', '$\\|\\boldsymbol{\\mu} - \\boldsymbol{\\mu}_h\\|_{L^2}$',
                        '$\\|\\boldsymbol{\\mu} - \\boldsymbol{\\mu}_h\\|_{Hdiv}$']
                    ))
                ]

            for fig_num, data_list, labels, line_types in plot_data:
                for data, label, line_type in zip(data_list, labels, line_types):
                    plotLines(fig_num, hhs, data, label, line_type)

            plt.show()
else:
        
    print('\n   ||||>>> Main selector option not recognized: ', main_selector)

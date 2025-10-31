#-----------------------------------
# General libraries
#-----------------------------------
import sys
import time
#-----------------------------------
# Our libraries
#-----------------------------------
from ddcm_classes import *
from ddcm_utils import *
from PoissonSolvers import *
from DDCMSolvers import *
from save_results_ddcm import *

#----------------------------------------------------------------------------
# Numerical experiments: test case 01 - Convex domain
#----------------------------------------------------------------------------
def main_EX1_(formulation='Natural', N=20, order_poly = 'Linear', **kwargs):

    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')
    
    print('----------------------------------------------------------')
    print('|::| Reference Domain and solution begins ...')
    print('---------------------------------------------')

    # Create (or import) the computational domain/mesh and Compute a reference solution
    nsd, Ls, Ns = 2, [1.0, 1.0], [N, N]
    hsize = max(Ls[0]/Ns[0], Ls[1]/Ns[1])
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)

    
    # These would be the ones corresponding to the analytical solution we propose
    h = fd.CellDiameter(domain.msh)
    params = {'perm': fd.Constant(1.0), 'zeda': fd.Constant(1.0), 
              'theta': fd.Constant(kwargs.get('theta', 0.0)), 	'kappa': fd.Constant(1.0), 
              'alpha': fd.Constant(kwargs.get('alpha', 0.0)), 	'beta': fd.Constant(kwargs.get('beta', 0.0)), 
              'gamma': fd.Constant(kwargs.get('gamma', 0.0)), 	'eta': fd.Constant(kwargs.get('eta', 0.0)),
              'ls': h, 'lmu': h, 'formulation': formulation} 
    
    
    # Propose an manufactured potential field and its associated gradient and also the exact flux
    def uexact(x):
        return fd.cos(fd.pi*x[0])*fd.cos(fd.pi*x[1]) 
    def eexact(x):
        return fd.grad(uexact(x))
    
    def lmbdexact(x): 
        A = fd.Constant(0.05)    # Increase A slightly for more variation
        ome = fd.Constant(2.0)   # Frequency
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

    # This will only be called for visualization purposes
    params_anal = {'exact-potential': uexact, 'exact-flux': sexact, 'exact-grad': eexact, 'exact-lmbd' : lmbdexact, 
                   'exact-mu': muexact, 'exact-etilde': etilde, 'exact-stilde': stilde}
    ref_prob = PoissonStdProblem(domain, params_anal, 'Irrelevant', forcings, None, PoissonAnalytical, orders=[3,3])

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

    def myObjFunc(g,gdata,s,sdata):
        kappa = params['kappa']
        return 0.5*fd.inner(g-gdata, g-gdata)*fd.dx + 0.5*kappa*fd.inner(s-sdata, s-sdata)*fd.dx

    if(formulation == 'Natural' and order_poly == 'Linear'):
        ords = [0,1]
    elif(formulation == 'Natural' and order_poly == 'Quadratic'):
        ords = [1,2]
    elif(formulation == 'Natural' and order_poly == 'Cubic'):
        ords = [2,3]
    elif(formulation == 'Equal-Order' and order_poly == 'Linear'):
        ords = [1,1]
    elif(formulation == 'Equal-Order' and order_poly == 'Quadratic'):
        ords = [2,2]
    elif(formulation == 'Equal-Order' and order_poly == 'Cubic'):
        ords = [3,3]
    else:
        raise ValueError('Invalid formulation and polynomial order')
    ddcm_prob = DDCMProblem(ddcm_data, myObjFunc, None, 'Mixed-Dirichlet-Neumann', forcings, DDCMStab, orders=ords, params=params)

    ddcm_prob.solve(0.0,"myddcmsol")

    # Compute some error norms
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

    output = (hsize, L2error_u, H1error_u,
              L2error_lmd, H1error_lmd,
              L2error_s, L2error_e, L2error_mu,
              Hdiverror_s, Hdiverror_e, Hdiverror_mu)

    return output
    
#-----------------------------------------------------------------------------------------
# Numerical experiments: test case 02 - Non-convex domain
#-----------------------------------------------------------------------------------------
def main_EX2_(formulation='Natural', hsize = 0.1, case_corner = 4, **kwargs):
    
    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')
    
    print('----------------------------------------------------------')
    print('|::| Reference Domain and solution begins ...')
    print('---------------------------------------------')
    
    #  Create (or import) the computational domain/mesh and Compute a reference solution    
    lc = hsize  #  characteristic length of the mesh 
    flag_singularity_refinement = True
    
    mymeshfile, tags = CreateMeshReentrantCorner('reentrant', case_corner, lc, flag_singularity_refinement)
    if(case_corner == 0):
        omega = 1.75*fd.pi
    elif(case_corner == 1):
        omega = 1.5*fd.pi
    elif(case_corner == 2):
        omega = 1.25*fd.pi
    elif(case_corner == 3):
        omega = 0.5*fd.pi
        
    nsd, Ls, Ns = 2, [1.0, 1.0], [100, 100]
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=mymeshfile, flag_quads=False)
    
    # These would be the ones corresponding to the analytical solution we propose
    h = fd.CellDiameter(domain.msh)
    params = {'perm': fd.Constant(1.0), 'zeda': fd.Constant(1.0), 
              'theta': fd.Constant(kwargs.get('theta', 0.0)), 'kappa': fd.Constant(1.0), 
              'alpha': fd.Constant(kwargs.get('alpha', 0.0)), 'beta': fd.Constant(kwargs.get('beta', 0.0)), 
              'gamma': fd.Constant(kwargs.get('gamma', 0.0)), 'eta': fd.Constant(kwargs.get('eta', 0.0)),
              'ls': h, 'lmu': h, 'formulation': formulation} 

    # We define the analytical expression for the potential field and its gradient
    def uexact(x):
        angle = fd.atan2(x[1] + 1e-12, x[0] + 1e-12)
        theta = fd.conditional(fd.gt(angle, 0), angle, angle + 2 * fd.pi)
        r = fd.sqrt(x[0]**2 + x[1]**2) + 1e-12
        alpha = fd.pi/omega
        return (r**alpha)*fd.sin(alpha*theta)
    def eexact(x):
        return fd.grad(uexact(x))
    # Propose an analytical expression for lambda
    def lmbdexact(x): 
        A = fd.Constant(0.05)   # Increase A slightly for more variation
        ome = fd.Constant(2.0)  # Frequency
        phi = fd.Constant(0.0)   # Phase angle 
        return  - (A / (ome * fd.pi)) * ( fd.cos(ome * fd.pi * x[0] + phi) + fd.cos(ome * fd.pi * x[1] + phi) )
    
    # Propose a flux-gradient relationship based on the dataset  
    def etilde(x):
        A = fd.Constant(0.05)        # Amplitude
        ome = fd.Constant(4.0)       # Frequency
        phi = fd.Constant(0.0)       # Phase angle 
        return fd.as_vector([eexact(x)[0] + A*fd.sin(ome*fd.pi*x[0] + phi),
                             eexact(x)[1] + A*fd.sin(ome*fd.pi*x[1] + phi)])
    def sexact(x):
        A = fd.Constant(0.0)  # 0.025 Amplitude
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

    # This will only be called for visualization purposes
    params_anal = {'exact-potential': uexact, 'exact-flux': sexact, 'exact-grad': eexact, 'exact-lmbd' : lmbdexact, 
                   'exact-mu': muexact, 'exact-etilde': etilde, 'exact-stilde': stilde}
    ref_prob = PoissonStdProblem(domain, params_anal, 'Irrelevant', forcings, None, PoissonAnalytical, orders=[3,3])

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

    def myObjFunc(g,gdata,s,sdata):
        kappa = params['kappa']
        return 0.5*fd.inner(g-gdata, g-gdata)*fd.dx + 0.5*kappa*fd.inner(s-sdata, s-sdata)*fd.dx

    if(formulation == 'Natural'):
        ords = [0,1]
    elif(formulation == 'Equal-Order'):
        ords = [1,1]
    ddcm_prob = DDCMProblem(ddcm_data, myObjFunc, None, 'Pure-Dirichlet', forcings, DDCMStab, orders=ords, params=params)

    ddcm_prob.solve(0.0, "myddcmsol")


    # Compute some error norms
    
    print('... Computing some error norms ...')
    x = fd.SpatialCoordinate(ddcm_data.domain.msh)
    L2error_u = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u , norm_type = 'L2')
    H1error_u = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u , norm_type = 'H1')
    H1semierror_u = error_seminorm(uexact(x), ddcm_prob.ddcm_solution.u , norm_type = 'H1')

    L2error_lmd = fd.errornorm(lmbdexact(x), ddcm_prob.ddcm_solution.lmd, norm_type = 'L2')
    H1error_lmd = fd.errornorm(lmbdexact(x), ddcm_prob.ddcm_solution.lmd, norm_type = 'H1')
    H1semierror_lmd = error_seminorm(lmbdexact(x), ddcm_prob.ddcm_solution.lmd, norm_type='H1')

    L2error_s = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s , norm_type = 'L2')
    L2error_e = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g , norm_type = 'L2')
    L2error_mu = fd.errornorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type = 'L2')
    if formulation == 'Equal-Order':
        Hdiverror_s = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s , norm_type = 'Hdiv')
        Hdivsemierror_s = error_seminorm(sexact(x), ddcm_prob.ddcm_solution.s, norm_type='Hdiv')

        Hdiverror_e = fd.errornorm(eexact(x), ddcm_prob.ddcm_solution.g , norm_type = 'Hdiv')
        Hdivsemierror_e = error_seminorm(eexact(x), ddcm_prob.ddcm_solution.g, norm_type='Hdiv')

        Hdiverror_mu = fd.errornorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type = 'Hdiv')
        Hdivsemierror_mu = error_seminorm(muexact(x), ddcm_prob.ddcm_solution.mu, norm_type='Hdiv')
    else:
        Hdiverror_s, Hdivsemierror_s = np.nan, np.nan
        Hdiverror_e, Hdivsemierror_e = np.nan, np.nan
        Hdiverror_mu, Hdivsemierror_mu = np.nan, np.nan

    print('----------------------------------------------------------')
    print('|> Data Driven solution OK')
    print('----------------------------------------------------------')
    
    elapsed_time = time.time() - ti
    print('----------------------------------------------------------')
    print('|> End DDCM Computation | Total time= %1.1f s' %(elapsed_time))
    print('----------------------------------------------------------')

    output = (hsize, L2error_u, H1error_u, H1semierror_u,
              L2error_lmd, H1error_lmd, H1semierror_lmd,
              L2error_s, L2error_e, L2error_mu,
              Hdiverror_s, Hdivsemierror_s,
              Hdiverror_e, Hdivsemierror_e,
              Hdiverror_mu, Hdivsemierror_mu)

    return output

#-----------------------------------------------------------------------------------------
# Numerical experiments: test case 03 - Data assignation
#-----------------------------------------------------------------------------------------
def main_EX3_(formulation='Natural', N=20, Nsamples=3, order_poly = 'Linear', **kwargs):

    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')
    
    print('----------------------------------------------------------')
    print('|::| Reference Domain and solution begins ...')
    print('---------------------------------------------')

    # Create (or import) the computational domain/mesh and Compute a reference solution

    nsd, Ls, Ns = 2, [1.0, 1.0], [N, N]
    hsize = max(Ls[0]/Ns[0], Ls[1]/Ns[1])
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)
    
    # These would be the ones corresponding to the analytical solution we propose

    h = fd.CellDiameter(domain.msh)

    params = {'perm': fd.Constant(1.0), 'zeda': fd.Constant(1.0), 
              'theta': fd.Constant(kwargs.get('theta', 0.0)), 	'kappa': fd.Constant(1.0), 
              'alpha': fd.Constant(kwargs.get('alpha', 0.0)), 	'beta': fd.Constant(kwargs.get('beta', 0.0)), 
              'gamma': fd.Constant(kwargs.get('gamma', 0.0)), 	'eta': fd.Constant(kwargs.get('eta', 0.0)),
              'ls': h, 'lmu': h, 'formulation': formulation} 
    
    
    # Propose an manufactured potential field and its associated gradient and also the exact flux

    def uexact(x):
        return fd.sin(fd.pi*x[0])*fd.sin(fd.pi*x[1])

    def eexactnp(x):
        return [np.pi*np.cos(np.pi*x[:,0])*np.sin(np.pi*x[:,1]), np.pi*np.sin(np.pi*x[:,0])*np.cos(np.pi*x[:,1])]

    def eexact(x):
        return fd.grad(uexact(x))

    def lmbdexact(x): 
        return fd.Constant(0.0)
    
    def sexact(x):
        return -fd.grad(uexact(x))
    
    def sexactnp(x):
        return [-np.pi*np.cos(np.pi*x[:,0])*np.sin(np.pi*x[:,1]), -np.pi*np.sin(np.pi*x[:,0])*np.cos(np.pi*x[:,1])]
    
    def muexact(x):
        return fd.as_vector([fd.Constant(0.0), fd.Constant(0.0)])
    def fdual(x):
        return fd.Constant(0.0)
    def qexact(x):
        zeda = params['zeda']
        return zeda*uexact(x) + fd.div(sexact(x))
            
    forcings = PoissonForcingData(qexact, sexact, uexact, lmbdexact, muexact, fdual)
    
    # This will only be called for visualization purposes
    params_anal = {'exact-potential': uexact, 'exact-flux': sexact, 'exact-grad': eexact, 'exact-lmbd' : lmbdexact, 
                   'exact-mu': muexact, 'exact-etilde': None, 'exact-stilde': None}
    ref_prob = PoissonStdProblem(domain, params_anal, 'Irrelevant', forcings, None, PoissonAnalytical, orders=[3,3])

    ref_prob.solve(tt=N, dir_outs="myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: OK')
    print('----------------------------------------------------------')

    # Read or sample some synthetic data from a standard solution: 
    # The idea is to produce some plausible data similar to what an experiment would deliver

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

    # Solve the Data Driven problem: This problem should share the same 
    # forcings (bcs, sources, etc.) as the standard problem above

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g,gdata,s,sdata):
        kappa = params['kappa']
        return 0.5*fd.inner(g-gdata, g-gdata)*fd.dx + 0.5*kappa*fd.inner(s-sdata, s-sdata)*fd.dx

    if(formulation == 'Natural' and order_poly == 'Linear'):
        ords = [0,1]
    elif(formulation == 'Natural' and order_poly == 'Quadratic'):
        ords = [1,2]
    elif(formulation == 'Natural' and order_poly == 'Cubic'):
        ords = [2,3]
    elif(formulation == 'Equal-Order' and order_poly == 'Linear'):
        ords = [1,1]
    elif(formulation == 'Equal-Order' and order_poly == 'Quadratic'):
        ords = [2,2]
    elif(formulation == 'Equal-Order' and order_poly == 'Cubic'):
        ords = [3,3]
    else:
        raise ValueError('Invalid formulation and polynomial order')
    
    ddcm_prob = DDCMProblem(ddcm_data, myObjFunc, None, 'Pure-Dirichlet', forcings, DDCMStab, orders=ords, params=params)
    ddcm_prob.solve(tt=Nsamples,dir_outs="myddcmsol")

    # Compute some error norms
    
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
  
# ------------------------------
# Helper functions
# ------------------------------
def get_norm_list(formulation):
    if formulation == 'Natural':
        return ['L2u', 'H1u', 'L2l', 'H1l', 'EsL2', 'EeL2', 'EmL2']
    elif formulation == 'Equal-Order':
        return ['L2u', 'H1u', 'L2l', 'H1l', 'EsL2', 'EeL2', 'EmL2', 'EsHdiv', 'EeHdiv', 'EmHdiv']
    else:
        raise ValueError("Invalid formulation")

def get_stab_params(formulation):
    if formulation == 'Natural':
        return {None: None}
    elif formulation == 'Equal-Order':
        return {
            # 'Unstabilized': None,
            'Fully stabilized': {'theta': 0.5, 'alpha': 0.125, 'beta': 0.5, 'gamma': 0.125, 'eta': 0.5},
            # 'Minimally stabilized': {'alpha': 0.25, 'eta': 0.25}
        }
    else:
        raise ValueError("Invalid formulation")

def get_filename_base(formulation, suffix, param=None):
    if formulation == 'Natural':
        return f'table_{formulation}_{suffix}'
    elif formulation == 'Equal-Order':
        if param == 'Unstabilized':
            return f'table_{formulation}_{param}_{suffix}'
        elif param in ['Fully stabilized', 'Minimally stabilized']:
            words = param.split()
            return f'table_{formulation}_{words[0]}_{words[1]}_{suffix}'
        else:
            raise ValueError('Invalid parameter for Equal-Order formulation!')
    else:
        raise ValueError('Invalid formulation')  
      
# ------------------------------
# Test Case 1: Convex Domain
# ------------------------------
def run_test_case_1(formulation, order_poly, foldername):
    stab_params = get_stab_params(formulation)
    norm_list = get_norm_list(formulation)

    for param, val in stab_params.items():
        kwargs = {} if val is None else val.copy()
        print(f'==== Solving for {formulation}: {param} ===')

        hhs = []
        errors = {key: [] for key in norm_list}
        for N in [5]:
            output = main_EX1_(formulation, N, order_poly, **kwargs)
            hhs.append(output[0])
            for i, key in enumerate(errors):
                errors[key].append(output[i + 1])

        # filename_base = get_filename_base(formulation, 'convex', param)
        # os.makedirs(foldername, exist_ok=True)
        # results = DDCM_save_results(hhs, errors, formulation)
        # results.save_results(order_poly, filename_base, foldername)

# ------------------------------
# Test Case 2: Non-Convex Domain
# ------------------------------
def run_test_case_2(formulation, order_poly, foldername, case_corner):
    corners_labels = ['7pi_4', '3pi_2', '5pi_4', 'pi_2']
    label = corners_labels[case_corner]
    stab_params = get_stab_params(formulation)
    # norm_list = get_norm_list(formulation)
    if formulation == 'Natural':
        norm_list =  ['L2u', 'H1u', 'H1semi_u',  'L2l', 'H1l', 'H1semi_l','EsL2', 'EeL2', 'EmL2']
    elif formulation == 'Equal-Order':
        norm_list =  ['L2u', 'H1u', 'H1semi_u', 'L2l', 'H1l', 'H1semi_l' , 'EsL2', 'EeL2', 'EmL2',
                      'Es_Hdiv', 'Es_Hdivsemi', 'EeHdiv', 'Ee_Hdivsemi', 'Em_Hdiv', 'Em_Hdivsemi']
    else:
        raise ValueError("Invalid formulation")


    for param, val in stab_params.items():
        kwargs = {} if val is None else val.copy()
        print(f'==== Solving for {formulation}: {param} ===')

        hhs = [0.1, 0.05, 0.025, 0.0125]
        # hhs = [0.5, 0.25]
        errors = {key: [] for key in norm_list}
        for h in hhs:
            output = main_EX2_(formulation, h, case_corner, **kwargs)
            for i, key in enumerate(errors):
                errors[key].append(output[i + 1])

        filename_base = get_filename_base(formulation, label, param)
        os.makedirs(foldername, exist_ok=True)
        results = DDCM_save_results(hhs, errors, formulation)
        results.save_results(order_poly, filename_base, foldername)

# ------------------------------
# Test Case 3: Data Assignation
# ------------------------------
def run_test_case_3(formulation, order_poly, foldername, Nsamples_list):
    stab_params = get_stab_params(formulation)
    norm_list = get_norm_list(formulation)
    Nhs = [10]

    for param, val in stab_params.items():
        kwargs = {} if val is None else val.copy()
        print(f'==== Solving for {formulation}: {param} ===')

        for Ns in Nsamples_list:
            print(f'||> Running mesh convergence for Nsamples = {Ns}')
            hhs = []
            errors = {norm: [] for norm in norm_list}
            for Nh in Nhs:
                output = main_EX3_(formulation, Nh, Ns, order_poly, **kwargs)
                hhs.append(output[0])
                for i, key in enumerate(errors):
                    errors[key].append(output[i + 1])

            filename_base = get_filename_base(formulation, f'Ns{Ns}', param)
            os.makedirs(foldername, exist_ok=True)
            results = DDCM_save_results(hhs, errors, formulation)
            results.save_results(order_poly, filename_base, foldername)

# ------------------------------
# Main Dispatcher
# ------------------------------
if __name__ == "__main__":
    main_selector = int(sys.argv[1])

    formulation = 'Equal-Order'  # or 'Equal-Order'
    order_poly = 'Linear'

    if main_selector == 1:
        run_test_case_1(formulation, order_poly, '01_test_case_convex_domain')
    elif main_selector == 2:
        for val in [0, 1, 2]:
            run_test_case_2(formulation, order_poly, '02_test_case_non_convex_domain', case_corner=val)
    elif main_selector == 3:
        run_test_case_3(formulation, order_poly, '03_test_case_data_assignation', Nsamples_list=[16])
        # 128, 256, 512, 1024, 2048
    else:
        raise ValueError("Invalid main_selector. Choose 1, 2, or 3.")

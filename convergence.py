import numpy as np
import time
from math import pi, cos
import matplotlib.pyplot as plt
from ddcm_classes import *


def convergence_study():
    N_data = np.array([10, 20, 40, 80, 160, 320, 640, 1280])    
    N_fem = np.array([10, 20, 40, 80, 160])
    
    num_data = len(N_data)
    num_fem = len(N_fem)
    
    hsize_matrix = np.zeros((num_fem))
    L2error_u_matrix = np.zeros((num_data, num_fem))
    H1error_u_matrix = np.zeros((num_data, num_fem))
    L2error_s_matrix = np.zeros((num_data, num_fem))
    
    start_time = time.time()
    
    for i, N_d in enumerate(N_data):
        synthetic_data_points = N_d * N_d
        print(f"\nRunning with N = {synthetic_data_points} synthetic data points\n")
        
        for j, N_f in enumerate(N_fem):
            print(f"\nRunning with Ns = {N_f} elements per direction\n")
            
            npoints_data, hsize, L2error_u, H1error_u, L2error_s = main_EX4_(Ns=N_f, Nbc=N_d, Nth=N_d)
            
            hsize_matrix[j] = hsize
            L2error_u_matrix[i, j] = L2error_u
            H1error_u_matrix[i, j] = H1error_u
            L2error_s_matrix[i, j] = L2error_s
    
    np.savetxt('n_data.txt', N_data)
    np.savetxt('mesh.txt', hsize_matrix)
    np.savetxt('L2error_u.txt', L2error_u_matrix)
    np.savetxt('H1error_u.txt', H1error_u_matrix)
    np.savetxt('L2error_s.txt', L2error_s_matrix)
    
    # Print the results in a formatted way
    results = "\nConvergence Study Results:\n"
    results += "N_data    N_fem     hsize      L2error_u    H1error_u    L2error_s\n"
    for i, N_d in enumerate(N_data):
        for j, N_f in enumerate(N_fem):
            results += f"{N_d:<8} {N_f:<8} {hsize_matrix[j]:<10.5f} {L2error_u_matrix[i, j]:<12.5e} {H1error_u_matrix[i, j]:<12.5e} {L2error_s_matrix[i, j]:<12.5e}\n"
    
    print(results)
    print(f"Total computation time: {time.time() - start_time:.2f} seconds")

    return results

def main_EX4_(Ns,Nbc, Nth):
    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')
    
    print('----------------------------------------------------------')
    print('|::| Reference Domain and solution begins ...')
    print('---------------------------------------------')

    nsd, Ls = 2, [1.0, 1.0]
    domain = ComputationalDomain(nsd, Ls, [Ns, Ns], meshfile=None, flag_quads=False)

    def uexact(x):
        return cos(pi * x[0]) * cos(pi * x[1])

    def sexact(x):
        kappa = fd.Constant(1.0)
        return -kappa * fd.grad(uexact(x))

    def mysource(x):
        return fd.div(-kappa * fd.grad(uexact(x)))

    def mysurfaceflux(x):
        return sexact(x)

    def mybcvals(x):
        return uexact(x)

    forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals)

    params = {'exact-potential': uexact, 'exact-flux': sexact}
    ref_prob = PoissonStdProblem(domain, params, 'Pure-Dirichlet', forcings, None, PoissonAnalytical, orders=[3, 3])

    ref_prob.solve(0.0, "myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: OK')
    print('----------------------------------------------------------')

    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution begins ...')
    print('-------------------------------------------------------')

    nsd, Ls, Ns = 2, [1.0, 1.0], [Ns, Ns]
    hsize = max(Ls[0] / Ns[0], Ls[1] / Ns[1])
    ddcm_domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)
    ddcm_data = PoissonDDCMData(ddcm_domain)

    ti = time.time()
    bcvals = np.linspace(1.0 + 1.e-15, 12.0, Nbc)
    gradu_samples = np.empty(shape=(Nbc * Nth, 2), dtype=np.float64)
    flux_samples = np.empty(shape=(Nbc * Nth, 2), dtype=np.float64)
    count = 0
    for bcvs in bcvals:
        g1 = 0.0
        s1 = 0.0
        L = 1.0
        g2 = (bcvs - 1.0) / L
        kappa = 1.0
        s2 = -kappa * g2
        gradu_sampled_vals = np.array([g1, g2])
        flux_sampled_vals = np.array([s1, s2])

        for angle in np.linspace(0.0, 2. * np.pi, Nth, endpoint=True):
            ct, st = np.cos(angle), np.sin(angle)
            gradu_samples[count, 0] = ct * gradu_sampled_vals[0] - st * gradu_sampled_vals[1]
            gradu_samples[count, 1] = st * gradu_sampled_vals[0] + ct * gradu_sampled_vals[1]
            flux_samples[count, 0] = ct * flux_sampled_vals[0] - st * flux_sampled_vals[1]
            flux_samples[count, 1] = st * flux_sampled_vals[0] + ct * flux_sampled_vals[1]
            count += 1

    print(' . Bcs x Rotate= ', time.time() - ti)

    ddcm_data.SetSampledData(gradu_samples, flux_samples, add_origin=True, data_file=None)

    ddcm_data.Distribute(ref_prob.std_solution, 'Interpolate-VDG0', flag_plot=True)

    npoints_data = ddcm_data.sampled_grad.shape[0]
    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution: %d data points' % (npoints_data))
    print('----------------------------------------------------------')

    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g, gdata, s, sdata):
        return fd.inner(g - gdata, g - gdata) * fd.dx + fd.inner(s - sdata, s - sdata) * fd.dx

    ddcm_prob = DDCMProblem(ddcm_data, myObjFunc, None, 'Pure-Dirichlet', forcings, NLSP5FA, orders=[0, 1])

    ddcm_prob.solve("myddcmsol")

    print('   ... Computing some error norms ...')
    x = fd.SpatialCoordinate(ddcm_data.domain.msh)
    L2error_u = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u, norm_type='L2')
    H1error_u = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u, norm_type='H1')
    L2error_s = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s, norm_type='L2')

    print('----------------------------------------------------------')
    print('|> Data Driven solution OK')
    print('----------------------------------------------------------')

    elapsed_time = time.time() - ti
    print('----------------------------------------------------------')
    print('|> End DDCM Computation | Total time= %1.1f s' % (elapsed_time))
    print('----------------------------------------------------------')

    return npoints_data, hsize, L2error_u, H1error_u, L2error_s



# Run the convergence study

convergence_study()



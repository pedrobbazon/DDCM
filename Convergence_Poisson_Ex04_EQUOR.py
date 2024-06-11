import numpy as np
import time
from math import pi, cos
import matplotlib.pyplot as plt
from ddcm_classes import *


def main_EX4_(Ns, Nbc, Nth):
    """
    Main function for the DDCM computation.
    """

    # Beginning Data Driven Computational Mechanics (DDCM) Computation
    ti = time.time()
    print('|> Beginning Data Driven Computational Mechanics (DDCM) Computation')

    # Reference Domain and solution begins ...
    print('----------------------------------------------------------')
    print('|::| Reference Domain and solution begins ...')
    print('---------------------------------------------')

    # 1. Create (or import) the computational domain/mesh and Compute a reference solution

    nsd, Ls = 2, [1.0, 1.0]
    domain = ComputationalDomain(nsd, Ls, [Ns, Ns], meshfile=None, flag_quads=False)

    # Define analytical solution functions
    def uexact(x):
        return fd.cos(fd.pi * x[0]) * fd.cos(fd.pi * x[1])

    def sexact(x):
        kappa = fd.Constant(1.0)
        return -kappa * fd.grad(uexact(x))

    def mysource(x):
        kappa = fd.Constant(1.0)
        return fd.div(-kappa * fd.grad(uexact(x)))

    def mysurfaceflux(x):
        return sexact(x)

    def mybcvals(x):
        return uexact(x)

    forcings = PoissonForcingData(mysource, mysurfaceflux, mybcvals)

    # This will be only called for visualization purposes
    params = {'exact-potential': uexact, 'exact-flux': sexact}
    ref_prob = PoissonStdProblem(domain, params, 'Pure-Dirichlet', forcings, None, PoissonAnalytical, orders=[3, 3])

    ref_prob.solve(0.0, "myrefsol")

    print('----------------------------------------------------------')
    print('|::| Reference domain and solution: OK')
    print('----------------------------------------------------------')

    # Synthetic data computation/distribution begins ...
    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution begins ...')
    print('-------------------------------------------------------')

    # Note in this case, I've made the data associated with the same domain (and mesh)
    # we used for the reference solution, but this doesn't have to be the case, it can be a finer mesh
    hsize = max(Ls[0] / Ns, Ls[1] / Ns)
    ddcm_domain = ComputationalDomain(nsd, Ls, [Ns, Ns], meshfile=None, flag_quads=False)
    ddcm_data = PoissonDDCMData(ddcm_domain)

    # In this manufactured example, we create by hand fluxes and grads
    # without actually solving a synthetic problem:
    bcvals = np.linspace(1.0 + 1.e-15, 12.0, Nbc)
    gradu_samples = np.empty((Nbc * Nth, 2), dtype=np.float64)
    flux_samples = np.empty((Nbc * Nth, 2), dtype=np.float64)
    count = 0
    for bcvs in bcvals:
        # Generate a flux s and a grad g
        g1, s1 = 0.0, 0.0
        g2 = (bcvs - 1.0)
        s2 = -g2
        gradu_sampled_vals = np.array([g1, g2])
        flux_sampled_vals = np.array([s1, s2])

        # Apply any manipulation/transformation you need to define
        # the synthetic data (rotations, scalings, removing outliers, etc)
        for angle in np.linspace(0.0, 2. * np.pi, Nth, endpoint=True):
            ct, st = np.cos(angle), np.sin(angle)
            gradu_samples[count] = [ct * gradu_sampled_vals[0] - st * gradu_sampled_vals[1],
                                    st * gradu_sampled_vals[0] + ct * gradu_sampled_vals[1]]
            flux_samples[count] = [ct * flux_sampled_vals[0] - st * flux_sampled_vals[1],
                                   st * flux_sampled_vals[0] + ct * flux_sampled_vals[1]]
            count += 1

    ddcm_data.SetSampledData(gradu_samples, flux_samples, add_origin=True, data_file=None)
    ddcm_data.Distribute(ref_prob.std_solution, 'Interpolate-VDG0', flag_plot=True)

    npoints_data = ddcm_data.sampled_grad.shape[0]
    print('----------------------------------------------------------')
    print('|::| Synthetic data computation/distribution: %d data points' % npoints_data)
    print('----------------------------------------------------------')

    # Data Driven solution begins ...
    print('----------------------------------------------------------')
    print('|> Data Driven solution begins ...')
    print('----------------------------------')

    def myObjFunc(g, gdata, s, sdata):
        return fd.inner(g - gdata, g - gdata) * fd.dx + fd.inner(s - sdata, s - sdata) * fd.dx

    ddcm_prob = DDCMProblem(ddcm_data, myObjFunc, None, 'Pure-Dirichlet', forcings, NLSP5FA_EQUOR, orders=[1, 1])

    ddcm_prob.solve("myddcmsol")

    # Compute some error norms
    print('... Computing some error norms ...')
    x = fd.SpatialCoordinate(ddcm_data.domain.msh)
    L2error_u = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u, norm_type='L2')
    H1error_u = fd.errornorm(uexact(x), ddcm_prob.ddcm_solution.u, norm_type='H1')
    L2error_s = fd.errornorm(sexact(x), ddcm_prob.ddcm_solution.s, norm_type='L2')
    L2error_lmd = fd.errornorm(fd.Constant(0), ddcm_prob.ddcm_solution.lmd, norm_type='L2')
    L2error_mu = fd.errornorm(fd.as_vector([fd.Constant(0),fd.Constant(0)]), ddcm_prob.ddcm_solution.mu, norm_type='L2')

    print('----------------------------------------------------------')
    print('|> Data Driven solution OK')
    print('----------------------------------------------------------')

    elapsed_time = time.time() - ti
    print('----------------------------------------------------------')
    print('|> End DDCM Computation | Total time= %1.1f s' % elapsed_time)
    print('----------------------------------------------------------')

    return npoints_data, hsize, L2error_u, H1error_u, L2error_s, L2error_lmd,  L2error_mu

def convergence_study():
    N_data = np.array([10, 20, 40, 80, 160])
    N_fem = np.array([10, 20, 40, 80, 160])

    num_data = len(N_data)
    num_fem = len(N_fem)

    hsize_matrix = np.zeros(num_fem)
    L2error_u_matrix = np.zeros((num_data, num_fem))
    H1error_u_matrix = np.zeros((num_data, num_fem))
    L2error_s_matrix = np.zeros((num_data, num_fem))
    L2error_lmd_matrix = np.zeros((num_data, num_fem))
    L2error_mu_matrix = np.zeros((num_data, num_fem))

    start_time = time.time()

    for i, N_d in enumerate(N_data):
        synthetic_data_points = N_d * N_d
        print(f"\nRunning with N = {synthetic_data_points} synthetic data points\n")
        for j, N_f in enumerate(N_fem):
            print(f"\nRunning with Ns = {N_f} elements per direction\n")
            npoints_data, hsize, L2error_u, H1error_u, L2error_s, L2error_lmd, L2error_mu = main_EX4_(Ns=N_f, Nbc=N_d, Nth=N_d)
            hsize_matrix[j] = hsize
            L2error_u_matrix[i, j] = L2error_u
            H1error_u_matrix[i, j] = H1error_u
            L2error_s_matrix[i, j] = L2error_s
            L2error_lmd_matrix[i, j] = L2error_lmd
            L2error_mu_matrix[i, j] = L2error_mu

    # Print the results in a formatted way
    results = "\nConvergence Study Results:\n"
    results += "N_data    N_fem     hsize    L2error_u    H1error_u    L2error_s     L2error_lmd\n"
    for i, N_d in enumerate(N_data):
        for j, N_f in enumerate(N_fem):
            results += (
                f"{N_d:<8} {N_f:<8} {hsize_matrix[j]:<10.5f} "
                f"{L2error_u_matrix[i, j]:<12.5e} {H1error_u_matrix[i, j]:<12.5e} "
                f"{L2error_s_matrix[i, j]:<12.5e} {L2error_lmd_matrix[i, j]:<12.5e}\n"
            )
    print(results)
    print(f"Total computation time: {time.time() - start_time:.2f} seconds")

    return N_data, N_fem, hsize_matrix, L2error_u_matrix, H1error_u_matrix, L2error_s_matrix, L2error_lmd_matrix, L2error_mu_matrix


# Run the convergence study
N_data, N_fem, hsize_matrix, L2error_u_matrix, H1error_u_matrix, L2error_s_matrix, L2error_lmd_matrix, L2error_mu_matrix = convergence_study()



# Create a 3x2 grid of subplots
fig, ((ax1, ax2), (ax3, ax4), (ax5, _)) = plt.subplots(3, 2, figsize=(12, 12))

# Define font sizes
title_fontsize = 10
label_fontsize = 8
legend_fontsize = 8
tick_fontsize = 8

# Plotting L2 error for u
ax1.set_title('Error in L2 norm for u', fontsize=title_fontsize)
for i in range(len(N_data)):
    ax1.loglog(hsize_matrix, L2error_u_matrix[i], '*-', label=f'N = {N_data[i]**2}')
ax1.set_xlabel(r'$\Delta_x = \Delta_y = h$', fontsize=label_fontsize)
ax1.set_ylabel(r'$\|u-u_{h}\|_{L2}$', fontsize=label_fontsize)
ax1.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
ax1.minorticks_on()
ax1.legend(loc='lower right', fontsize=legend_fontsize)
ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)
ax1.tick_params(axis='both', which='minor', labelsize=tick_fontsize)

# Plotting H1 error for u
ax2.set_title('Error in H1 norm for u', fontsize=title_fontsize)
for i in range(len(N_data)):
    ax2.loglog(hsize_matrix, H1error_u_matrix[i], 's-', label=f'N = {N_data[i]**2}')
ax2.set_xlabel(r'$\Delta_x = \Delta_y = h$', fontsize=label_fontsize)
ax2.set_ylabel(r'$\|u-u_{h}\|_{H1}$', fontsize=label_fontsize)
ax2.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
ax2.minorticks_on()
ax2.legend(loc='lower right', fontsize=legend_fontsize)
ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)
ax2.tick_params(axis='both', which='minor', labelsize=tick_fontsize)

# Plotting L2 error for s
ax3.set_title('Error in L2 norm for s', fontsize=title_fontsize)
for i in range(len(N_data)):
    ax3.loglog(hsize_matrix, L2error_s_matrix[i], 'o-', label=f'N = {N_data[i]**2}')
ax3.set_xlabel(r'$\Delta_x = \Delta_y = h$', fontsize=label_fontsize)
ax3.set_ylabel(r'$\|s-s_{h}\|_{L2}$', fontsize=label_fontsize)
ax3.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
ax3.minorticks_on()
ax3.legend(loc='lower right', fontsize=legend_fontsize)
ax3.tick_params(axis='both', which='major', labelsize=tick_fontsize)
ax3.tick_params(axis='both', which='minor', labelsize=tick_fontsize)

# Plotting L2 error for lambda
ax4.set_title('Error in L2 norm for lambda', fontsize=title_fontsize)
for i in range(len(N_data)):
    ax4.loglog(hsize_matrix, L2error_lmd_matrix[i], 'x-', label=f'N = {N_data[i]**2}')
ax4.set_xlabel(r'$\Delta_x = \Delta_y = h$', fontsize=label_fontsize)
ax4.set_ylabel(r'$\|\lambda-\lambda_{h}\|_{L2}$', fontsize=label_fontsize)
ax4.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
ax4.minorticks_on()
ax4.legend(loc='lower right', fontsize=legend_fontsize)
ax4.tick_params(axis='both', which='major', labelsize=tick_fontsize)
ax4.tick_params(axis='both', which='minor', labelsize=tick_fontsize)

# Plotting L2 error for mu
ax5.set_title('Error in L2 norm for mu', fontsize=title_fontsize)
for i in range(len(N_data)):
    ax5.loglog(hsize_matrix, L2error_mu_matrix[i], '*-', label=f'N = {N_data[i]**2}')
ax5.set_xlabel(r'$\Delta_x = \Delta_y = h$', fontsize=label_fontsize)
ax5.set_ylabel(r'$\|\mu-\mu_{h}\|_{L2}$', fontsize=label_fontsize)
ax5.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
ax5.minorticks_on()
ax5.legend(loc='lower right', fontsize=legend_fontsize)
ax5.tick_params(axis='both', which='major', labelsize=tick_fontsize)
ax5.tick_params(axis='both', which='minor', labelsize=tick_fontsize)

# Hide the empty subplot (ax6)
fig.delaxes(_)

plt.tight_layout()
plt.show()





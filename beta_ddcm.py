import firedrake
from firedrake import *
from petsc4py import PETSc

import time
import os
import matplotlib.pyplot as plt

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS
import scipy.sparse as sp

os.environ["OMP_NUM_THREADS"] = "1"

#-------------------------------------------------------------------------------------
def SolvePoissonCG(mesh, kappa, flag_quads, HomDirVal, orderU=1):

    print( "|> Solving with CG (primal) formulation\n")

    W = FunctionSpace(mesh, "CG", orderU)
    #print(W.cell_node_list)

    f = Constant(1.0)

    u = TrialFunction(W)
    v = TestFunction(W)

    a = inner(kappa*grad(u), grad(v))*dx
    L = f*v*dx
    
    # Pressure (natural) boundary conditions
    uboundary = Constant(HomDirVal)
    bcall = DirichletBC(W, uboundary, "on_boundary")

    w = Function(W)
    solve(a == L, w, bcs=[bcall])

    #wref = Function(W)
    #wref.dat.data[:] = w.dat.data[:]

    return w

def SolvePoissonHdiv(mesh, kappa, flag_quads, HomDirVal, orderF=1, orderU=0):

    print( "|> Solving with Hdiv (mixed) formulation\n")

    if(flag_quads):
        HD = FunctionSpace(mesh, "RTCF", orderF)
        DG = FunctionSpace(mesh, "DQ", orderU)
    else:
        HD = FunctionSpace(mesh, "RT", orderF)
        DG = FunctionSpace(mesh, "DG", orderU)

    W = HD * DG

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    x, y = SpatialCoordinate(mesh)
    
    f = Constant(1.0)
    n = FacetNormal(mesh)

    # Pressure (natural) boundary conditions
    uboundary = Constant(HomDirVal)
    
    a = ((1.0/kappa)*dot(sigma, tau) - div(tau)*u - div(sigma)*v)*dx
    L = -f*v*dx - uboundary*dot(tau,n)*ds
    
    w = Function(W)
    solve(a == L, w, bcs=[])
    sigma, u = w.sub(0), w.sub(1)

    return sigma, u

# Domain
Lx, Ly = 1.0, 1.0
Nx, Ny = 21, 21
flag_quads = True
mesh = RectangleMesh(Nx, Ny, Lx, Ly, quadrilateral=flag_quads)
DGtype = "DG"
if(flag_quads):
    DGtype = "DQ"

DG0 = FunctionSpace(mesh, DGtype, degree=0)

area_els = Function(DG0)
tstdg = TestFunction(DG0)
areas = assemble(Constant(1.0, domain=mesh)*tstdg*dx)
area_els = areas.dat.data_ro[:]
#print(area_els.dat.data[:])

# Permeability field
random_perm = False
if(random_perm): # generate random conductivity with beta distribution
    pcg = PCG64(seed=0)
    rg = RandomGenerator(pcg)
    # beta distribution
    kappa = rg.beta(DG0, 2.0, 5.0)
    kappa.dat.data[:] += 1.0
else:
    kappa = Constant(1.0)

# Homegeneous Dirichlet boundary condition
HomDirVal = 3.14

# Source term
f = Constant(1.0)

#-------------------------------------------------------------------------------------
# Data (e.g., some measured data relating heat flux and temperature gradient)

# Generate some synthetic data:

#1. Solve the forward problem by some method
forward_solve = 'Mixed-Hdiv'
#forward_solve = 'Primal-CG'

if(forward_solve == 'Primal-CG'):
    u = SolvePoissonCG(mesh, kappa, flag_quads, HomDirVal, orderU=1)
    uref = Function(u.function_space())
    uref.assign(u)
    # Recover the temperature gradient and the corresponding flux on each element
    VDG0 = VectorFunctionSpace(mesh, family=DGtype, degree=0)
    Gradu = project(grad(u), VDG0)
    gradu_vals = Gradu.dat.data 
    Flux = project(-kappa*grad(u), VDG0)
    flux_vals = Flux.dat.data
    # Save the reference solution for further visualization
    File("gradu_ref.pvd").write(project(Gradu, VDG0, name="GradT"))
    File("flux_ref.pvd").write(project(Flux, VDG0, name="Flux"))
    File("temp_ref.pvd").write(project(u, FunctionSpace(mesh, "CG", 1), name="Temperature"))
elif(forward_solve == 'Mixed-Hdiv'):
    orderF, orderU = 2, 1
    fluxhdiv, u =  SolvePoissonHdiv(mesh, kappa, flag_quads, HomDirVal, orderF, orderU)
    fluxref = Function(fluxhdiv.function_space())
    fluxref.assign(fluxhdiv)
    uref = Function(u.function_space())
    uref.assign(u)
    VDG0 = VectorFunctionSpace(mesh, family=DGtype, degree=0)
    Gradu = project(grad(u), VDG0)
    gradu_vals = Gradu.dat.data
    Flux = project(fluxhdiv, VDG0)
    flux_vals = Flux.dat.data
    # Save the reference solution for further visualization
    File("gradu_ref.pvd").write(project(Gradu, VDG0, name="GradT"))
    File("flux_refCG.pvd").write(project(fluxref, VectorFunctionSpace(mesh, "CG", orderF), name="Flux"))
    File("flux_refDG.pvd").write(project(fluxref, VectorFunctionSpace(mesh, DGtype, orderF), name="Flux"))
    File("temp_refCG.pvd").write(project(uref, FunctionSpace(mesh, "CG", 1), name="Temperature"))
    File("temp_refDG.pvd").write(project(uref, FunctionSpace(mesh, DGtype, orderU), name="Temperature"))
else:
    raise ValueError('forward_solve option not recognized\n')

    
flag_dist = 'R2'

def ComputeState(fluxes, grads, flag_dist):
  if(flag_dist == 'R4'):
    flux0 = fluxes[:,0]
    flux1 = fluxes[:,1]
    grad0 = grads[:,0]
    grad1 = grads[:,1]
    state_vector = np.dstack([grad0, grad1, flux0, flux1])[0]
  elif(flag_dist == 'R3'):
    size_data = fluxes.shape[0]
    flux0 = fluxes[:,0]
    flux1 = fluxes[:,1]
    grad0 = grads[:,0]
    grad1 = grads[:,1]
    flux0rot = np.zeros_like(flux0)
    flux1rot = np.zeros_like(flux1)
    maggrad = np.linalg.norm(grads, axis=1)
    maggradfix = np.where(maggrad != 0, maggrad, 1.0 + maggrad) # This can just happend when both grad0 and grad1 are zero
    cost = grad0 / maggradfix
    sint = grad1 / maggradfix
    grad0 = maggrad
    flux0rot = cost*flux0 + sint*flux1
    flux1rot = -sint*flux0 + cost*flux1
    state_vector = np.dstack([grad0, flux0rot, flux1rot])[0]
  elif(flag_dist == 'R2'):
    magflux = np.linalg.norm(fluxes, axis=1)
    maggrad = np.linalg.norm(grads, axis=1)
    state_vector = np.dstack([maggrad, magflux])[0]
  else:
    raise ValueError('Option flag_dist not recognized')
  return state_vector

current_state = ComputeState(flux_vals, gradu_vals, flag_dist)

#3. Sample the solution at the point cloud

#sampling_type = 'uniform_interval_sample'
#sampling_type = 'totally_random_sample'
sampling_type = 'uniform_domain_sample'

if(sampling_type == 'totally_random_sample'):
    num_points = 10
    np.random.seed(0)
    xs = np.random.random_sample((num_points, 2))
elif(sampling_type == 'uniform_domain_sample'):
    one_sample_per_element = False
    if(one_sample_per_element == True):
        # One point per element centroid
        X = interpolate(mesh.coordinates, VDG0)
        xs = X.dat.data
    else:
        Nsamples_x = 3
        Nsamples_y = 3
        Deltax = Lx / Nsamples_x
        Deltay = Ly / Nsamples_y
        xls = np.linspace(0.5*Deltax, Lx-0.5*Deltax, Nsamples_x)
        yls = np.linspace(0.5*Deltay, Ly-0.5*Deltay, Nsamples_y)
        xlsg, ylsg = np.meshgrid(xls, yls)
        xs = np.dstack([xlsg, ylsg]).reshape(Nsamples_x*Nsamples_y,2)
        #print(xs)
        
point_cloud = VertexOnlyMesh(mesh, xs)

#4. Generate "observed" data
generator = np.random.default_rng(0)
signal_to_noise = 1e10

Gu_ar = Gradu.dat.data_ro[:]
gradu_rangex = Gu_ar[:,0].max() - Gu_ar[:,0].min()
gradu_rangey = Gu_ar[:,1].max() - Gu_ar[:,1].min()
devx = Constant(gradu_rangex / signal_to_noise, domain=point_cloud)
devy = Constant(gradu_rangey / signal_to_noise, domain=point_cloud)
zetax, zetay = generator.standard_normal(len(xs)), generator.standard_normal(len(xs))
if(sampling_type == 'uniform_interval_sample'):
    nsamples = 10
    gradu_obs_vals = np.dstack((np.linspace(Gu_ar[:,0].min(), Gu_ar[:,0].max(), nsamples), \
                                np.linspace(Gu_ar[:,1].min(), Gu_ar[:,1].max(), nsamples)))[0]
elif(sampling_type == 'totally_random_sample' or sampling_type == 'uniform_domain_sample'):
    gradu_obs_vals = np.array(Gradu.at(xs)) + np.transpose(np.array([float(devx)*zetax , float(devy)*zetay]))
else:
    raise Exception('Sampling option not recognized')

Flux_ar = Flux.dat.data_ro[:]
flux_rangex = Flux_ar[:,0].max() - Flux_ar[:,0].min()
flux_rangey = Flux_ar[:,1].max() - Flux_ar[:,1].min()
devx = Constant(flux_rangex / signal_to_noise, domain=point_cloud)
devy = Constant(flux_rangey / signal_to_noise, domain=point_cloud)
zetax, zetay = generator.standard_normal(len(xs)), generator.standard_normal(len(xs))
if(sampling_type == 'uniform_interval_sample'):
    flux_obs_vals = np.dstack((np.linspace(Flux_ar[:,0].min(), Flux_ar[:,0].max(), nsamples), \
                               np.linspace(Flux_ar[:,1].min(), Flux_ar[:,1].max(), nsamples)))[0]
elif(sampling_type == 'totally_random_sample'or sampling_type == 'uniform_domain_sample'):
    flux_obs_vals = np.array(Flux.at(xs)) + np.transpose(np.array([float(devx)*zetax , float(devy)*zetay]))

#5. Store data as a function defined on the point_cloud
P0DG = VectorFunctionSpace(point_cloud, "DG", 0)
gradu_obs = Function(P0DG)
gradu_obs.dat.data[:] = gradu_obs_vals
flux_obs = Function(P0DG)
flux_obs.dat.data[:] = flux_obs_vals

# Add the origin
flag_add_origin = True
if(flag_add_origin):
    flux_obs_vals = np.append(flux_obs_vals, [[0.0, 0.0]], axis=0)
    gradu_obs_vals = np.append(gradu_obs_vals, [[0.0, 0.0]], axis=0)

observed_data = ComputeState(flux_obs_vals, gradu_obs_vals, flag_dist)

flag_plot = True
if(flag_plot):
    plt.figure(figsize= (9,6))
    plt.rcParams['font.size'] = '16'
    plt.axes(facecolor = "lightgray")
    plt.ylabel('$||{\sigma}||_2$',fontsize=20)
    plt.xlabel('$||grad(u)||_2$', fontsize=20)
    magflux = np.linalg.norm(flux_obs_vals, axis=1)
    maggrad = np.linalg.norm(gradu_obs_vals, axis=1)
    l = plt.plot(maggrad, magflux,'or')
    plt.setp(l, 'markersize', 10)
    plt.setp(l, 'markerfacecolor', 'white')
    #plt.legend(['', ''])
    plt.grid()
    plt.show()

def Computekdtree(observed_data, current_state):
  mytree = scipy.spatial.KDTree(observed_data)
  dist, indexes = mytree.query(current_state)
  return dist, indexes

start = time.time()
distances, indexes = Computekdtree(observed_data, current_state)
#print(distances, distances.shape)
#print(indexes, indexes.shape)
#raise Exception()
end = time.time()
print('Completed in: ',end-start)

# Space to which the optimization variables belong,
# which are totally independent of the spaces used to construct solutions
# for sampling purposes
Vu = FunctionSpace(mesh, family="CG", degree=1)
Vsigma = VectorFunctionSpace(mesh, family=DGtype, degree=0)

u = Function(Vu)
sigma, tau = Function(Vsigma), TestFunction(Vsigma)
v = TestFunction(Vu)

tau = TrialFunction(Vsigma)
a = (inner(-tau, grad(v)))*dx
petsc_mat = assemble(a).M.handle
indptr, indices, data = petsc_mat.getValuesCSR()
scipy_mat = sp.csr_matrix((data, indices, indptr), shape=petsc_mat.getSize())

#plt.spy(scipy_mat)
#plt.show()

# Define it once and for all
size_u = u.dat.data.shape[0]
size_sigma = sigma.dat.data.shape[0] # actually the size of one components of the flux
flux_start0, flux_end0 = 0, size_sigma
flux_start1, flux_end1 = flux_end0, flux_end0 + size_sigma
u_start, u_end = flux_end1, flux_end1 + size_u

scipy_mat_exp = sp.csr_matrix((size_u, (size_u+2*size_sigma)), dtype=float)
scipy_mat_zeros = sp.csr_matrix((size_u, size_u), dtype=float)
for i in range(size_u):
  scipy_mat_exp[i,0:flux_end1] = scipy_mat[i,:]
  scipy_mat_exp[i,u_start:u_end] = scipy_mat_zeros[i,:]

list_nodes = Vu.boundary_nodes('on_boundary')
nnod_boundary = len(list_nodes)
#print(list_nodes)
for i in list_nodes:
  scipy_mat_exp[i,:] = 0.0
#  scipy_mat_exp[i,2*size_sigma+i] = 1.0

scipy_mat_fireord = sp.csr_matrix((size_u, (size_u+2*size_sigma)), dtype=float)
for j, k in enumerate(range(0,2*size_sigma,2)):
    scipy_mat_fireord[:,j] = scipy_mat_exp[:,k]
    scipy_mat_fireord[:,j+size_sigma] = scipy_mat_exp[:,k+1]

'''
results = {
    "objective_function": [],
    "number_of_iterations": [],
    "error": [],
    "something": []
}
results["objective_function"].append(Jotap)
import json
with open("results.txt", "w") as output_file:
    json.dump(results, output_file, indent=2)
'''

flag_fixpoint = True

def ObFunc(x, sigma, u, v, params, observed_data, indexes, flag_fixpoint):

  # Assign the values in x to sigma and u    
  sigma.dat.data[:,0] = x[flux_start0:flux_end0]
  sigma.dat.data[:,1] = x[flux_start1:flux_end1]
  u.dat.data[:]       = x[u_start:u_end]
  #x[u_start:u_end] = u.dat.data[:]

  # Build the current state pairs (sigma,gu)
  VDG0 = VectorFunctionSpace(mesh, family=DGtype, degree=0)
  Gradu = project(grad(u), VDG0)
  gradu_vals = Gradu.dat.data 
   
  flux_vals = sigma.dat.data

  current_state = ComputeState(flux_vals, gradu_vals, flag_dist)
  '''
  distances = np.zeros(current_state.shape[0])
  for i in range(current_state.shape[0]):
      distances[i] = np.linalg.norm(current_state[i] - observed_data[indexes[i]], ord=2)
      print(distances[i])
  '''

  if(flag_fixpoint):
      distances = np.linalg.norm(current_state - observed_data[indexes], ord=2, axis=1)
  else:
      distances, indexes = Computekdtree(observed_data, current_state)

  p = 2
  Jotap = sum(area*dist**p for dist,area in zip(distances,area_els))
  
  return Jotap

eps_der = 1e-4

def derObFunc(x, sigma, u, v, params, observed_data, indexes, flag_fixpoint):

  grad_Jotap = np.zeros_like(x)
  VDG0 = VectorFunctionSpace(mesh, family=DGtype, degree=0)

  # Assign the values in x to sigma and u    
  sigma.dat.data[:,0] = x[flux_start0:flux_end0]
  sigma.dat.data[:,1] = x[flux_start1:flux_end1]
  u.dat.data[:]       = x[u_start:u_end]
  #x[u_start:u_end] = u.dat.data[:]
  Gradu = project(grad(u), VDG0)
  gradu_vals = Gradu.dat.data
  flux_vals = sigma.dat.data
  current_state = ComputeState(flux_vals, gradu_vals, flag_dist)
  if(flag_fixpoint):
      distances = np.linalg.norm(current_state - observed_data[indexes], ord=2, axis=1)
  else:
      distances, indexes = Computekdtree(observed_data, current_state)
  Jotap = sum(area*(dist**2) for dist,area in zip(distances,area_els))

  #----
  sigma.dat.data[:,0] = x[flux_start0:flux_end0]*(1 + eps_der)
  sigma.dat.data[:,1] = x[flux_start1:flux_end1]
  flux_vals = sigma.dat.data
  current_state = ComputeState(flux_vals, gradu_vals, flag_dist)
  if(flag_fixpoint):
      distances_px = np.linalg.norm(current_state - observed_data[indexes], ord=2, axis=1)
  else:
      distances_px, indexes = Computekdtree(observed_data, current_state)
  grad_Jotap[flux_start0:flux_end0] = area_els*(distances_px[0:size_sigma]**2 - distances[0:size_sigma]**2) / (x[flux_start0:flux_end0]*eps_der)

  #----
  sigma.dat.data[:,0] = x[flux_start0:flux_end0] 
  sigma.dat.data[:,1] = x[flux_start1:flux_end1]*(1 + eps_der)
  flux_vals = sigma.dat.data
  current_state = ComputeState(flux_vals, gradu_vals, flag_dist)
  if(flag_fixpoint):
      distances_py = np.linalg.norm(current_state - observed_data[indexes], ord=2, axis=1)
  else:
      distances_py, indexes = Computekdtree(observed_data, current_state)
  grad_Jotap[flux_start1:flux_end1] = area_els*(distances_py[0:size_sigma]**2 - distances[0:size_sigma]**2) / (x[flux_start1:flux_end1]*eps_der)
  
  #----
  sigma.dat.data[:,0] = x[flux_start0:flux_end0] 
  sigma.dat.data[:,1] = x[flux_start1:flux_end1]
  flux_vals = sigma.dat.data
  magflux = np.linalg.norm(flux_vals, axis=1)
  for ix, xu in enumerate(x[u_start:u_end]):
    if(np.abs(xu) > 1e-12):
      u.dat.data[ix] = xu*(1 + eps_der)
      Gradu = project(grad(u), VDG0)
      gradu_vals = Gradu.dat.data
      current_state = ComputeState(flux_vals, gradu_vals, flag_dist)
      if(flag_fixpoint):
          distances_pu = np.linalg.norm(current_state - observed_data[indexes], ord=2, axis=1)
      else:
          distances_pu, indexes = Computekdtree(observed_data, current_state)
      Jotap_p = sum(area*dist**2 for dist,area in zip(distances_pu,area_els))
      grad_Jotap[u_start+ix] = (Jotap_p - Jotap) / (xu*eps_der)
      u.dat.data[ix] = xu
  
  return grad_Jotap

# Initialization
# Add noise to the synthetic u and Flux solution to define an initial condition
signal_to_noise = 200
X = interpolate(mesh.coordinates, VDG0)
xs = X.dat.data
point_cloud = VertexOnlyMesh(mesh, xs)

Flux_ar = Flux.dat.data_ro[:]
flux_rangex = Flux_ar[:,0].max() - Flux_ar[:,0].min()
flux_rangey = Flux_ar[:,1].max() - Flux_ar[:,1].min()
devx = Constant(flux_rangex / signal_to_noise, domain=point_cloud)
devy = Constant(flux_rangey / signal_to_noise, domain=point_cloud)
zetax, zetay = generator.standard_normal(len(xs)), generator.standard_normal(len(xs))
flux_obs_vals = np.array(Flux.at(xs)) + np.transpose(np.array([float(devx)*zetax , float(devy)*zetay]))
Flux.dat.data[:] = flux_obs_vals

u_ar = u.dat.data_ro[:]
u_range = u_ar[:].max() - u_ar[:].min()
dev = Constant(u_range / signal_to_noise)
zeta = generator.standard_normal(len(u_ar))
u_obs_vals = u_ar[:] + np.array([float(dev)*zeta])
u.dat.data[:] = u_obs_vals

sigmaopt = Function(Vsigma)
uopt = Function(Vu)
sigmaopt.dat.data[:] = Flux.dat.data
urefproj = project(uref, Vu)
uopt.dat.data[:] = urefproj.dat.data
x0 = np.hstack([sigmaopt.dat.data_ro[:,0], sigmaopt.dat.data_ro[:,1], uopt.dat.data_ro[:]])

# Compute the initial indexes
Gradu = project(grad(uopt), VDG0)
gradu_vals = Gradu.dat.data 
flux_vals = sigmaopt.dat.data
current_state = ComputeState(flux_vals, gradu_vals, flag_dist)
distances, indexes = Computekdtree(observed_data, current_state)

# Test the value and gradient of the objective function
params = {}
Jotap = ObFunc(x0, Flux, u, v, params, observed_data, indexes, flag_fixpoint)
der_Jotap = derObFunc(x0, Flux, u, v, params, observed_data, indexes, flag_fixpoint)

# Linear restriction (as a nonlinear one)
def cons_f(x, Vsigma, v, f, bcuall):
  sigma = Function(Vsigma)
  sigma.dat.data[:,0] = x[flux_start0:flux_end0]
  sigma.dat.data[:,1] = x[flux_start1:flux_end1]
  L = (inner(-sigma, grad(v)) - f*v)*dx
  vec = assemble(L)
  mylist = []
  for i in range(len(vec.dat.data)):
    mylist.append(vec.dat.data[i])
  return mylist

def cons_J(x):
    return scipy_mat_fireord

# Test the nonlinear contraint
#nlcval = cons_f(x0, Vsigma, v, f, bcall)
#print(nlcval)
#nlc = NonlinearConstraint(lambda x: cons_f(x, Vsigma, v, f, bcall), 0.0, 0.0, jac='2-point', hess=BFGS())
#nlc = NonlinearConstraint(lambda x: cons_f(x, Vsigma, v, f, bcall), 0.0, 0.0, jac=cons_J, hess=BFGS())
#nlc = NonlinearConstraint(lambda x: cons_f(x, Vsigma, v, f, bcall), 0.0, 0.0, jac=cons_J, hess='2-point')

# Lower and upper bounds
lst_low, lst_up = [], []

# Fluxes are free
for i in range(2*size_sigma):
    lst_low.append(-np.inf)
    lst_up.append(np.inf)

for i in range(size_u):
    lst_low.append(-np.inf)
    lst_up.append(+np.inf)

# Boundary temperatures must satisfy prescribe Dirichlet data
for i in list_nodes:
  lst_low[2*size_sigma + i] = HomDirVal
  lst_up [2*size_sigma + i] = HomDirVal

bounds = Bounds(lst_low, lst_up)

# Linear constraint (Balance equation)
L = f*v*dx
vec = assemble(L)
vboundary = Constant(0.0) # Homogeneous condition for the test functions
bcall = DirichletBC(Vu, vboundary, "on_boundary")
bcall.apply(vec)
lower_bnd, upper_bnd = [], []
for i in range(size_u):
  lower_bnd.append(vec.dat.data_ro[i])
  upper_bnd.append(vec.dat.data_ro[i])

lc = LinearConstraint(scipy_mat_fireord, lower_bnd, upper_bnd)

#optim_solver = 'ipopt'
optim_solver = 'scipy'
if(optim_solver == 'scipy'):
    progress = []
    progress_val = []
    def cb(x, convergence):
        progress_val.append(ObFunc(x, Flux, u, v, params, observed_data, indexes, flag_fixpoint))
    #def cb(x):
    #    progress_val.append(ObFunc(x, Flux, u, params, observed_data, indexes, flag_fixpoint))

    tini = time.time()
    optim_iters = 3
    for it in range(optim_iters):
        res = minimize(ObFunc, x0, args=(Flux, u, v, params, observed_data, indexes, flag_fixpoint), method='trust-constr', \
                       jac=derObFunc, hess=BFGS(), bounds=bounds,
                       constraints=[lc], options={'verbose': 4, 'maxiter': 800}, callback=cb)
        
        # Compute the current indexes
        Flux.dat.data[:,0] = res.x[flux_start0:flux_end0]
        Flux.dat.data[:,1] = res.x[flux_start1:flux_end1]
        u.dat.data[:]      = res.x[u_start:u_end]
        Gradu = project(grad(u), VDG0)
        gradu_vals = Gradu.dat.data 
        flux_vals = Flux.dat.data
        current_state = ComputeState(flux_vals, gradu_vals, flag_dist)
        distances, indexes = Computekdtree(observed_data, current_state)
        x0 = res.x
        print(indexes)
        
    #res = minimize(ObFunc, x0, args=(Flux, u, v, params, observed_data), method='L-BFGS-B', \
    #               jac=derObFunc, hess=BFGS(),
    #               constraints=[lc], options={'disp': True, 'maxiter': 800}, callback=cb)
    print('|> Computing time=', time.time() - tini)

    plt.plot(progress_val, '-ob')
    plt.loglog()
    plt.savefig('history.png')
    plt.show()
    sigma.dat.data[:,0] = res.x[flux_start0:flux_end0]
    sigma.dat.data[:,1] = res.x[flux_start1:flux_end1]
    u.dat.data[:]       = res.x[u_start:u_end]
else:
    import cyipopt

    class DDCM:
        
        def __init__(self):
            pass

        def objective(self, x):
            #
            # The callback for calculating the objective
            #
            return ObFunc(x, Flux, u, v, params, observed_data, indexes, flag_fixpoint)

        def gradient(self, x):
            #
            # The callback for calculating the gradient
            #
            return derObFunc(x, Flux, u, v, params, observed_data, indexes, flag_fixpoint)

        def constraints(self, x):
            #
            # The callback for calculating the constraints
            #
            return np.array(cons_f(x, Vsigma, v, f, bcall))

        def jacobian(self, x):
            #
            # The callback for calculating the Jacobian
            #
            return cons_J(x).todense()

        def intermediate(
                self,
                alg_mod,
                iter_count,
                obj_value,
                inf_pr,
                inf_du,
                mu,
                d_norm,
                regularization_size,
                alpha_du,
                alpha_pr,
                ls_trials):
            
            #
            # Example for the use of the intermediate callback.
            #
            print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))

    cl, cu = [], []
    for i in range(size_u):
        cl.append(0.0)
        cu.append(0.0)
        
    nlp = cyipopt.Problem(n=len(x0),m=len(cl),
                          problem_obj=DDCM(),cl=cl, cu=cu)
    
    #
    # Set solver options
    #

    #nlp.addOption('derivative_test', 'second-order')
    #nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('tol', 1e-7)
    nlp.add_option('max_iter', 400)
    #
    # Scale the problem (Just for demonstration purposes)
    #
    #nlp.set_problem_scaling(obj_scaling=2, x_scaling=[1, 1, 1, 1])
    #nlp.add_option('nlp_scaling_method', 'user-scaling')

    #
    # Solve the problem
    #
    x, info = nlp.solve(x0)
    #print("Solution of the primal variables: x=%s\n" % repr(x))
    #print("Solution of the dual variables: lambda=%s\n" % repr(info['mult_g']))
    #print("Objective=%s\n" % repr(info['obj_val']))

    sigma.dat.data[:,0] = x[flux_start0:flux_end0]
    sigma.dat.data[:,1] = x[flux_start1:flux_end1]
    u.dat.data[:]       = x[u_start:u_end]

errorH1 = errornorm(u, uref , norm_type='H1')
errorL2 = errornorm(u, uref , norm_type='L2')
print('|---------------------------------')
print('|> mesh h= ', Lx/Nx)
print('|> eL2= ', errorL2)
print('|> eH1= ', errorH1)
print('|---------------------------------')
#print('Balance eq.=', cons_f(res.x, Vsigma, v, f, bcall))

File("temp.pvd").write(project(u, Vu, name="Temperature"))
Gradu = project(grad(u), VDG0)
File("gradu.pvd").write(project(Gradu, VDG0, name="GradT"))
File("flux.pvd").write(project(sigma, VDG0, name="Flux"))

from firedrake import *
from firedrake.pyplot import tricontourf
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sys

"""SOME COMMENTS ABOUT THE CODE: 

    - The first part of the code is dedicated to define all the functions that will be used. In this sense, we've defined:
        - The function to generate the rotation matrix;
        - The function to solve the Poisson problem simulating an experimental apparatus
        - The functions to solve the current problem in two different formulations (Poisson CG and PoissonHdiv);
        - The function to define the flux law used in the PoissonHdiv;
        - The function to compute the state of the system (pressure gradients and fluxes); and
        - The function to compute the nearest neighbors using the KDTree algorithm.
    
    - The second part of the code is dedicated to solve the current problem. In this sense, we've defined the domain parameters, the permeability, 
      the boundary conditions, the source term, and the formulation to be used. Then, we've called the function to solve the problem and recover 
      the pressure gradient and the corresponding flux on each element. Finally, we've saved the reference solution for further visualization.

    - The third part of the code is dedicated to generate the experimental data for the data-driven computational mechanics approach. In this sense,
      we've defined two options: the first one is to generate the experimental data using the Poisson problem, which simulates an experimental apparatus,
      and the second one is sampling the solution obtained by the  previous current problem.  
      
    - The fourth part of the code is dedicated to compute the state of the system (pressure gradients and fluxes) for the observed data and the current state.
    
    - The fifth part of the code is dedicated to compute the nearest neighbors using the KDTree algorithm.
    
    - The last part of the code is dedicated to apply the data-driven computational mechanics approach. In this sense, we've defined two formulations: the first
      one is called Formulation 5B, and the second one is called Formulation 5A.    

"""

"   ===================== SETTING THE FEM SOLVER =====================    "

solpet = {
    'snes_type': 'newtonls',
    'snes_monitor': None,
    'snes_converged_reason': None,
    'snes_linesearch_type': 'basic',
    'snes_max_it': 100,
    'snes_divergence_tolerance': 1e20,
    'snes_rtol': 1e-9,
    'snes_atol': 1e-11,
    'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
}

"   ============================ FUNCTIONS ============================    "
    
def RotationalMatrix(theta):
    "Generate a 2D rotation matrix."
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]]) 
    

def ExperimentalPoissonCG(mesh,Time, dt, t, theta, dtheta, n, xs):
    "Generate experimental data for the data-driven computational mechanics approach."
    
    # Function space
    W = FunctionSpace(mesh, 'CG', 1)
    
    # Permeability
    kappa = Constant(1.0)

    # Source term
    f = Constant(0.0)

    # Imposed Neumann BC
    ql = Constant(0.0); qr = Constant(0.0)

    # Variational Formulation
    u = TrialFunction(W)
    v = TestFunction(W)
    a = inner(kappa*grad(u), grad(v))*dx
    L = f*v*dx - ql*v*ds(1) - qr*v*ds(3)
    w = Function(W, name='Pressure')

    # Boundary Conditions
    bc_bottom = DirichletBC(W, Constant(0.0), 3)  # Bottom boundary condition
    p = Constant(1.0*t+0.01)       # Pressure on the top boundary (time dependent b.c.)
    bc_top = DirichletBC(W, p, 4)  # Top boundary condition

    # Allocating memory to store the rotated solutions
    grad_rot = [];  flux_rot = []
   
    #Output Files
    outfile_u = File('Pressure_exp.pvd');  outfile_gradu = File('Gradu_exp.pvd'); outfile_flux = File('Flux_exp.pvd')
      
    # Solve
    while t <= Time:
        
        # Solve the Poisson problem
        solve(a == L, w, bcs=[bc_bottom, bc_top])
        
        print(f"Solution at time {t:.2f}\t===\tOK")
        
        # Obtain the gradient and flux 
        gradu = project(grad(w), VectorFunctionSpace(mesh, 'DG', 0),name='Gradu')
        flux = project(-kappa*grad(w), VectorFunctionSpace(mesh, 'DG', 0),name = 'Flux')
        
        # Save solutions for each time step
        outfile_u.write(w, time=t); outfile_gradu.write(gradu, time=t); outfile_flux.write(flux, time=t)
            
        """We should notice that the pressure gradient and the flux are constants in the entire domain for each time step,
           so we just need to extract one value for each variable and rotate it."""
        
        gradu_aux = np.array(gradu.at(xs))
        flux_aux = np.array(flux.at(xs))
         
        # Rotation 
        while n <= theta:

            # Rotate the obtained solutions
            R = RotationalMatrix(n)
            gradu_rot_values = np.reshape([R.dot(vec) for vec in gradu_aux], (-1, 2))
            flux_rot_values = np.reshape([R.dot(vec) for vec in flux_aux], (-1, 2))
            
            #print(f'Angle {n:.2f}\t+++++\tOK\n')
            
           
            # Store the rotated solutions
            grad_rot.append(gradu_rot_values[:,:])
            flux_rot.append(flux_rot_values[:,:])   
                     
            # Update the angle 
            n += dtheta
            
        # Reseting the angle    
        n = 0    
        # Update time and boundary condition
        t += dt
        p.assign(1.0*t+0.01)
            
    return w, gradu, flux, grad_rot, flux_rot
 
 
def PoissonCG(mesh, kappa, f, pressure_dirichlet, orderU):
    
    # Construct the function space
    W = FunctionSpace(mesh, "CG", orderU)

    # Define the trial and test functions
    u = TrialFunction(W)  
    v = TestFunction(W)

    # Variational form
    a = inner(kappa*grad(u),grad(v))*dx
    L = f*v*dx

    # Apply boundary conditions in the entire domain (Dirichlet)
    bc = DirichletBC(W, pressure_dirichlet, "on_boundary")

    # Solve the problem
    w = Function(W, name='Pressure')
    solve(a == L, w, bcs=[bc])

    return w


PorousMedia_model = 'Standard'
#PorousMedia_model = 'ForchheimerLin'
def FluxLaw(sigma, u):
  if(PorousMedia_model == 'Standard'):
    kappa = Constant(1.0)
    
    return sigma + kappa * grad(u)

  elif(PorousMedia_model == 'ForchheimerLin'):
    muokap = Constant(1.0); betarho = Constant(10.0)
    eta = Constant(0.0001); modreg = eta*(sqrt(1 + inner(sigma,sigma)/eta**2) - 1.0)
    
    return grad(u) + muokap*sigma + betarho * modreg * sigma


def PoissonHdiv(mesh, f, pressure_dirichlet, Quad_mesh, orderF=1, orderU=2):

  # Define the function spaces
  if(Quad_mesh):
    HD = FunctionSpace(mesh, "RTCF", orderF)
    CG = FunctionSpace(mesh, "CG", orderU)
  else:
    HD = FunctionSpace(mesh, "DRT", orderF)
    CG = FunctionSpace(mesh, "CG", orderU)

  # Mixed finite element function space
  W = HD * CG

  w = Function(W)
  sigma, u = split(w)
  tau, v = TestFunctions(W)
  
  # Variational form
  Res  = (inner(FluxLaw(sigma,u),tau))*dx
  Res += inner(sigma,grad(v))*dx
  Res += f*v*dx

  # Define boundary conditions
  bcall = DirichletBC(W.sub(1), pressure_dirichlet, "on_boundary")

  # Solve the problem
  solve(Res == 0, w, bcs=[bcall], solver_parameters=solpet)
  sigma, u = w.sub(0), w.sub(1)

  return sigma, u


def ComputeState(flux, grad, State_option):
  if(State_option == 'R4'):
    state_vector = np.column_stack((grad, flux))
  elif(State_option == 'R2'):
    magflux = np.linalg.norm(flux, axis=1)
    maggrad = np.linalg.norm(flux, axis=1)
    state_vector = np.column_stack((maggrad, magflux))
  else:
    raise ValueError('State_option not recognized')
  return state_vector


def KDTree(observed_data, current_state):
  mytree = sp.spatial.KDTree(observed_data)
  dist, indexes = mytree.query(current_state, k=1)
  return dist, indexes

 
"   ============================ MAIN CODE ============================    "



"PART 01: SOLVE THE CURRENT PROBLEM "



# Domain parameters
Lx, Ly = 1.0, 1.0
Nx, Ny = 25, 25 
mesh = RectangleMesh(Nx, Ny, Lx, Ly, quadrilateral=False) 
x, y = SpatialCoordinate(mesh)


# Permeability 
Constant_perm = True
if (Constant_perm):
  # Constant
    kappa = Constant(1.0)
else:
  # Gaussian
    kappa0 = Constant(1.0)
    A = Constant(1.0)
    kappa = kappa0 + exp(-A*((x - Lx/2)**2 + (y - Ly/2)**2))
    
# Boundary conditions
pressure_dirichlet = Constant(1.0)

# Source term 
f = project(Constant(3.0),FunctionSpace(mesh, "DG", 0))

Formulation_solver = 'Primal-CG'
#Formulation_solver = 'DualMixed-Hdiv'
print(f'Solving the current problem with: {Formulation_solver}\n')

if(Formulation_solver == 'Primal-CG'):
    
  # Call the Poisson solver function
  orderU=1
  u = PoissonCG(mesh, kappa, f, pressure_dirichlet, orderU)
  
  # Recover the pressure gradient and the corresponding flux on each element
  Gradu = project(grad(u), VectorFunctionSpace(mesh, family='DG', degree=0),name = 'Gradu')
  gradu_vals = Gradu.dat.data
  Flux = project(-kappa*grad(u), VectorFunctionSpace(mesh, family='DG', degree=0),name = 'Flux')
  flux_vals = Flux.dat.data

  # Save the reference solution for further visualization
  File("Pressure_PrimalCG_ref.pvd").write(u)
  File("Gradu_PrimalCG_ref.pvd").write(Gradu)
  File("Flux_PrimalCG_ref.pvd").write(Flux)
        
elif(Formulation_solver == 'DualMixed-Hdiv'):
  
  Quad_mesh = False
  orderF, orderU = 2, 1
  
  fluxhdiv, u =  PoissonHdiv(mesh, f, pressure_dirichlet, Quad_mesh, orderF, orderU)
  u = Function(u.function_space(),name = 'Pressure')
  

  # Recover the pressure gradient and the corresponding flux on each element
  Gradu = project(grad(u), VectorFunctionSpace(mesh, family='DG', degree=0),name = 'Gradu')
  gradu_vals = Gradu.dat.data
  Flux = project(fluxhdiv, VectorFunctionSpace(mesh, family='DG', degree=0),name='Flux')
  flux_vals = Flux.dat.data
  
  # Save the reference solution for further visualization
  File("Pressure_DualMixedHdiv_ref.pvd").write(u)
  File("Gradu_DualMixedHdiv_ref.pvd").write(Gradu)
  File("Flux_DualMixedHdiv_ref.pvd").write(Flux)
        
else:
        
  raise ValueError('Formulation_solver option not recognized\n')



"PART 02: SAMPLING OR EXPERIMENTAL DATA"



Sampling_data = 'Experimental'
#Sampling_data = 'Numerical'
print(f'Sampling process: {Sampling_data}\n')

if(Sampling_data == 'Experimental'):
  
  Nsamples_x = 4; Nsamples_y = 4
  Deltax = Lx / Nsamples_x
  Deltay = Ly / Nsamples_y
  xls = np.linspace(0.5*Deltax, Lx-0.5*Deltax, Nsamples_x)
  yls = np.linspace(0.5*Deltay, Ly-0.5*Deltay, Nsamples_y)
  xlsg, ylsg = np.meshgrid(xls, yls)
  xs = np.dstack([xlsg, ylsg]).reshape(Nsamples_x*Nsamples_y,2)
  
    
  # Time and angle parameters
  Time =  1;   N_time = 10;    dt = Time/N_time;   t0 = 0
  theta = np.pi;     N_theta = 4;   dtheta = theta/N_theta;   theta0 = 0.    
  visualization = False
  
  # Call the experimental Poisson solver function
  [u_exp, gradu_exp, flux_exp, grad_rot, flux_rot] = ExperimentalPoissonCG(mesh, Time, dt, t0, theta, dtheta, theta0, xs)

  # Convert the lists to arrays
  grad_rot = np.concatenate(np.asarray(grad_rot),axis=0)
  flux_rot = np.concatenate(np.asarray(flux_rot),axis=0)
  
  # Save the experimental data
  gradu_obs_vals, flux_obs_vals = grad_rot, flux_rot
  
  if(visualization):
    # Reshape the arrays
    grad_rot = grad_rot.reshape(N_time+1,N_theta+1,-1)
    flux_rot = flux_rot.reshape(N_time+1,N_theta+1,-1)
    
    # Evaluate the L2 norm of the gradient and flux
    grad_rot_normalized = np.linalg.norm(grad_rot,ord = 2,axis=2)
    flux_rot_normalized = np.linalg.norm(flux_rot,ord = 2,axis=2)
    
    # Reshape the normalized arrays
    grad_rot_normalized = grad_rot_normalized.reshape((N_time+1)*(N_theta+1),1)
    flux_rot_normalized = flux_rot_normalized.reshape((N_time+1)*(N_theta+1),1)

    plt.figure(figsize=(12,8))
    plt.scatter(grad_rot_normalized,flux_rot_normalized, label='$||u||_{L2}$',marker = 'o')
    plt.title('Numerical solution to the Poisson problem')
    plt.xlabel('$||grad(u)||$', fontsize=12)
    plt.ylabel('$||\sigma||$', fontsize=12)
    plt.grid()
    plt.show()
              
elif(Sampling_data == 'Numerical'):
  
  Nsamples_x = 5; Nsamples_y = 5
  Deltax = Lx / Nsamples_x
  Deltay = Ly / Nsamples_y
  xls = np.linspace(0.5*Deltax, Lx-0.5*Deltax, Nsamples_x)
  yls = np.linspace(0.5*Deltay, Ly-0.5*Deltay, Nsamples_y)
  xlsg, ylsg = np.meshgrid(xls, yls)
  xs = np.dstack([xlsg, ylsg]).reshape(Nsamples_x*Nsamples_y,2)

  
  # Define a generator to produce the random numbers
  generator = np.random.default_rng(0)
  # Define the white noise to be added to the data
  signal_to_noise = 1e9

  # Extract the data from the standard solution (pressure gradient)
  Gu_ar = Gradu.dat.data_ro[:]

  # Evaluate the range variation of the data in each coordinate
  gradu_rangex = Gu_ar[:,0].max() - Gu_ar[:,0].min()
  gradu_rangey = Gu_ar[:,1].max() - Gu_ar[:,1].min()

  # Divide the range by the signal to noise ratio
  devx = Constant(gradu_rangex / signal_to_noise)
  devy = Constant(gradu_rangey / signal_to_noise)

  # Generate two matrix of random numbers with normal distribution (mean = 0, std = 1)
  zetax, zetay = generator.standard_normal(len(xs)), generator.standard_normal(len(xs))

  # Observed data (pressure gradient) (standard solution + white noise)
  gradu_obs_vals = np.array(Gradu.at(xs)) + np.transpose(np.array([float(devx)*zetax , float(devy)*zetay]))
  
  # Extract the data from the standard solution (flux)
  Flux_ar = Flux.dat.data_ro[:]

  # Evaluate the range variation of the data in each coordinate
  flux_rangex = Flux_ar[:,0].max() - Flux_ar[:,0].min()
  flux_rangey = Flux_ar[:,1].max() - Flux_ar[:,1].min()

  # Divide the range by the signal to noise ratio
  devx = Constant(flux_rangex / signal_to_noise)
  devy = Constant(flux_rangey / signal_to_noise)

  # Generate two matrix of random numbers with normal distribution (mean = 0, std = 1)
  zetax, zetay = generator.standard_normal(len(xs)), generator.standard_normal(len(xs))

  # Observed data (flux) (standard solution + white noise
  flux_obs_vals = np.array(Flux.at(xs)) + np.transpose(np.array([float(devx)*zetax , float(devy)*zetay]))
   
else:
  
  raise ValueError('Sampling_data option not recognized\n')
            
            

"PART 03: COMPUTE STATES"


State_option = 'R4'
#State_option = 'R2'

# Compute the state of pressure gradient and flux for the observed data
observed_data = ComputeState(flux_obs_vals, gradu_obs_vals, State_option)
# Compute the state of pressure gradient and flux for the current state
current_state = ComputeState(flux_vals, gradu_vals, State_option)



"PART 04: KDTREE AND SAVE ASSIGNATION"


# Call the KDTree function
distances, indexes = KDTree(observed_data, current_state)



# Plot the observed data, the current state, and the nearest neighbors
KDTree_plot = True
if(KDTree_plot):
  ordem = 2
  plt.figure(figsize=(8, 6))
  plt.scatter(np.linalg.norm(observed_data[:,:2],ord=ordem,axis=1), np.linalg.norm(observed_data[:,-2:],ord=ordem,axis=1), marker='^', color='blue', label='Observed Data')
  plt.scatter(np.linalg.norm(current_state[:,:2],ord=ordem,axis=1), np.linalg.norm(current_state[:,-2:],ord=ordem,axis=1), marker='.', color='red', label='Current State')
  plt.scatter(np.linalg.norm(observed_data[indexes,:2],ord=ordem,axis=1), np.linalg.norm(observed_data[indexes,-2:],ord=ordem,axis=1), marker='x', color='green', label='Nearest Neighbors')
  #for k in range(len(indexes)):
  #  plt.plot([np.linalg.norm(current_state[k,:2],ord=ordem), np.linalg.norm(observed_data[indexes[k], :2],ord=ordem)], \
  #          [np.linalg.norm(current_state[k,-2:],ord=ordem), np.linalg.norm(observed_data[indexes[k], -2:],ord=ordem)], linestyle = '--', color="0.5")
  #  
  #  plt.text((np.linalg.norm(current_state[k,:2],ord=ordem) + np.linalg.norm(observed_data[indexes[k], :2],ord=ordem)) / 2, \
  #          (np.linalg.norm(current_state[k,-2:],ord=ordem) + np.linalg.norm(observed_data[indexes[k], -2:],ord=ordem)) / 2, f'{distances[k]:.2f}', color='k')
    
  plt.xlabel('$||grad(u)||_{L^{2}}$')
  plt.ylabel('$||{\sigma}||_{L^{2}}$')
  plt.title('KDTree Nearest Neighbors')
  #plt.xscale('log')
  #plt.yscale('log')
  plt.legend()
  plt.grid(True)
  plt.show()


# Define the pressure gradient data and the flux data as functions
gradpdata = Function(VectorFunctionSpace(mesh, family='DG', degree=0),name = 'GradP-data') 
fluxdata = Function(VectorFunctionSpace(mesh, family='DG', degree=0),name = 'Flux-data')

# Define the pressure gradiente data, based on the indexes of the nearest neighbors determined previously   
gradpdata.dat.data[:] = observed_data[indexes,0:2]

# Define the flux data, based on the indexes of the nearest neighbors determined previously
fluxdata.dat.data[:] = observed_data[indexes,2:4]

# Save the pressure gradient and the flux data
File("Gradu_data.pvd").write(gradpdata)
File("Flux_data.pvd").write(fluxdata)


"PART 05: DDCM"



"Formulation: 5B"



print('|>\tStarting the DDCM approach - Formulation 5B\t<|\n')

# Space of piecewise continuous functions
CG1  = FunctionSpace(mesh, family="CG", degree=1)

# Space of piecewise constant vector functions
VDG0 = VectorFunctionSpace(mesh, family='DG', degree=0)

# Mixed spaces
W = VDG0 * CG1 * VDG0 * CG1 * VDG0

# Unknown field
w = Function(W)

q, p, g, l, m = split(w)

# Test functions
dq, dp, dg, dl, dm = TestFunctions(W)

# Functional to be minimized
J = inner(q-fluxdata, q-fluxdata)*dx + inner(g-gradpdata, g-gradpdata)*dx

# First variations
dqJ = derivative(J, q, dq)
dgJ = derivative(J, g, dg)

def Compatibility(g, p):
  return g - grad(p)

# Residual
dL  =  dqJ + dgJ
dL +=  -inner(dq, grad(l)) * dx
dL += (-inner(q, grad(dl)) - f * dl ) * dx
dL += inner(Compatibility(g,p), dm) * dx
dL += inner(dg, m) * dx - inner(m, grad(dp)) * dx

# Jacobian
Jac = derivative(dL, w)

# Boundary condition for pressure
bcp = DirichletBC(W.sub(1), Constant(pressure_dirichlet), "on_boundary")

# Homegeneous boundary condition for scalar Lagrange multiplier
bcl = DirichletBC(W.sub(3), Constant(0.0), "on_boundary")

# Non-linear solve (see solver parameters above)
solve(dL == 0, w, bcs=[bcp, bcl], J=Jac, solver_parameters=solpet)

# Split the solution for post-processing
qpl, ppl, gpl, lpl, mpl = split(w)

# Check mass conservation
flux = Function(VDG0)
flux.assign(w.sub(0))
L = (inner(-flux, grad(dl)) - f*dl)*dx
vec = assemble(L)
aux = Function(CG1)
aux.dat.data[:] = vec.dat.data[3] # I don't understand the behavior of the cofunction stuff

# Recall the Lagrange multiplier satisfies the essential B.C.
bc = DirichletBC(CG1, Constant(0.0), "on_boundary")
bc.apply(aux)

print('Formulation 5B: total mass loss= ', np.linalg.norm(aux.dat.data[:]))

# Save fields for visualization in Paraview
File("Flux5B.pvd").write(project(qpl, VDG0, name="Flux"))
File("Pressure5B.pvd").write(project(ppl, CG1, name="Pressure"))
File("Grad5B.pvd").write(project(gpl, VDG0, name="GradP"))
File("lambda5B.pvd").write(project(lpl, CG1, name="Lambda"))
File("mu5B.pvd").write(project(mpl, VDG0, name="mu"))


# Evaluate the error for formulation 5B
pnormL2_5B    = errornorm(u_exp, w.sub(1), norm_type='L2')
pnormH1_5B    = errornorm(u_exp, w.sub(1), norm_type='H1')
fluxnormL2_5B = errornorm(flux_exp, w.sub(0), norm_type='L2')
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
print(f'p_L2:{pnormL2_5B:.6} \t p_H1:{pnormH1_5B:.6} \t u_L2: {fluxnormL2_5B:.6}\n')
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')


sys.exit()

"Formulation: 5A"



print('|>\tStarting the DDCM approach - Formulation 5A\t<|\n')

orderFlux, orderPres = 2, 1
# Space of piecewise continuous functions
RT  = FunctionSpace(mesh, family='RT', degree=orderFlux)

# Space of piecewise constant vector functions
DG = FunctionSpace(mesh, family='DG', degree=orderPres)

# Mixed spaces
W = RT * DG * RT * DG * RT

# Unknown field
w = Function(W)
q, p, g, l, m = split(w)

# Test functions
dq, dp, dg, dl, dm = TestFunctions(W)

# Functional to be minimized
J = inner(q-fluxdata, q-fluxdata)*dx + inner(g-gradpdata, g-gradpdata)*dx

# First variations
dqJ = derivative(J, q, dq)
dgJ = derivative(J, g, dg)

n = FacetNormal(mesh)

# Residual
dL  =  dqJ + dgJ
dL +=  inner(div(dq), l) * dx
dL += (inner(div(q), dl) - f * dl ) * dx
dL += inner(dg, m) * dx  + inner(dp, div(m)) * dx
dL += inner(dm, g) * dx + inner(p, div(dm)) * dx - Constant(pressure_dirichlet) * inner(dm,n) * ds

# Jacobian
Jac = derivative(dL, w)

# Non-linear solve (see solver parameters above)
solve(dL == 0, w, bcs=[], J=Jac, solver_parameters=solpet)

# Split the solution for post-processing
qpl, ppl, gpl, lpl, mpl = split(w)

# Save fields for visualization in Paraview
File("Flux5A.pvd").write(project(qpl, RT, name="Flux"))
File("Pressure5A.pvd").write(project(ppl, DG, name="Pressure"))
File("Grad5A.pvd").write(project(gpl, RT, name="GradP"))
File("lambda5A.pvd").write(project(lpl, DG, name="Lambda"))
File("mu5A.pvd").write(project(mpl, RT, name="mu"))


# Evaluate the error for formulation 5A
pnormL2_5A    = errornorm(u, w.sub(1), norm_type='L2')
fluxnormL2_5A = errornorm(Flux, w.sub(0), norm_type='L2')
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
print(f'p_L2:{pnormL2_5A:.6} \t u_L2: {fluxnormL2_5A:.6}\n')
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

sys.exit()

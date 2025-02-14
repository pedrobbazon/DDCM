# Import

from firedrake import *

from firedrake.adjoint import *

import cyipopt

from firedrake.__future__ import interpolate



# Begin tape annotation for adjoint

continue_annotation()



# Define mesh and function spaces

n = 250

mesh = UnitSquareMesh(n, n)

A = FunctionSpace(mesh, "CG", 1)  # function space for control

P = FunctionSpace(mesh, "CG", 1)  # function space for solution



# Define constants

V = Constant(0.4)  # volume bound on the control

p = Constant(5)  # power used in the solid isotropic material

eps = Constant(1.0e-3)  # epsilon used in the solid isotropic material

alpha = Constant(1.0e-8)  # regularisation coefficient in functional



# Create file for plotting in Paraview

controls = VTKFile("ouput/controls.pvd")



# Conductivity with penalisation

def k(rho):

    return eps + (1 - eps) * rho ** p



def forward_solve(x):



    # Convert numpy vector of design variable to Firedrake

    rho = Function(A, name ='density')

    rho.vector()[:] = x



    # Define functions

    T = Function(P, name="Temperature")

    v = TestFunction(P)



    # Define volume source term

    f_i = interpolate(Constant(1.0e-2), P)

    f = assemble(f_i)



    # Define BCs

    bc1 = DirichletBC(P,0.0,1)

    bc2 = DirichletBC(P,0.0,4)

    bc = [bc1,bc2]



    # Define varitaitional form and solve forward problem

    F = inner(grad(v), k(rho) * grad(T)) * dx - f * v * dx

    solve(F==0,T,bcs=bc)



    # Define compliance (objective function)

    J = assemble(f * T * dx + alpha * inner(grad(rho), grad(rho)) * dx)



    # Compute gradient of objective function and convert to numpy vector

    m = Control(rho)

    dJdrho = compute_gradient(J, m)

    

    dJdrho_np = dJdrho.vector()[:]

    return J, dJdrho_np



# Computes volume fraction and its gradient

def volume_frac(x):



    # Convert numpy vector of design variable to Firedrake

    rho = Function(A, name ='density')

    rho.vector()[:] = x



    # Computes current volume

    vol = assemble(rho*dx)



    # Compute gradient of volume fraction

    m = Control(rho)

    dVoldrho = compute_gradient(vol, m)

    dVoldrho_np = dVoldrho.vector()[:]



    return vol, dVoldrho_np



def plotter(x):

    

    # Convert numpy vector of design variable to Firedrake

    rho_viz = Function(A, name ='density')

    rho_viz.vector()[:] = x



    # Save to pvd file

    controls.write(rho_viz)



    return controls



# Define initialisation of density

rho_0 = interpolate(V, A)

x0 = assemble(rho_0).vector()[:]



# Define bounds for constraints and density

lb = np.zeros((len(x0),))

ub = np.ones((len(x0),))

cl = [0.0]

cu = [0.4]



# Class for ipopt

class optimization(object):

    def __init__(self):

        pass



    def objective(self, x):

        # Callback for calculating the objective



        J, dJdrho_np = forward_solve(x)

        controls = plotter(x)



        return J



    def gradient(self, x):

        

        J, dJdrho_np = forward_solve(x)



        return dJdrho_np



    def constraints(self, x):

        

        vol, dVoldrho_np = volume_frac(x)



        return vol



    def jacobian(self, x):

        

        vol, dVoldrho_np = volume_frac(x)



        return dVoldrho_np



# Define nonlinear optimization problem for ipopt

nlp = cyipopt.Problem(

            n=len(x0),

            m=1,

            problem_obj=optimization(),

            lb=lb,

            ub=ub,

            cl=cl,

            cu=cu

            )



# Define solver options

# nlp.addOption('mu_strategy', 'adaptive')

nlp.addOption('tol', 1e-20)

nlp.addOption('max_iter', 100)

nlp.addOption('acceptable_tol',1e-12)



# Solve using ipopt

x, info = nlp.solve(x0)

x_pvd = Function(P)

x_pvd.vector()[:] = x

VTKFile("solution.pvd").write(x_pvd)





tape = get_working_tape()

tape.clear_tape()

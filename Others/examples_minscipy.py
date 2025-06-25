from firedrake import *
from firedrake.adjoint import *

continue_annotation()

# define the space and sources
mesh = UnitIntervalMesh(3000000)
V = FunctionSpace(mesh, "CG", 1)
u = Function(V)
uref = Function(V)
v = TestFunction(V)

# compute target solution
with stop_annotating():
  x = SpatialCoordinate(mesh)
  Fref = inner(grad(v), grad(uref)) * dx + 100.0*v*uref*dx - v*cos(pi * x[0]**2)*dx
  solve(Fref == 0, uref)

# create forward problem
source = Function(V)
F = inner(grad(v), grad(u)) * dx + 100.0*v*u*dx - v*source*dx
solve(F == 0, u)

# define the reduced objective functional
J = assemble((u - uref)**2 * dx)
rf = ReducedFunctional(J, Control(source))

# define the boundary conditions
T_lb = Function(V, name="Lower bound")
T_ub = Function(V, name="Upper bound")
T_lb.assign(-1.0)
T_ub.assign(1.0)

# Setting up the problem using minimize that uses Scipy
sol = minimize(rf, method='Newton-CG', bounds=(T_lb, T_ub), tol=1e-12, options={"disp": True, "maxiter": 5})

from firedrake import *
from firedrake.adjoint import *

mesh = UnitSquareMesh(3, 3)
V = FunctionSpace(mesh, "DG", 0)
u = Function(V)
v = TestFunction(V)
f = Function(V)

continue_annotation()
solve((u - f)*v*dx == 0, u)
J = assemble((u - Constant(1))**2 * dx) 
J_post = lambda J, *args: print("norm(u) = ", norm(u))  # print quantity of interest on newest value of state
Jred = ReducedFunctional(J, Control(f), eval_cb_post=J_post)
stop_annotating()

# basic steepest descent
for ii in range(2):
  print("J(ii = ", ii, ") =", Jred(f))

  # compute the gradient
  gradJ = Jred.derivative()

  # update control
  f -= 0.2*gradJ
print("J(final) =", Jred(f))

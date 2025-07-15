#%%
from firedrake import *
from firedrake.__future__ import interpolate
import firedrake.adjoint as fda
import numpy as np
import matplotlib.pyplot as plt

mesh = RectangleMesh(20, 30, 1, .5, .0, -.5)

# Mock physical problem: r_max -> limits
mesh_physical = Mesh(mesh.coordinates.copy(deepcopy=True))
x, y = SpatialCoordinate(mesh_physical)
V = FunctionSpace(mesh_physical, "CG", 1)
u = Function(V)
v = TestFunction(V)
fda.continue_annotation()
r_max = Constant(0.1)  # This is the parameter we will optimise
y_max = .2
x_max = r_max
vertical_limit = (abs(y_max-abs(y)) + (y_max-abs(y))) / (2*y_max)
horizontal_limit = (abs(x_max-abs(x)) + (x_max-abs(x))) / (2*x_max)
threshold = vertical_limit * horizontal_limit
F_physical = u*v*dx - threshold*v*dx
bcs_physical = [DirichletBC(V, .0, (2,3,4))]
nvp_physical = NonlinearVariationalProblem(F_physical, u, bcs=bcs_physical)
nvs_physical = NonlinearVariationalSolver(nvp_physical)
nvs_physical.solve()
limits = abs(u**1/2)

# Mock interpolation problem: limits -> dT
mesh_optimisation = Mesh(mesh.coordinates.copy(deepcopy=True))
n_optimisation = FacetNormal(mesh_optimisation)
Q_optimisation = mesh_optimisation.coordinates.function_space()
n = FacetNormal(mesh_optimisation)
dT = Function(Q_optimisation)
optimisation_initial_coordinates = Function(Q_optimisation)
optimisation_initial_coordinates.dat.data[:] = mesh_physical.coordinates.dat.data[:]
mesh_optimisation.coordinates.assign(optimisation_initial_coordinates + dT)

S = FunctionSpace(mesh_optimisation, "CG", 1)

limits_interpolated = assemble(interpolate(limits, S, allow_missing_dofs=True))
limits_penalty = assemble(limits_interpolated * dx(degree=2))

J = limits_penalty
Jred = fda.ReducedFunctional(J, fda.Control(dT))
fda.stop_annotating()
opts = {"riesz_representation": None}
dJ = Jred.derivative(options=opts)
gradJ = Function(Q_optimisation)
W = TestFunction(Q_optimisation)
bcs = DirichletBC(Q_optimisation , as_vector([0,0]), [2,3,4])
F = inner(grad(gradJ), grad(W))*dx + dot(gradJ, W)*dx - dJ
nvp_optimisation = NonlinearVariationalProblem(F, gradJ, bcs=bcs)
nvs_optimisation = NonlinearVariationalSolver(nvp_optimisation)

for rampf in np.linspace(0.1, .5, num=5):
    r_max.assign(rampf)
    nvs_physical.solve()
    optimisation_initial_coordinates.dat.data[:] = mesh_physical.coordinates.dat.data[:]
    limits_interpolated.assign(assemble(interpolate(limits, S, allow_missing_dofs=True)))
    dT.assign(.0)
    ii = 0
    tripcolor(limits_interpolated)
    plt.show()
    while Jred(dT) > 2e-3 and ii < 20:
        triplot(mesh_optimisation)
        plt.show()
        print("J(ii =", ii, ") =", Jred(dT))
        dJ.assign(Jred.derivative(options=opts))
        bcs.zero(dJ)
        nvs_optimisation.solve()
        dT -= 0.25*gradJ
        ii += 1
    tripcolor(limits_interpolated)
    mesh_physical.coordinates.dat.data[:] += dT.dat.data[:]


# %%

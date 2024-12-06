from firedrake import *
import firedrake.adjoint as fda

fda.continue_annotation()

mesh = UnitSquareMesh(20, 20)
T = mesh.coordinates
Q = T.function_space()
dT = Function(Q)  # this will be the control

# create embedded mesh; dT2 will represent dT on it
mesh2 = UnitSquareMesh(10, 10)
mesh2.coordinates.dat.data[:, 0] *= 0.5
mesh2.coordinates.dat.data[:, 1] *= 0.5
mesh2.coordinates.dat.data[:, 0] += 0.1
mesh2.coordinates.dat.data[:, 1] += 0.1
T2 = mesh2.coordinates
Q2 = T2.function_space()
dT2 = Function(Q2).interpolate(dT)

T.assign(T + dT)  # don't do this before defining dT2
T2.assign(T2 + dT2)

# assemble reduced functional; optimum is disc
# centered at (0.5,0.5) with radius 0.25
x2, y2 = SpatialCoordinate(mesh2)
f2 = (x2-0.5)**2 + (y2-0.5)**2 - 0.25**2
Jpyadj = assemble(f2 * dx)
Jred = fda.ReducedFunctional(Jpyadj, fda.Control(dT))
fda.stop_annotating()

# shape optimize
F1 = VTKFile("mesh1.pvd")
F2 = VTKFile("mesh2.pvd")
opts = {"riesz_representation": "H1"}
N = 2
for ii in range(N):
    print("J(ii =", ii,") = ", Jred.__call__(dT))
    F1.write(T)
    F2.write(T2)
    gradJ = Jred.derivative(options=opts)
    dT.interpolate(-gradJ)
print("J(ii =", N, ") = ", Jred.__call__(dT))
F1.write(T)
F2.write(T2)

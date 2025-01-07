from firedrake import *

mesh1 = UnitSquareMesh(2,2)
mesh2 = UnitSquareMesh(2,2)
u1 = Function(FunctionSpace(mesh1, "CG", 1)) 
u2 = Function(FunctionSpace(mesh2, "CG", 1)) 

print("interpolate before moving")
u2.interpolate(u1)

X = SpatialCoordinate(mesh1)
Q = mesh1.coordinates.function_space()
dT1 = Function(Q).interpolate(0.26*X)
Qf = mesh2.coordinates.function_space()
dT2 = Function(Qf).interpolate(dT1)
mesh1.coordinates.assign(mesh1.coordinates + dT1)
mesh2.coordinates.assign(mesh2.coordinates + dT2)

m1 = mesh1.coordinates.vector().array()
m2 = mesh2.coordinates.vector().array()
import numpy as np
assert np.allclose(m1, m2) # verify the mesh coordinates coincide

print("interpolate after moving")
try:
  u2.interpolate(u1)
  print("interpolation successful")
except:
  print("failed to interpolate after moving, increase tolerance")
  mesh1.tolerance += 0.01
  u2.interpolate(u1)

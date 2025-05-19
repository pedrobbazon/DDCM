from firedrake import *
from firedrake.__future__ import Interpolator

mesh = UnitSquareMesh(3,3)
V = FunctionSpace(mesh, "CG", 1)
u = Function(V) # function to be evaluated on the bdry

X = SpatialCoordinate(mesh)
W = FunctionSpace(mesh, "CG", 1)
x, y = Function(W), Function(W)
DirichletBC(W, X[0], "on_boundary").apply(x)
DirichletBC(W, X[1], "on_boundary").apply(y)
points = []
for ii in range(len(x.vector().array())):
  x_ = x.vector().array()[ii]
  y_ = y.vector().array()[ii]
  if (x_**2+y_**2) > 0:
    points.append([x_, y_])
vom = VertexOnlyMesh(mesh, points)
P0DG = FunctionSpace(vom, "DG", 0)
p = Function(P0DG).interpolate(u)
I = Interpolator(TestFunction(V), P0DG)
q = assemble(I.interpolate(u))
assert norm(p-q)==0

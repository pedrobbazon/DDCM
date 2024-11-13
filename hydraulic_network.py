import firedrake as fd
from firedrake import *
from firedrake.output import VTKFile
#from firedrake.__future__ import interpolate

from ddcm_classes import *
from ddcm_utils import *


Nx = 1
Ny = 1
Lx = 1.0
Ly = 1.0
flag_quads = False
msh = fd.RectangleMesh(Nx, Ny, Lx, Ly, quadrilateral=flag_quads)

Vp = fd.FunctionSpace(msh, 'CG', degree=1)
Vq = fd.FunctionSpace(msh, 'HDiv Trace', degree=0)
VDG0 = fd.FunctionSpace(msh, 'DG', degree=0)

W = Vp * Vq

w = fd.Function(W)
p, q = fd.split(w)
v, tau = fd.TestFunctions(W)

x = fd.SpatialCoordinate(msh)
n = fd.FacetNormal(msh)
d = fd.as_vector([n[1], -n[0]])

#size_edges = fd.Function(Vq)
tst = fd.TestFunction(Vq)
size_edges = (fd.Constant(0.5)*(tst('+') + tst('-'))*fd.dS(domain=msh) + fd.Constant(1.0)*tst*fd.ds(domain=msh))
hs = fd.Function(Vq)
fd.assemble(size_edges, tensor=hs)
print(hs.dat.data[:])

a  = q('+')*fd.dot(fd.grad(v),d)*(fd.dS) + q*fd.dot(fd.grad(v),d)*(fd.ds) 
a += (q('+') + hs*fd.dot(d,fd.grad(p)))*tau*fd.dS  + (q + hs*fd.dot(d,fd.grad(p)))*tau*fd.ds

w = fd.Function(W, name='Solution')
fd.solve(a == 0, w, bcs=[], solver_parameters=petsc_opt1)


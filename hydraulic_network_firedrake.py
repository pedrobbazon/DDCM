from firedrake import *
from pyop2.datatypes import IntType

Nx = 3
Ny = 3
Lx = 1.0
Ly = 1.0
flag_quads = True
msh = RectangleMesh(Nx, Ny, Lx, Ly, quadrilateral=flag_quads)

Vp = FunctionSpace(msh, 'CG', degree=1)
Vq = FunctionSpace(msh, 'HDiv Trace', degree=0)
VDG0 = FunctionSpace(msh, 'DG', degree=0)

W = Vp * Vq

w = Function(W)
p, q = split(w)
v, tau = TestFunctions(W)

x = SpatialCoordinate(msh)
n = FacetNormal(msh)
d = as_vector([n[1], -n[0]])

def gradtan(p,d):
  return dot(grad(p)('+'),d('+'))

kappa = Constant(1.0)

tst = TestFunction(Vq)
size_edges = (Constant(0.5)*(tst('+') + tst('-'))*dS(domain=msh) + Constant(1.0)*tst*ds(domain=msh))
hs = Function(Vq.dual())
assemble(size_edges, tensor=hs)
print(hs.dat.data[:])

a = inner( q('+') , gradtan(v, d) ) * dS + inner( q('+') + gradtan(p, d), tau('+') )* dS
a += inner(q , dot(grad(v),d)) * ds + inner(q + dot(grad(p),d) , tau ) * ds

petsc_opt1 = {
    'snes_type': 'newtonls',
    'snes_monitor': None,
    'snes_converged_reason': None,
    'snes_linesearch_type': 'basic',
    'snes_max_it': 10,
    'snes_divergence_tolerance': 1e20,
    'snes_rtol': 1e-11,
    'snes_atol': 1e-10,
    'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
}

class FixValueIn(DirichletBC):
    @utils.cached_property
    def nodes(self):
        if W.mesh().mpi_comm().size > 1: raise NotImplementedError
        return np.asarray([0], dtype=IntType)

class FixValueOut(DirichletBC):
    @utils.cached_property
    def nodes(self):
        if W.mesh().mpi_comm().size > 1: raise NotImplementedError
        return np.asarray([15], dtype=IntType)

bcdl = FixValueIn(W.sub(0), Constant(2.0), 1)
bcdr = FixValueOut(W.sub(0), Constant(0.0), 2)

#w = Function(W, name='Solution')
solve(a == 0, w, bcs=[bcdl, bcdr], solver_parameters=petsc_opt1)
p = w.sub(0)
q = w.sub(1)
print(p.dat.data[:])
print(q.dat.data[:]*hs.dat.data[:])

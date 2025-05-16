import numpy as np
from firedrake import *
from pyop2.datatypes import IntType

Nx = 2
Ny = 2
Lx = 1.0
Ly = 1.0
flag_quads = True
msh = RectangleMesh(Nx, Ny, Lx, Ly, quadrilateral=flag_quads)

Vp = FunctionSpace(msh, 'CG', degree=1)
Vq = FunctionSpace(msh, 'HDiv Trace', degree=0)

W = Vp * Vq

w = Function(W)
p, q = split(w)
v, tau = TestFunctions(W)

x = SpatialCoordinate(msh)
n = FacetNormal(msh)
d = as_vector([n[1], -n[0]])

def gradtan(p,d):
  return dot(grad(p)('+'),d('+'))

tst = TestFunction(Vq)
size_edges = (Constant(0.5)*(tst('+') + tst('-'))*dS(domain=msh) + Constant(1.0)*tst*ds(domain=msh))
hs = Function(Vq.dual())
assemble(size_edges, tensor=hs)
print(hs.dat.data[:])


def kappa(q):
  return Constant(1.0)

a = inner( q('+') , gradtan(v, d) ) * dS + inner( q('+') + kappa(q)('+')*gradtan(p, d), tau('+') )* dS
a += inner(q , dot(grad(v),d)) * ds + inner(q + kappa(q)*dot(grad(p),d) , tau ) * ds

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
        return np.asarray([8], dtype=IntType)

bcdl = FixValueIn(W.sub(0), Constant(2.0), 1)
bcdr = FixValueOut(W.sub(0), Constant(0.0), 2)

#w = Function(W, name='Solution')
solve(a == 0, w, bcs=[bcdl, bcdr], solver_parameters=petsc_opt1)
p = w.sub(0)
q = w.sub(1)
print(p.dat.data[:])
print(q.dat.data[:])
VTKFile('pressure.pvd').write(w.sub(0))



#-------------------------------------------------
# Parameters
D  = Constant(1.0)
rho = Constant(1.2)
mu = Constant(1.0)
A = Constant(0.25*np.pi*D**2)
af = Constant(0.3164)
b = Constant(0.25)

Nx = 3
Ny = 3
Lx = 1.0
Ly = 1.0
flag_quads = True
msh = RectangleMesh(Nx, Ny, Lx, Ly, quadrilateral=flag_quads)

Vp = FunctionSpace(msh, 'CG', degree=1)
Vq = FunctionSpace(msh, 'HDiv Trace', degree=0)

W = Vp * Vq

w = Function(W)
wold = Function(W)
p, q = split(w)
pold, qold = split(wold)
#qold = Function(Vq)
v, tau = TestFunctions(W)

x = SpatialCoordinate(msh)
n = FacetNormal(msh)
d = as_vector([n[1], -n[0]])

tst = TestFunction(Vq)
size_edges = (Constant(0.5)*(tst('+') + tst('-'))*dS(domain=msh) + Constant(1.0)*tst*ds(domain=msh))
hs = Function(Vq.dual())
assemble(size_edges, tensor=hs)
print(hs.dat.data[:])

def gradtanplus(p,d):
  return dot(grad(p)('+'), d('+'))

def gradtan(p,d):
  return dot(grad(p), d)

def FluxLawLin(q, dpds):
  return q + dpds

eps = Constant(0.0)
def FluxLawLam(q, dpds):
  ubar = q
  Re = rho*ubar*D/mu
  f = Constant(64) / (Re + eps)
  return 0.5*rho * ubar * ubar * f / D + dpds

def FluxLaw(q, qold, dpds):
  ubar = q / A
  ubarold = qold / A
  Re = rho*ubarold*D/mu
  f = af * (Re)**(-b)
  return  f * (0.5*rho*ubar*ubarold)/D + dpds

dpdsplus = gradtanplus(p,d)
dvdsplus = gradtanplus(v, d)
dpds =  gradtan(p,d)
dvds = gradtan(v,d)

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
        return np.asarray([8], dtype=IntType)

bcdl = FixValueIn(W.sub(0), Constant(0.2), 1)
bcdr = FixValueOut(W.sub(0), Constant(0.0), 2)

# Integrate over internal and external facets
a  = inner(q('+'), dvdsplus)*dS + inner(FluxLawLin(q('+'), dpdsplus), tau('+'))*dS
a += inner(q, dvds)*ds          + inner(FluxLawLin(q,      dpds),     tau)*ds
solve(a == 0, w, bcs=[bcdl, bcdr], solver_parameters=petsc_opt1)


# Integrate over internal and external facets
a  = inner(q('+'), dvdsplus)*dS + inner(FluxLawLam(q('+'), dpdsplus), tau('+'))*dS
a += inner(q, dvds)*ds          + inner(FluxLawLam(q,      dpds),     tau)*ds
solve(a == 0, w, bcs=[bcdl, bcdr], solver_parameters=petsc_opt1)

print(w.sub(0).dat.data[:])
print(w.sub(1).dat.data[:])

#qold.assign(w.sub(1))
wold.assign(w)
qoldk = Function(Vq)
qoldk.assign(w.sub(1))
for k in range(10):

  dpdsplus = gradtanplus(p, d)
  dvdsplus = gradtanplus(v, d)
  dpds =  gradtan(p, d)
  dvds = gradtan(v, d)

  a = inner(q('+'), dvdsplus)*dS + inner(FluxLaw(q('+'), qoldk('+'), dpdsplus), tau('+'))*dS
  a += inner(q, dvds)*ds         + inner(FluxLaw(q, qoldk, dpds), tau)*ds

  print(qoldk.dat.data)
  solve(a == 0, w, bcs=[bcdl, bcdr], solver_parameters=petsc_opt1)
  wold.assign(w)
  qoldk.assign(w.sub(1))
  
  #qold.assign(w.sub(1))

print(w.sub(0).dat.data[:])
print(w.sub(1).dat.data[:])

p = w.sub(0)
q = w.sub(1)
print(p.dat.data[:])
print(q.dat.data[:])
VTKFile('pressure.pvd').write(w.sub(0))
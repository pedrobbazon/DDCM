import firedrake
import numpy as np
from petsc4py import PETSc

from firedrake import *
from pyop2.datatypes import IntType

# Casos nao lineares


# Parameters
D  = Constant(0.02)
rho = Constant(1000.0)
mu = Constant(0.001)
A = Constant(0.25*np.pi*D**2)
af = Constant(0.3164)
b = Constant(0.25)
oneatm = 13600*9.8*0.76
pin = 5000.5
pout = 0.0*oneatm

Nx = 3
Ny = 3
Lx = 2.0
Ly = 2.0
flag_quads = True
msh = RectangleMesh(Nx, Ny, Lx, Ly, quadrilateral=flag_quads)

Vp = FunctionSpace(msh, 'CG', degree=1)
Vq = FunctionSpace(msh, 'HDiv Trace', degree=0)

W = Vp * Vq

w = Function(W)
wold = Function(W)
p, q = split(w)
qold = Function(Vq)
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
  ubar = q / A
  return 32.0*mu*ubar/D + dpds

eps = Constant(0.0)
def FluxLawLam(q, dpds):
  ubar = q / A
  Re = rho*ubar*D/mu
  f = Constant(64) / (Re+eps)
  return 0.5*rho * ubar * ubar * f / D + dpds

def FluxLaw(q, qold, dpds):
  ubar = q / A
  ubarold = qold / A
  Re = rho*sqrt(ubar**2)*D/mu
  f = af * (Re)**(-b)
  return  f * (0.5*rho*ubar*ubar)/D + dpds

def regularized_power(x, b, eps=1e-6):
    return (x**2 + eps**2)**(b/2)

def FluxLawReg(q, qold, dpds):
  ubar = q / A
  ubarold = qold / A
  Re = rho*ubar*D/mu
  Re_reg = regularized_power(Re, b, eps=Constant(1e-6))
  f = af / Re_reg
  return  sign(ubar) * f * (0.5*rho*ubar*ubar)/D + dpds

'''
def log10(x):
    return ln(x) / ln(as_ufl(10.0))

def FluxLawReg(q, qold, dpds):
  ubar = q / A
  ubarold = qold / A
  eta = Constant(1.0e-5)
  modubarreg = eta*(sqrt(1.0 + ubarold*ubarold/eta**2) - 1.0)
  Re = rho*modubarreg*D/mu
  f = 0.25 / (log10(5.74/Re**0.9))**2
  return  f * (0.5*rho*ubarold*ubar)/D + dpds
'''

dpdsplus = gradtanplus(p,d)
dvdsplus = gradtanplus(v, d)
dpds =  gradtan(p,d)
dvds = gradtan(v,d)

petsc_opt1 = {
    'snes_type': 'newtonls',
    'snes_monitor': None,
    'snes_converged_reason': None,
    'snes_linesearch_type': 'basic',
    'snes_max_it': 20,
    'snes_divergence_tolerance': 1e40,
    'snes_rtol': 1e-20,
    'snes_atol': 1e-20,
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

bcdl = FixValueIn(W.sub(0), Constant(pin), 1)
bcdr = FixValueOut(W.sub(0), Constant(pout), 2)

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
print('oooooooooooooooooooooooo')

qold.assign(w.sub(1))
for k in range(2):

  print('---------------------------')
  print('iteration', k)
  dpdsplus = gradtanplus(p, d)
  dvdsplus = gradtanplus(v, d)
  dpds =  gradtan(p, d)
  dvds = gradtan(v, d)

  a = inner(q('+'), dvdsplus)*dS + inner(FluxLawReg(q('+'), qold('+'), dpdsplus), tau('+'))*dS
  a += inner(q, dvds)*ds         + inner(FluxLawReg(q, qold, dpds), tau)*ds

  solve(a == 0, w, bcs=[bcdl, bcdr], solver_parameters=petsc_opt1)

  error = np.linalg.norm(w.sub(1).dat.data[:] - wold.sub(1).dat.data[:])
  print('error', error)

  wold.assign(w)
  qold.assign(w.sub(1))
  #print('pressure', w.sub(0).dat.data[:])
  print('flux', (w.sub(1).dat.data[:]))

p = w.sub(0)
q = w.sub(1)
VTKFile('pressure.pvd').write(w.sub(0))


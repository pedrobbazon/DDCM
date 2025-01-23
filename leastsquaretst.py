import firedrake as fd
from firedrake import *
from firedrake.output import VTKFile

petsc_opt1 = {
    'snes_type': 'newtonls',
    'snes_monitor': None,
    'snes_converged_reason': None,
    'snes_linesearch_type': 'basic',
    'snes_max_it': 100,
    'snes_divergence_tolerance': 1e20,
    'snes_rtol': 1e-9,
    'snes_atol': 1e-11,
    'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
}

Nx, Ny = 40, 40
Lx, Ly = 1.0, 1.0
msh = fd.RectangleMesh(Nx, Ny, Lx, Ly, originX=0.0, originY=0.0, quadrilateral=False)

# Spaces
CG  = fd.FunctionSpace(msh, "CG", degree=1)
Hdiv  = fd.FunctionSpace(msh, family="RT", degree=2)
#VCG  = fd.VectorFunctionSpace(msh, family="CG", degree=2)
W = Hdiv * CG
#W = VCG * CG

# Unknown field
w = fd.Function(W)
s, u = fd.split(w)

# Test function
vw = fd.TestFunction(W)

zeda = fd.Constant(1.0)
q = fd.Constant(1.0)
J = 0.5*((fd.div(s) + zeda*u - q)**2) * fd.dx + fd.Constant(1.0e-3) * (fd.inner(s+fd.grad(u),s+fd.grad(u))) * fd.dx

 # Optimallity conditions
dJ = fd.derivative(J, w, vw)

# Jacobian
Jac = fd.derivative(dJ, w)

bc_lst = [fd.DirichletBC(W.sub(1), fd.Constant(0.0), "on_boundary")]

# Non-linear solve (see solver parameters above)
fd.solve(dJ == 0, w, bcs=bc_lst, J=Jac, solver_parameters=petsc_opt1)

filename = './s.pvd'
w.sub(0).rename('Flux')
VTKFile(filename).write(w.sub(0))

filename = './u.pvd'
w.sub(1).rename('Potential')
VTKFile(filename).write(w.sub(1))
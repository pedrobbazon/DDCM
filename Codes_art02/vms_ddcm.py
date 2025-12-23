# Import firedrake
import firedrake as fd

# Formulation
formulation = 'dual'
projector = 'OSGS'

# ===== GEOMETRY AND SETTINGS =====

# Geometry
Lx, Ly = 1.0, 1.0
Nx, Ny = 200, 200
mesh = fd.RectangleMesh(Nx, Ny, Lx, Ly, originX=0.0, originY=0.0, quadrilateral=False)
h = fd.CellDiameter(mesh)
x = fd.SpatialCoordinate(mesh)

# Constants
zeta  = fd.Constant(1.0)   # Reaction coefficient
gamma = fd.Constant(1.0)   # Conductivity
kappa = fd.Constant(1.0)

# Stabilization parameters
if formulation == 'primal':
    ku = fd.Constant(1.0)
    kl = fd.Constant(1.0)
    ke = fd.Constant(0.125)
    ks = fd.Constant(0.125)
    km = fd.Constant(0.125)
    L0 = h
    L1 = 1/h
elif formulation == 'dual':
    ku = fd.Constant(0.25)
    kl = fd.Constant(0.25)
    ke = fd.Constant(0.25)
    ks = fd.Constant(1.0)
    km = fd.Constant(0.25)
    L0 = fd.Constant(1.0)
    L1 = fd.Constant(1.0)
else:
    raise Exception('Unknown formulation')

def etilde(x):
    etx = fd.sin(fd.pi*x[0])*fd.cos(fd.pi*x[1])
    ety = fd.cos(fd.pi*x[0])*fd.sin(fd.pi*x[1])
    return fd.pi*fd.as_vector([etx, ety])

def stilde(x):
    return - gamma * etilde(x)

q = fd.Constant(1.0)

# ===== VARIATIONAL MULTISCALE FORMULATION =====

# Construct the appropriate mixed function space
if projector == 'ASGS':
    # Scalar and vector function spaces
    SF = fd.FunctionSpace(mesh, family="CG", degree=1)
    VF = fd.VectorFunctionSpace(mesh, family="CG", degree=1)
    # Mixed space
    W = fd.MixedFunctionSpace([VF, SF, VF, SF, VF])
elif projector == 'OSGS':
    # Scalar and vector function spaces
    SF = fd.FunctionSpace(mesh, family="CG", degree=1)
    VF = fd.VectorFunctionSpace(mesh, family="CG", degree=1)
    VscS = fd.FunctionSpace(mesh, family="CG", degree=1)
    VscV = fd.VectorFunctionSpace(mesh, family="CG", degree=1)
    # Mixed space
    W = fd.MixedFunctionSpace([VF, SF, VF, SF, VF, VscV, VscS, VscV, VscS, VscV])
else:
    raise ValueError('Unknown projection operator')

# Unknown fields and test functions
w = fd.Function(W)
if projector == 'ASGS':
    s, u, e, lam, mu = fd.split(w)
    del_s, del_u, del_e, del_lam, del_mu = fd.TestFunctions(W)
elif projector == 'OSGS':
    s, u, e, lam, mu, PL2res_s, PL2res_u, PL2res_e, PL2res_lam, PL2res_mu = fd.split(w)
    del_s, del_u, del_e, del_lam, del_mu, del_sc_s, del_sc_u, del_sc_e, del_sc_lam, del_sc_mu = fd.TestFunctions(W)
else:
    raise Exception('Unknown projection operator')

# Linear form (standard terms)
L = (fd.inner(etilde(x), del_e) + kappa * fd.inner(stilde(x), del_s)) * fd.dx

# Set the stabilization parameters
if formulation == 'primal':
    tau_u   = ku * L0 ** 2
    tau_e   = ke
    tau_s   = ks / kappa
    tau_lam = - kl / (zeta * L1) ** 2
    tau_mu  = - km
elif formulation == 'dual':
    tau_u   = ku * L0 ** 2
    tau_e   = ke
    tau_s   = (ks / kappa) * (h ** 2)
    tau_lam = - kl / (zeta * L1) ** 2
    tau_mu  = - km * (h ** 2)
else:
    raise Exception('Unknown formulation')

# Source term
L += q * del_lam * fd.dx

if formulation == 'primal':
    # Bilinear form (Standard Galerkin terms - primal)
    B = (zeta * fd.inner(lam, del_u) - fd.inner(mu, fd.grad(del_u))) * fd.dx
    B += (fd.inner(e, del_e) + fd.inner(mu, del_e)) * fd.dx
    B += (kappa * fd.inner(s, del_s) - fd.inner(fd.grad(lam), del_s)) * fd.dx
    B += (zeta * fd.inner(u, del_lam) - fd.inner(s, fd.grad(del_lam))) * fd.dx
    B += (-fd.inner(fd.grad(u), del_mu) + fd.inner(e, del_mu)) * fd.dx

    # Set the boundary conditions accordingly
    bc_lst = [fd.DirichletBC(W.sub(1), fd.Constant(0.0), "on_boundary"),  # u
              fd.DirichletBC(W.sub(3), fd.Constant(0.0), "on_boundary")]  # lam

elif formulation == 'dual':
    # Bilinear form (Standard Galerkin terms - dual)
    B = (zeta * fd.inner(lam, del_u) + fd.inner(fd.div(mu), del_u)) * fd.dx
    B += (fd.inner(e, del_e) + fd.inner(mu, del_e)) * fd.dx
    B += (kappa * fd.inner(s, del_s) + fd.inner(lam, fd.div(del_s))) * fd.dx
    B += (zeta * fd.inner(u, del_lam) + fd.inner(fd.div(s), del_lam)) * fd.dx
    B += (fd.inner(u, fd.div(del_mu)) + fd.inner(e, del_mu)) * fd.dx

    # Set the boundary conditions accordingly
    bc_lst = [fd.DirichletBC(W.sub(0), fd.as_vector([0.0,0.0]), "on_boundary"), # s
              fd.DirichletBC(W.sub(4), fd.as_vector([0.0,0.0]), "on_boundary")] # mu

else:
    raise Exception('Unknown formulation')

# Residuals
res_u = - zeta * lam - fd.div(mu)
res_e = etilde(x) - e - mu
res_s = kappa * (stilde(x) - s) + fd.grad(lam)
res_lam = q - zeta * u - fd.div(s)
res_mu = fd.grad(u) - e

if projector == 'ASGS':
    sub_u = tau_u * res_u
    sub_e = tau_e * res_e
    sub_s = tau_s * res_s
    sub_lam = tau_lam * res_lam
    sub_mu = tau_mu * res_mu

elif projector == 'OSGS':

    B += (fd.inner(PL2res_u, del_sc_u) - fd.inner(res_u, del_sc_u)) * fd.dx
    B += (fd.inner(PL2res_e, del_sc_e) - fd.inner(res_e, del_sc_e)) * fd.dx
    B += (fd.inner(PL2res_s, del_sc_s) - fd.inner(res_s, del_sc_s)) * fd.dx
    B += (fd.inner(PL2res_lam, del_sc_lam) - fd.inner(res_lam, del_sc_lam)) * fd.dx
    B += (fd.inner(PL2res_mu, del_sc_mu) - fd.inner(res_mu, del_sc_mu)) * fd.dx

    sub_u = tau_u * (res_u - PL2res_u)
    sub_e = tau_e * (res_e - PL2res_e)
    sub_s = tau_s * (res_s - PL2res_s)
    sub_lam = tau_lam * (res_lam - PL2res_lam)
    sub_mu = tau_mu * (res_mu - PL2res_mu)

else:
    raise Exception('Unknown projection operator')

# Stabilization terms
B += (zeta * fd.inner(sub_lam, del_u) - fd.inner(sub_mu, fd.grad(del_u))) * fd.dx
B += (fd.inner(sub_e, del_e) + fd.inner(sub_mu, del_e)) * fd.dx
B += (kappa * fd.inner(sub_s, del_s) + fd.inner(sub_lam, fd.div(del_s))) * fd.dx
B += (zeta * fd.inner(sub_u, del_lam) - fd.inner(sub_s, fd.grad(del_lam))) * fd.dx
B += (fd.inner(sub_u, fd.div(del_mu)) + fd.inner(sub_e, del_mu)) * fd.dx

# Total residual (LHS - RHS)
Res = B - L

# Jacobian
Jac = fd.derivative(Res, w)

# Solve
fd.solve(Res == 0, w, bcs=bc_lst, J=Jac)

# ===== SAVE FOR VISUALIZATION =====

# Save results
s, u, e, lam, mu, *_ = w.subfunctions
s.rename('DDCM.s')
u.rename('DDCM.u')
e.rename('DDCM.e')
lam.rename('DDCM.lam')
mu.rename('DDCM.mu')
output = fd.VTKFile('output.pvd')
output.write(s, u, e, lam, mu)




























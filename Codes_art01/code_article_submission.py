# Import firedrake
import firedrake as fd

# Geometry  
Lx, Ly = 1.0, 1.0
Nx, Ny = 100, 100
mesh = fd.RectangleMesh(Nx, Ny, Lx, Ly, originX=0.0, originY=0.0, quadrilateral=False)
h = fd.CellDiameter(mesh)
x = fd.SpatialCoordinate(mesh)

# Parameters and Manufacuted fields  
pars = {'zeta':  fd.Constant(1.0), 
        'kappa': fd.Constant(1.0),
        'alpha': fd.Constant(0.125),   
        'gamma': fd.Constant(0.125),
        'eta':   fd.Constant(0.5), 
        'beta':  fd.Constant(0.5), 'lmu': h,
        'theta': fd.Constant(0.5), 'ls': h,
        'formulation': 'Equal-Order'
    }
 
def etilde(x):
    etx = fd.cos(fd.pi*x[0])*fd.sin(fd.pi*x[1])
    ety = fd.sin(fd.pi*x[0])*fd.cos(fd.pi*x[1])
    return fd.pi*fd.as_vector([etx, ety])

def stilde(x):
    kappa = pars['kappa']
    stx = fd.cos(fd.pi*x[0])*fd.sin(fd.pi*x[1])
    sty = fd.sin(fd.pi*x[0])*fd.cos(fd.pi*x[1])
    return -kappa * fd.pi*fd.as_vector([stx, sty])

def q(x):
    source = fd.Constant(0.0)
    return source

# Function spaces
formulation = pars['formulation']
if(formulation == 'Natural'):
    SF = fd.FunctionSpace(mesh, family='CG', degree=1)
    VF =  fd.VectorFunctionSpace(mesh, family='DG', degree=0) 
elif(formulation == 'Equal-Order'):
    SF = fd.FunctionSpace(mesh, family='CG', degree=1)
    VF = fd.VectorFunctionSpace(mesh, family='CG', degree=1)

# Mixed space (s  u  e  lam  mu)
W = fd.MixedFunctionSpace([VF, SF, VF, SF, VF])

# Unknown field
w = fd.Function(W)
s, u, e, lam, mu = fd.split(w)

kap = pars['kappa'] # assert units in objective function
# Proximity function
dgrad, dflux = e-etilde(x), s-stilde(x)
L = 0.5*fd.inner(dgrad,dgrad)*fd.dx + 0.5*kap*fd.inner(dflux,dflux)*fd.dx

# Balance and Compatibility constraint
L -= (fd.inner(s, fd.grad(lam)) + fd.inner(pars['zeta']*u - q(x), lam))*fd.dx
L += fd.inner(e - fd.grad(u), mu)*fd.dx

# Stabilization terms
if(pars['formulation'] == 'Equal-Order'): 
    CT = e - fd.grad(u)
    L += 0.5*pars['alpha']*fd.inner(CT,CT)*fd.dx

    ET = e + mu - etilde(x)
    L -= 0.5*pars['gamma']*fd.inner(ET,ET)*fd.dx

    ST = kap*(s-stilde(x)) - fd.grad(lam)
    L -= (0.5*pars['eta']/kap)*fd.inner(ST,ST)*fd.dx

    BT = fd.div(s) + pars['zeta']*u - q(x)
    L += (0.5*pars['theta']*kap*pars['ls']**2)*fd.inner(BT,BT)*fd.dx

    MT = fd.div(mu) + pars['zeta']*lam
    L -= (0.5*pars['beta']*pars['lmu']**2)*fd.inner(MT,MT)*fd.dx

# Test function
dw = fd.TestFunction(W) 
# Optimallity conditions
dL = fd.derivative(L, w, dw)
# Jacobian
Jac = fd.derivative(dL, w)
# solve problem
bclist = [fd.DirichletBC(W.sub(1), fd.Constant(0.0), 'on_boundary'), 
        fd.DirichletBC(W.sub(3), fd.Constant(0.0), 'on_boundary')]
fd.solve(dL == 0, w, bcs=bclist, J=Jac)

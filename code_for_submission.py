#-----------------------------------
# General libraries
#-----------------------------------
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import firedrake as fd
from firedrake.output import VTKFile
from firedrake.__future__ import interpolate

# ---------------------------------------
#  Auxiliary classes
# ---------------------------------------

class PoissonProblem:
    
    def __init__(self, mesh, exact_fields, orders):
        self.mesh = mesh
        self.exact_fields = exact_fields
        self.orders = orders
        self.fields = None
    
    #------------------------------------------------------------
    # Private methods
    def _poisson_analytical_solver(self):
        # Function spaces
        order_s, order_u = self.orders
        U = fd.FunctionSpace(self.mesh, 'CG', order_u)
        V = fd.VectorFunctionSpace(self.mesh, 'CG', order_s)
        
        # Interpolate fields
        u = fd.assemble(interpolate(self.exact_fields['exact-potential'], U))
        s = fd.assemble(interpolate(self.exact_fields['exact-flux'], V))
        e = fd.assemble(interpolate(self.exact_fields['exact-grad'], V))
        lmbd = fd.assemble(interpolate(self.exact_fields['exact-lmbd'], U))
        mu = fd.assemble(interpolate(self.exact_fields['exact-mu'], V))
        et = fd.assemble(interpolate(self.exact_fields['exact-etilde'], V))
        st = fd.assemble(interpolate(self.exact_fields['exact-stilde'], V))

        return u, s, e, lmbd, mu, et, st

    def _poisson_results(self, field, name, dir_outs, mode, tt):
        # Create directory
        filename = os.path.join(dir_outs, f"{name}.pvd")
        field.rename(f"Reference.{name}")
        VTKFile(filename, mode=mode).write(field, time=tt)
    #------------------------------------------------------------
    
    #------------------------------------------------------------
    # Instance methods 
    def solve_poisson(self):
        # Solve
        self.fields = self._poisson_analytical_solver()        

    def save_poisson(self, tt=0.0, dir_outs='myrefsol'):
        if self.fields is None:
            self.solve()
        if dir_outs is not None and not os.path.exists(dir_outs):
            os.makedirs(dir_outs)
            
        # Save
        mode = 'w' if tt == 0.0 else 'a'
        for field, name in zip(self.fields, ['u', 's', 'e', 'l', 'm', 'et', 'st']):
            self._poisson_results(field, name, dir_outs, mode, tt)
    #------------------------------------------------------------
  
  
class DDCMProblem():
    
    def __init__(self, mesh, pars, exact_fields, orders):
        self.mesh = mesh
        self.pars = pars
        self.exact_fields = exact_fields
        self.orders = orders
        self._ddcm_data()
            
    #------------------------------------------------------------
    # Private methods    
    def _ddcm_data(self):
        self.sdata = self.exact_fields['exact-stilde']
        self.edata = self.exact_fields['exact-etilde']
        
    def _ddcm_objective_function(self, e, s):
        kappa = self.pars['kappa']
        J =  0.5*fd.inner(e-self.edata, e-self.edata)*fd.dx
        J += 0.5*kappa*fd.inner(s-self.sdata, s-self.sdata)*fd.dx
        return  J
            
    def _ddcm_solver(self):
        
        petsc_opt1 = {
        'snes_type': 'newtonls',
        'snes_monitor': None,
        'snes_converged_reason': None,
        'snes_linesearch_type': 'basic',
        'snes_max_it': 2,
        'snes_divergence_tolerance': 1e20,
        'snes_rtol': 1e-3,
        'snes_atol': 1e-3,
        'mat_type': 'aij',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps'
        }
        
        # Mesh coordinates and normal unit outward vector
        x = fd.SpatialCoordinate(self.mesh)
        n = fd.FacetNormal(self.mesh)
        
        # Function spaces
        order_s, order_u = self.orders
        formulation = self.pars['formulation']
        if(formulation == 'Natural'):
            U = fd.FunctionSpace(self.mesh, family="CG", degree=order_u)
            V =  fd.VectorFunctionSpace(self.mesh, family='DG', degree=order_s)
            
        elif(formulation == 'Equal-Order'):
            U = fd.FunctionSpace(self.mesh, family="CG", degree=order_u)
            V = fd.VectorFunctionSpace(self.mesh, family="CG", degree=order_s)
        else:
            raise ValueError('Invalid formulation.')

        # Mixed space (s  u  e  lam  mu)
        W = fd.MixedFunctionSpace([V, U, V, U, V])
        
        # Unknown field
        w = fd.Function(W)
        s, u, e, lam, mu = fd.split(w)

        # Test function
        vw = fd.TestFunction(W)

        # Functional to be minimized
        J = self._ddcm_objective_function(e, s)
        
        # Stabilization paramters
        zeda  = self.pars['zeda']
        theta = self.pars['theta']
        kappa = self.pars['kappa']
        alpha = self.pars['alpha']
        beta  = self.pars['beta']
        gamma = self.pars['gamma']
        eta  = self.pars['eta']
        ls    = self.pars['ls']
        lmu   = self.pars['lmu']
        
        # Right-hand side fields and boundary values
        q = self.exact_fields['exact-source']
        f = self.exact_fields['exact-dualsource']
        u_bc = self.exact_fields['exact-potential']
        lmbd_bc = self.exact_fields['exact-lmbd']

        # Build the Lagrangian:
        
        # Proximity function
        L  = J

        # Balance and Compatibility constraint
        L -= fd.inner(s, fd.grad(lam)) * fd.dx
        L += fd.inner(zeda*u - q, lam) * fd.dx
        L += fd.inner(e - fd.grad(u), mu) * fd.dx
        
        # Stabilization terms
        resC = e - fd.grad(u)
        L += 0.5*alpha*fd.inner(resC,resC) * fd.dx

        resE = e + mu - self.edata
        L  -= 0.5*gamma*fd.inner(resE,resE) * fd.dx

        resK = kappa*s - fd.grad(lam) - kappa*self.sdata
        L  -= (0.5*eta/kappa)*fd.inner(resK,resK) * fd.dx

        resB = fd.div(s) + zeda*u - q
        L += (0.5*theta*kappa*ls**2)*fd.inner(resB,resB) * fd.dx

        resM = fd.div(mu) + zeda*lam - f
        L -= (0.5*beta*lmu**2)*fd.inner(resM,resM) * fd.dx
        
        L -= u*f*fd.dx
        
        # Boundary conditions
        bc_lst = []
        bc_lst.append(fd.DirichletBC(W.sub(1), u_bc, "on_boundary"))
        bc_lst.append(fd.DirichletBC(W.sub(3), lmbd_bc, "on_boundary"))
        
        # Optimallity conditions
        dL = fd.derivative(L, w, vw)

        # Jacobian
        Jac = fd.derivative(dL, w)

        # Non-linear solver
        fd.solve(dL == 0, w, bcs=bc_lst, J=Jac, solver_parameters=petsc_opt1)
                
        return w.sub(0), w.sub(1), w.sub(2), w.sub(3), w.sub(4)
    
    def _ddcm_results(self, field, name, dir_outs, mode, tt):
        # Create directory
        filename = os.path.join(dir_outs, f"{name}.pvd")
        field.rename(f"DDCM.{name}")
        VTKFile(filename, mode=mode).write(field, time=tt)
    #------------------------------------------------------------
    
    #------------------------------------------------------------
    # Instance methods 
    def solve_ddcm(self):
        # Solve
        self.fields = self._ddcm_solver()        

    def save_ddcm(self, tt=0.0, dir_outs='myddcmsol'):
        if self.fields is None:
            self.solve()
        if dir_outs is not None and not os.path.exists(dir_outs):
            os.makedirs(dir_outs)
            
        # Save
        mode = 'w' if tt == 0.0 else 'a'
        for field, name in zip(self.fields, ['s', 'u', 'e', 'l', 'm']):
            self._ddcm_results(field, name, dir_outs, mode, tt)
    #------------------------------------------------------------


# === MAIN === #

# ---------------------------------------
# Geometry  
# ---------------------------------------
Lx, Ly = 1.0, 1.0
Nx, Ny = 100, 100
mesh = fd.RectangleMesh(Nx, Ny, Lx, Ly, originX=0.0, originY=0.0, quadrilateral=False)
h = fd.CellDiameter(mesh)
x = fd.SpatialCoordinate(mesh)

# ---------------------------------------
# Parameters and Manufacuted fields  
# ---------------------------------------
pars = {'zeda':  fd.Constant(1.0), 
        'kappa': fd.Constant(1.0),
        'alpha': fd.Constant(0.125),   
        'gamma': fd.Constant(0.125),
        'eta':   fd.Constant(0.5), 
        'beta':  fd.Constant(0.5), 'lmu': h,
        'theta': fd.Constant(0.5), 'ls': h,
        'formulation': 'Natural'
    }

# Potential field 
def uexact(x):
    return fd.cos(fd.pi*x[0])*fd.cos(fd.pi*x[1]) 

# Gradient field
def eexact(x):
    return fd.grad(uexact(x))

# Scalar lagrangian multiplier
def lmbdexact(x): 
    A = fd.Constant(0.05)    # Increase A slightly for more variation
    ome = fd.Constant(2.0)   # Frequency
    phi = fd.Constant(0.0)   # Phase angle 
    return  - (A / (ome * fd.pi)) * (
        fd.cos(ome * fd.pi * x[0] + phi) + fd.cos(ome * fd.pi * x[1] + phi)
    )
    
# Propose a flux-gradient relationship based on the dataset  
def etilde(x):
    A = fd.Constant(0.05)        # Amplitude
    ome = fd.Constant(4.0)       # Frequency
    phi = fd.Constant(0.0)       # Phase angle 
    return fd.as_vector([eexact(x)[0] + A*fd.sin(ome*fd.pi*x[0] + phi),
                            eexact(x)[1] + A*fd.sin(ome*fd.pi*x[1] + phi)])
def sexact(x):
    A = fd.Constant(0.025)  # Amplitude
    eps = fd.Constant(1.0e-10)
    met = fd.sqrt(eexact(x)[0]**2 + eexact(x)[1]**2) + eps
    etn = fd.as_vector([eexact(x)[0]/met, eexact(x)[1]/met])
    mst = met - A*met**3
    return fd.as_vector([-mst*etn[0], -mst*etn[1]])

# by eq.(14)
def muexact(x):
    return etilde(x) - eexact(x)
# by eq.(13)
def fdual(x):
    zeda = pars['zeda']
    return zeda*lmbdexact(x) + fd.div(muexact(x))
# by eq.(15)
def stilde(x):
    kappa = pars['kappa']
    return sexact(x) - (1/kappa)*fd.grad(lmbdexact(x))
# by eq.(16)
def qexact(x):
    zeda = pars['zeda']
    return zeda*uexact(x) + fd.div(sexact(x))

# Organize them in a simple dictionary
exact_fields = {'exact-potential': uexact(x), 
                'exact-flux': sexact(x), 
                'exact-grad': eexact(x), 
                'exact-lmbd' : lmbdexact(x), 
                'exact-mu': muexact(x), 
                'exact-etilde': etilde(x), 
                'exact-stilde': stilde(x),
                'exact-dualsource': fdual(x),
                'exact-source': qexact(x)
            }

# ---------------------------------------
# Reference problem  
# ---------------------------------------
print('Solving the reference problem')
poisson = PoissonProblem(mesh, exact_fields, orders=[3,3])
poisson.solve_poisson()
poisson.save_poisson(tt=0.0, dir_outs='test_mypoissonrefsol')

# ---------------------------------------
# DDCM problem
# ---------------------------------------
print('Solving the DDCM problem')
ddcm = DDCMProblem(mesh, pars, exact_fields, orders=[0,1])
ddcm.solve_ddcm()
ddcm.save_ddcm(tt=0.0, dir_outs='test_myddcmsol')

# ---------------------------------------
# Complete
# ---------------------------------------
print('Done!')

'''
def ddcm_results(field, name, dir_outs, mode, tt):
    # Create directory
    filename = os.path.join(dir_outs, f"{name}.pvd")
    field.rename(f"DDCM.{name}")
    VTKFile(filename, mode=mode).write(field, time=tt)


tt=0.0
dir_outs = 'new_myddcmsol'
fields = w.sub(0), w.sub(1), w.sub(2), w.sub(3), w.sub(4)
mode = 'w' if tt == 0.0 else 'a'
for field, name in zip(fields, ['s', 'u', 'e', 'l', 'm']):
    ddcm_results(field, name, dir_outs, mode, tt)
#------------------------------------------------------------
'''






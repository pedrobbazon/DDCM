import numpy as np
import matplotlib.pyplot as plt
import firedrake as fd
from firedrake.output import VTKFile

# Definir a malha para o domínio retangular 
mesh = fd.RectangleMesh(nx = 100, ny = 100, Lx = 1.0, Ly = 1.0, originX = 0.0, originY = 0.0) 
x = fd.SpatialCoordinate(mesh)

# Definir os espaços de funções
V = fd.FunctionSpace(mesh,'CG',1)
W = fd.VectorFunctionSpace(mesh, "CG", 1) 

# def uexact(x):
#     return fd.cos(fd.pi*x[0])*fd.cos(fd.pi*x[1]) 

# def eexact(x):
#     return fd.grad(uexact(x))

# def etilde(x):
#     A = fd.Constant(0.2)        # amplitude
#     f = fd.Constant(1.0)        # frequencia
#     phi = fd.Constant(0.0)      # angulo de fase
#     return fd.as_vector([eexact(x)[0] + A*fd.sin(f*fd.pi*eexact(x)[0] + phi),
#                          eexact(x)[1] + A*fd.sin(f*fd.pi*eexact(x)[1] + phi)])
    
# def stilde(x):
#     A = fd.Constant(0.2)    # amplitude
#     f = fd.Constant(1.0)    # frequencia
#     phi = fd.Constant(0.0)  # angulo de fase
#     return fd.as_vector([-etilde(x)[0] + A*fd.sin(f*fd.pi*etilde(x)[0] + phi),
#                          -etilde(x)[1] + A*fd.sin(f*fd.pi*etilde(x)[1] + phi)])


params = {'perm': fd.Constant(1.0), 'visc' : fd.Constant(1.0),
            'forch_coef': fd.Constant(1.0), 'zeda': fd.Constant(1.0), 
            'theta': fd.Constant(0.5), 'kappa': fd.Constant(1.0), 
            'alpha': fd.Constant(0.125), 'beta': fd.Constant(0.5), 
            'gamma': fd.Constant(0.125), 'eta': fd.Constant(0.5)} 

def uexact(x):
    forch_coef = params['forch_coef']
    visc = params['visc']
    perm = params['perm']
    return - visc/(2*perm)*(x[0]**2 + x[1]**2) - (forch_coef/3)*(x[0]**2 + x[1]**2)**(3/2) 

def eexact(x):
    forch_coef = params['forch_coef']
    visc = params['visc']
    perm = params['perm']
    return fd.as_vector([-forch_coef*x[0]*fd.sqrt(x[0]**2 + x[1]**2) - visc*x[0]/perm,
                            -forch_coef*x[1]*fd.sqrt(x[0]**2 + x[1]**2) - visc*x[1]/perm])

# Propose an analytical expression for lambda
def lmbdexact(x):
    x0, y0  = fd.Constant(0.5), fd.Constant(0.5)
    A = fd.Constant(1.0)
    return fd.exp(A*((x[0]-x0)**2 + (x[1]-y0)**2))

# def sexact(x):
    # return fd.as_vector([x[0], x[1]])

# Propose a flux-gradient relationship based on the dataset
sref = lambda x: fd.as_vector([x[0], x[1]])      
def stilde(x):
    A = fd.Constant(0.1)    # Amplitude
    f = fd.Constant(1.0)     # Frequency
    phi = fd.Constant(0.0)   # Phase angle 
    return fd.as_vector([sref(x)[0] + A*fd.sin(f*fd.pi*sref(x)[0] + phi),
                         sref(x)[1] + A*fd.sin(f*fd.pi*sref(x)[1] + phi)])  
def etilde(x):
    A = fd.Constant(0.1)        # Amplitude
    f = fd.Constant(1.0)         # Frequency
    phi = fd.Constant(0.0)       # Phase angle 
    visc = params['visc']        # Viscosity
    perm = params['perm']        # Permeability
    forch_coef = params['forch_coef']   # Forcheimer's coefficient 
    # Regularization of (s*s)^(1/2)
    epsilon = 1e-2
    modfluxreg = epsilon*(fd.sqrt(1.0 + fd.inner(stilde(x),stilde(x))/epsilon**2) - 1.0)
    
    return fd.as_vector([-visc*perm*stilde(x)[0] - forch_coef*modfluxreg*stilde(x)[0] + A*fd.sin(f*fd.pi*x[0] + phi),
                         -visc*perm*stilde(x)[1] - forch_coef*modfluxreg*stilde(x)[1] + A*fd.sin(f*fd.pi*x[1] + phi)])


gradex = fd.Function(W, name = 'eexact')
gradex.interpolate(eexact(x))
gradtilde = fd.Function(W, name = 'etilde')
gradtilde.interpolate(etilde(x))
fluxtilde = fd.Function(W, name = 'stilde')
fluxtilde.interpolate(stilde(x))

VTKFile('gradex.pvd').write(gradex)
VTKFile('gradtilde.pvd').write(gradtilde)
VTKFile('fluxtilde.pvd').write(fluxtilde)

# Extrair ao longo de uma linha (e.g., y = 0.0)
xval = mesh.coordinates.dat.data[:]
indices = np.where(np.isclose(xval[:, 1], 0.0, atol=1e-8))

gradex_val = gradex.dat.data[indices, 0].flatten()  
sort_indices_ex = np.argsort(gradex_val)
gradex_val = gradex_val[sort_indices_ex]

gradtilde_val = gradtilde.dat.data[indices, 0].flatten()
fluxtilde_val = fluxtilde.dat.data[indices, 0].flatten()
sort_indices_tilde = np.argsort(gradtilde_val)
gradtilde_val = gradtilde_val[sort_indices_tilde]
fluxtilde_val = fluxtilde_val[sort_indices_tilde]

# Plotar
plt.figure(figsize=(8, 6))
plt.plot(gradex_val, -gradex_val, '-r')
plt.plot(gradtilde_val, -gradtilde_val, '--b')
plt.plot(gradtilde_val, fluxtilde_val, '--g')
plt.grid()
plt.show()



# def kper(etx, pert):
#   return -etx + pert*(np.sin(3*etx))

# def derkper(etx, pert):
#   return -1 + pert*(3*np.cos(3*etx))

# pert = 0.25
# etx = np.linspace(-3,3, 100)
# stx = kper(etx,pert)
# print(stx)
# dstx = derkper(etx,pert)
# plt.plot(etx,-etx,'-r', etx,stx,'-b', etx,dstx)
# plt.grid()
# plt.savefig('stx_vs_etx.png')
# plt.show()
import os; os.system('cls||clear'); os.environ["OMP_NUM_THREADS"] = "1"
import firedrake as fd
from firedrake.__future__ import interpolate
import matplotlib.pyplot as plt
import numpy as np

# Input:
P = 4 # Order
Nx, Ny = 4, 2 # Number of elements in each direction

# Define 1D mesh:
mesh_1D = fd.PeriodicIntervalMesh(ncells=Nx,length=1)

# Define 2D mesh via extrusion:
mesh_2D = fd.ExtrudedMesh(mesh=mesh_1D,layers=Ny,layer_height=1/Ny, extrusion_type='uniform')

# Define function spaces and high-order coordinates for 2D mesh:
V_2D = fd.FunctionSpace(mesh_2D,"CG",P)
W_2D = fd.VectorFunctionSpace(mesh_2D,"CG",P)
X_2D = fd.Function(W_2D).interpolate(mesh_2D.coordinates)

# Make 2D mesh high-order:
mesh_2D = fd.Mesh(X_2D)

# Define function and print max value:
y_coord = fd.Function(V_2D).interpolate(X_2D.sub(1))

print(np.max(y_coord.dat.data[:])) # Gives 1 (as expected)
print(np.max(X_2D.dat.data[:,1]))   # Gives 1 (as expected)

# Move mesh and print max value:
mesh_2D.coordinates.dat.data[:,1] = mesh_2D.coordinates.dat.data[:,1]*1.5

print(np.max(y_coord.dat.data[:])) # Gives 1 (NOT as expected. Should be 1.5, right?)
print(np.max(X_2D.dat.data[:,1])) # Gives 1.5 (as expected)

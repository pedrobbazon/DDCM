from firedrake import *
from firedrake.output import VTKFile
from firedrake.__future__ import interpolate

class StandardModel:
    def __init__(self, model):
        if (model == 'Darcy-Linear'):
            self.kappa = Constant(1.0)
        elif(model == 'Darcy-Nonlinear'):
            self.kappa0 = Constant(1.0)
        elif(model == 'Forcheimer'):
            self.kappa0 = Constant(1.0)
        else:
            print('ERROR: In standard_model()')
            print('REASON: Model not implemented')
            exit()
#
# geometry data: domain size, mesh, etc ...
class SimpleComputationalDomain:
    def __init__(self, nsd, Lsides, Ndivs):
        self.Lx = Lsides[0]
        self.Ly = Lsides[1]
        self.Lz = Lsides[2]
        # mesh discretization
        self.Nx = Ndivs[0]
        self.Ny = Ndivs[1]
        self.Nz = Ndivs[2]
        # define the mesh for 
        self.msh = ''
        # dimension
        self.dim = nsd
        # element type
        #self.elem = 
        # domain volume
        self.vol = 1.0

class GeneralComputationalDomain:
    def __init__(self, nsd, meshfile):
        # define the mesh for 
        self.msh = Mesh(meshfile)
        # dimension
        self.dim = nsd
        # element type
        #self.elem = 
        # domain volume
        self.vol = 1.0
#
# IO utils
class IO_DDMC:
    def __init__(self, dir):
        # creating directory and file to write
        if(dir == None):
            dir = 'results'
        if os.path.exists(dir) == False:
            os.makedirs(dir)
        self.dir = dir

#
# Spaces and functions utils
class Spaces:
    def __init__(self, domain):
        self.sp = FunctionSpace(domain.msh, "CG", 1)
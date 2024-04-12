from firedrake import *
from firedrake.output import VTKFile
from firedrake.__future__ import interpolate

class StandardModel:
    def __init__(self, model):
        self.model = model
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

    def FluxLaw(self, potencial, fluxo):
        if(model == 'Darcy-Linear'):
            return fluxo + self.kappa * grad(potencial)
        elif():

#
# geometry data: domain size, mesh, etc ...
class SimpleComputationalDomain:
    def __init__(self, nsd, Lsides, Ndivs, flag_quads):
        # Number of spatial dimension
        if(nsd == 3):
            raise Exception('3D not implemented yet')
        else:
            self.dim = nsd
        # Domain size
        self.Lx = Lsides[0]
        self.Ly = Lsides[1]
        # Mesh discretization
        self.Nx = Ndivs[0]
        self.Ny = Ndivs[1]
        # Element type 
        self.flag_quads = flag_quads
        self.msh = RectangleMesh(Nx, Ny, Lx, Ly, quadrilateral=flag_quads)
        # UFL element type
        #self.elem = self.msh.ufl_elem???
        # Domain volume
        self.vol = 1.0

class GeneralComputationalDomain:
    def __init__(self, nsd, meshfile):
        # Number of spatial dimension
        if(nsd == 3):
            raise Exception('3D not implemented yet')
        else:
            self.dim = nsd
        # Read the mesh
        self.msh = Mesh(meshfile)
        # UFL element type
        #self.elem = self.msh.ufl_elem???
        # Domain volume
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
class CommonSpaces:
    def __init__(self, domain, deg_flux, deg_pot):
        self.deg_flux = deg_flux
        self.deg_pot = deg_pot
        self.sp = FunctionSpace(domain.msh, "CG", 1)
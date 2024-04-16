from firedrake import *
from firedrake.output import VTKFile
from firedrake.__future__ import interpolate

class StandardModel:
    def __init__(self, model, params, petsc_opt):
        self.petsc_opt = petsc_opt
        
        self.model = model
        if (model == 'Darcy-Linear'):
            self.kappa = params[0]

        elif(model == 'Darcy-Nonlinear'):
            self.kappa0 = params[0]

        elif(model == 'Forcheimer-Reg'):
            self.eta = params[0]
            self.muokap = params[1]
            self.betarho = params[2]

        elif(model == 'Klinkenberg'):
            self.kapomu = params[0]
            self.DK = params[1]
        else:
            print('ERROR: In standard_model()')
            print('REASON: Model not implemented')
            exit()

    #def InitialGuess(self, udfguess):
    #    return udfguess()

    def FluxLaw(self, potential, flux):
            
        if(self.model == 'Darcy-Linear'):
            return flux + self.kappa * grad(potential)

        elif(self.model == 'Forcheimer-Reg'):
            modreg = self.eta*(sqrt(1.0 + inner(flux,flux)/self.eta**2) - 1.0)
            return grad(potential) + self.muokap*flux + self.betarho * modreg * flux

        elif(self.model == 'Klinkenberg'):
            return flux + self.kapomu * grad(potential) + (self.DK/potential)*grad(potential)

    

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
class IO_DDCM:
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

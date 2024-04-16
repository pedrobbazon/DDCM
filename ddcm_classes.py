import firedrake as fd
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
class ComputationalDomain:
    def __init__(self, nsd=2, Lsides=[1.0,1.0], Ndivs=[1,1], meshfile=None, flag_quads=False):
        # Number of spatial dimension
        if(nsd == 3):
            raise Exception('3D not implemented yet')
        else:
            self.dim = nsd

        self.meshfile = meshfile
        self.simple_domain = False
        if(self.meshfile == None): # Then, it is obviously a simple rectangle mesh
            self.simple_domain = True

        # Domain size
        self.Lx = Lsides[0]
        self.Ly = Lsides[1]
        # Mesh discretization
        self.Nx = Ndivs[0]
        self.Ny = Ndivs[1]

        # Element type 
        self.flag_quads = flag_quads

        # Build the mesh
        self.msh = self.Create()

        # Some useful spaces associated to this domain mesh
        self.DGtype = 'DG'
        if(self.flag_quads):
            self.DGtype = "DQ"
        self.DG0  = fd.FunctionSpace(self.msh, self.DGtype, degree=0)
        self.VDG0 = fd.FunctionSpace(self.msh, self.DGtype, degree=0)
        self.CG   = fd.FunctionSpace(self.msh, 'CG', degree=1)
        self.Hdiv = None # to be defined if needed on each case

        # Domain volume
        one = fd.Function(self.DG0).interpolate(fd.Constant(1.0))
        self.volume = fd.assemble(one*dx)

        # Element areas
        tstdg          = fd.TestFunction(self.DG0)
        self.area_els  = fd.assemble(one*tstdg*dx)

    def Create(self):
        if(self.simple_domain):
            return fd.RectangleMesh(self.Nx, self.Ny, 
                                    self.Lx, self.Ly, 
                                    quadrilateral=self.flag_quads)
        else:
            return fd.Mesh(self.meshfile)

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
# Set up the (discrete) data to be used
class DDCMData():
    def __init__(self, read_data=False):
        self.ddcm_data = None

    #----------------------------------------------------------------------------
    # Read some experimental (Flux-Grad) data set and accommodate it 
    # into the appropriate format to be consumed by ddcm_solvers
    def SetManualData():
        ddcm_data = 0
        return ddcm_data

    #----------------------------------------------------------------------------
    # From a standard FEM solution of the corresponding physical problem,
    # sample the (Flux-Grad) data set and accommodate it 
    # into the appropriate format to be consumed by ddcm_solvers
    def CreateSyntheticData(std_solution, read_data):
        ddcm_data = 0
        return ddcm_data

class StandardSolution():
    def __init__(self):
        self.u = None # Potential function (pressure, temperature or displacement)
        self.s = None # Flux function (mass flux, heat flux or Stress tensor)
        self.g = None # Gradient (usual gradient or symmetric gradient)

class DDCMSolution():
    def __init__(self):
        self.u   = None  # Potential function (pressure, temperature or displacement)
        self.s   = None  # Flux function (mass flux, heat flux or Stress tensor)
        self.g   = None  # Gradient (usual gradient or symmetric gradient)
        self.lmd = None  # Lagrange multiplier associated to balance equation
        self.mu  = None  # Lagrange multiplier associated to compatibility condition
        
class StdModel():
    def __init__(self, flux_law, fem_formulation, setting_type):
        self.flux_law        = flux_law        # Darcy-Linear, Darcy-Nonlinear, Forcheimer, etc
        self.fem_formulation = fem_formulation # Dual-Mixed-Hdiv, Mixed-Hdiv, Primal-CG
        self.setting_type    = setting_type    # Uniaxial, Real-Setting

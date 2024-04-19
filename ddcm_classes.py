import firedrake as fd
from firedrake import *
from firedrake.output import VTKFile
from firedrake.__future__ import interpolate
#
#-----------------------------------------------------------------------------
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
        self.VDG0 = fd.VectorFunctionSpace(self.msh, self.DGtype, degree=0)
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
# Actual functions containing a standard solution
class StdSolution():
    def __init__(self):
        self.u = None # Potential function (pressure, temperature or displacement)
        self.s = None # Flux function (mass flux, heat flux or Stress tensor)
        self.g = None # Gradient (usual gradient or symmetric gradient)

from PoissonSolvers import *

#
# Encapsulate the forcings
class ForcingData():
    def __init__(self, volumetric_source, boundary_flux, boundary_vals):
        self.volumetric_source = volumetric_source
        self.boundary_flux = boundary_flux
        self.boundary_vals = boundary_vals

# 
# Define the Standard problem, model and solver
class StdProblem():
    def __init__(self, domain, params, setting_type_flag: str, forcings, flux_law, fem_formulation, orders):
        self.domain = domain
        self.params = params
        self.setting_type    = setting_type_flag  # Uniaxial or Real-Setting
        self.forcings = forcings                  # Source functio, Surface flux and BC vals
        self.flux_law        = flux_law           # Darcy-Linear, Darcy-Nonlinear, Forcheimer, etc
        self.fem_formulation = fem_formulation    # Dual-Mixed-Hdiv, Mixed-Hdiv, Primal-CG
        self.std_solution = StdSolution()
        self.orders = orders                      # Orders of Flux and potential approximations
    
    def solve(self, dir_outs=None):
        
        if(self.flux_law != None):
            fluxlaw_name = self.flux_law.__name__+'()'
        else:
            fluxlaw_name = 'No explicit'
        print( "|:| Solving with %s() FEM-formulation and %s flux law" 
              %(self.fem_formulation.__name__, fluxlaw_name))

        sigma, u, g = self.fem_formulation(self.domain, self.domain.flag_quads, 
                                           self.params, self.setting_type, 
                                           self.forcings.volumetric_source,
                                           self.forcings.boundary_flux,
                                           self.forcings.boundary_vals,
                                           self.flux_law, 
                                           self.orders,
                                           dir_outs)
        self.std_solution.u = u
        self.std_solution.s = sigma 
        self.std_solution.g = g 

        return self.std_solution

#
# IO utility functions
class IO_DDCM:
    def __init__(self, dir):
        # creating directory and file to write
        if(dir == None):
            dir = 'results'
        if os.path.exists(dir) == False:
            os.makedirs(dir)
        self.dir = dir

# 
# Actual functions containing the DDCM solution
class DDCMSolution():
    def __init__(self):
        self.u   = None  # Potential function (pressure, temperature or displacement)
        self.s   = None  # Flux function (mass flux, heat flux or Stress tensor)
        self.g   = None  # Gradient (usual gradient or symmetric gradient)
        self.lmd = None  # Lagrange multiplier associated to balance equation
        self.mu  = None  # Lagrange multiplier associated to compatibility condition

#
# Discrete data to be used in the DDCM problem
class DDCMData():
    def __init__(self, read_data=False):
        self.ddcm_data = None
        self.read_data = read_data
        self.domain = domain
        self.params = params
        self.setting_type    = setting_type_flag  # Uniaxial or Real-Setting
        self.forcings = forcings                  # Source functio, Surface flux and BC vals
        self.flux_law        = flux_law           # Darcy-Linear, Darcy-Nonlinear, Forcheimer, etc
        self.fem_formulation = fem_formulation    # Dual-Mixed-Hdiv, Mixed-Hdiv, Primal-CG
        self.std_solution = StdSolution()
        self.orders = orders                      # Orders of Flux and potential approximations

    #----------------------------------------------------------------------------
    # From a standard FEM solution of the corresponding physical problem,
    # sample the (Flux-Grad) data and accommodate it in
    # the appropriate format to be consumed by ddcm_solvers
    def SetSyntheticData(self, std_solution):
        if(self.read_data):
            # Do something
            ddcm_data = 0
            return ddcm_data
        else:
            a = 1
    
    def AddNoisetoData(self):
        return 0

#
# Program the different DDCM formulations
class DDCMProblem():

    def __init__(self, ddcm_data, ddcm_formulation: str):
        self.ddcm_data = ddcm_data
        self.ddcm_formulation = ddcm_formulation

    def solve(self):
        if(self.ddcm_formulation == 'NLSQP-5F-A'):

            ddcm_sol = 0
    
        elif(self.ddcm_formulation == 'NLSQP-5F-B'):

            ddcm_sol = 0

        else:
            raise ValueError(self.ddcm_formulation, 'not implemented')
    

        return ddcm_sol

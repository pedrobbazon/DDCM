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
        print("|> Solving with:\n   - %s() FEM-formulation\n   - %s Flux-Law" 
              %(self.fem_formulation.__name__, fluxlaw_name))
        print("   - Setting type: %s" %(self.setting_type))
        print("   ... Solving ...")
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


import DDCMSolvers
import scipy as sp

#
# Discrete data to be used in the DDCM problem
class DDCMData():
    def __init__(self, domain):
        self.domain = domain
        self.ddcm_fluxdata = fd.Function(domain.VDG0)
        self.ddcm_graddata = fd.Function(domain.VDG0)

    #----------------------------------------------------------------------------
    # Read the (Flux-Grad) data from a file and accommodate it in
    # the appropriate format to be consumed by ddcm_solvers
    def ReadDataFromFile(self):
        return 0

    #----------------------------------------------------------------------------
    # From a standard FEM solution of the corresponding physical problem,
    # sample the (Flux-Grad) data and accommodate it in
    # the appropriate format to be consumed by ddcm_solvers
    def AppendSyntheticData(self, synthetic_solution):

        #state_vector = np.dstack([grad0, grad1, flux0, flux1])[0]
        return 0    
    
    def AddNoisetoData(self, noise_parameters):
        return 0

    def Distribute(self, ref_solution):
        #reference_state = np.dstack([grad0, grad1, flux0, flux1])[0]
        #ddcm_state = np.dstack([grad0, grad1, flux0, flux1])[0]
        distances, indexes = sp.KDTree(ddcm_state, reference_state)
        #self.ddcm_graddata.dat.data[:] = observed_data[indexes,0:2]
        #self.ddcm_fluxdata.dat.data[:] = observed_data[indexes,2:4]

        return 0

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
# Program the different DDCM formulations
class DDCMProblem():

    def __init__(self, ddcm_data, ddcm_formulation):
        self.ddcm_data = ddcm_data
        self.ddcm_formulation = ddcm_formulation
        self.J = None     # Function to be minimized

    def solve(self):
        ddcm_sol = self.ddcm_formulation(self.ddcm_data)
    
        return ddcm_sol

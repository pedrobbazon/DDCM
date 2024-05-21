import firedrake as fd
from firedrake import *
from firedrake.output import VTKFile
#from firedrake.__future__ import interpolate
#
import numpy as np
import scipy
from scipy.spatial import KDTree
import time
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
        self.TDG0 = fd.TensorFunctionSpace(self.msh, self.DGtype, degree=0)
        self.CG1  = fd.FunctionSpace(self.msh, 'CG', degree=1)
        self.VCG1 = fd.VectorFunctionSpace(self.msh, 'CG', degree=1)
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
from MechanicalSolvers import *

#
# Encapsulate the forcings
class MechanicalForcingData():
    def __init__(self, body_force, boundary_force, boundary_vals):
        self.body_force = body_force
        self.boundary_force = boundary_force
        self.boundary_vals = boundary_vals

class PoissonForcingData():
    def __init__(self, volumetric_source, boundary_flux, boundary_vals):
        self.volumetric_source = volumetric_source
        self.boundary_flux = boundary_flux
        self.boundary_vals = boundary_vals

# 
# Define the Standard Poisson problem, model and solver
class PoissonStdProblem():
    def __init__(self, domain, params, setting_type_flag: str, forcings, flux_law, fem_formulation, orders):
        self.domain = domain
        self.params = params
        self.setting_type    = setting_type_flag  # Uniaxial or Real-Setting
        self.forcings        = forcings           # Source functio, Surface flux and BC vals
        self.flux_law        = flux_law           # Darcy-Linear, Darcy-Nonlinear, Forcheimer, etc
        self.fem_formulation = fem_formulation    # Dual-Mixed-Hdiv, Mixed-Hdiv, Primal-CG
        self.std_solution    = StdSolution()
        self.orders          = orders             # Orders of Flux and potential approximations
    
    def solve(self, tt=0.0, dir_outs=None):
        
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
                                           tt, dir_outs)
        self.std_solution.u = u
        self.std_solution.s = sigma 
        self.std_solution.g = g 

        return self.std_solution
    
# Define the Standard Mechanical problem, model and solver
class MechanicalStdProblem():
    def __init__(self, domain, params, setting_type_flag: str, forcings, 
                 stress_strain_law, fem_formulation, orders, energy_function=None):
        self.domain = domain
        self.params = params
        self.setting_type = setting_type_flag     # Uniaxial or Real-Setting
        self.forcings = forcings                  # Source functio, Surface flux and BC vals
        self.flux_law        = stress_strain_law  # Stress-Strain constitutive law: linear isotropic, hyperelastic, etc.
        self.energy_function = energy_function    # Energy as a functions of deformation measures
        self.fem_formulation = fem_formulation    # Primal-CG will be possibly the only one for now
        self.std_solution = StdSolution()
        self.orders = orders                      # Orders of Stress and displacement field approximations
    
    def solve(self, dir_outs=None):
        
        if(self.flux_law != None):
            tensdef_name = self.flux_law.__name__+'()'
        else:
            tensdef_name = 'No explicit'
        print("|> Solving with:\n   - %s() FEM-formulation\n   - %s StressDef-Law" 
              %(self.fem_formulation.__name__, tensdef_name))
        print("   - Setting type: %s" %(self.setting_type))
        print("   ... Solving ...")
        s, u, g = self.fem_formulation(self.domain, self.domain.flag_quads, 
                                       self.params, self.setting_type, 
                                       self.forcings.volumetric_source,
                                       self.forcings.boundary_flux,
                                       self.forcings.boundary_vals,
                                       self.flux_law, 
                                       self.orders,
                                       dir_outs)
        self.std_solution.s = s 
        self.std_solution.u = u
        self.std_solution.g = g 

        return self.std_solution
    
from DDCMSolvers import *

#
# Discrete data to be used in the DDCM problem
class DDCMData():
    def __init__(self, domain):
        # The DDCM data is associated to a domain
        self.domain = domain
        # The complete sampled data set 
        self.g_sampled = None
        self.s_sampled = None
        '''
        Data distributed over the domain to be used by the DDCM solver:
        By option we assign the data to the elements.
        Therefore, the natural choice is to use elementwise discontinuous spaces,
        this way, the induced domain decompostion from the data
        will conform to the element edges
        '''
        self.g_distrib = fd.Function(domain.VDG0)
        self.s_distrib = fd.Function(domain.VDG0)

    #----------------------------------------------------------------------------
    # Read the (Flux-Grad) data from a file and accommodate it in
    # the appropriate format to be consumed by ddcm_solvers
    def ReadDataFromFile(self, datafile):
        dataset = np.loadtxt(datafile)
        # We expect the data in this format:
        self.sampled_gradu = dataset[:,0:2] 
        self.sampled_flux  = dataset[:,2:4] 
        return 0

    def SetSampledData(self, gradu_vals, flux_vals, flag_savetxt=None):
        self.sampled_grad = gradu_vals 
        self.sampled_flux = flux_vals
        if(flag_savetxt != None):
            np.savetxt('DATA.txt', (gradu_vals, flux_vals))
        return 0
    
    def AddNoisetoData(self, noise_parameters: dict):
        #self.sampled_gradu += GaussianNoise()
        #self.sampled_gradu += GaussianNoise()
        return 0

    def Distribute(self, ref_solution, flag_project='Project-VDG0', flag_plot=False):

        # Something reasonable would be to ask that g and s to be in the same space
        if(flag_project == 'Project-VDG0'):
            refgrad = fd.project(ref_solution.g, self.domain.VDG0).dat.data
            refflux = fd.project(ref_solution.s, self.domain.VDG0).dat.data
        else:
            raise ValueError(flag_project+' option not recognized in method Distribuite()')
        
        reference_state = np.dstack([refgrad[:,0], refgrad[:,1], refflux[:,0], refflux[:,1]])[0]

        # These are only vectors with data, not associated yet to any domain
        sampgrad = self.sampled_grad
        sampflux = self.sampled_flux
        sampled_state = np.dstack([sampgrad[:,0], sampgrad[:,1], sampflux[:,0], sampflux[:,1]])[0]

        # Choose those values that are closest to the reference solution
        # and distribute over the domain (this induces a domain decomposition)
        mytree = scipy.spatial.KDTree(sampled_state)
        distances, indexes = mytree.query(reference_state)
        self.graddata.dat.data[:] = self.sampled_grad[indexes,0:2]
        self.graddata.rename('GradDataDistrib')
        self.fluxdata.dat.data[:] = self.sampled_flux[indexes,0:2]
        self.fluxdata.rename('FluxDataDistrib')

        if(flag_plot):
            index_field = fd.Function(self.domain.DG0)
            index_field.dat.data[:] = indexes.astype(float)
            index_field.rename('IndexDataDistrib')
            VTKFile('indexdata.pvd').write(index_field)
            VTKFile('graddata.pvd').write(self.graddata)
            VTKFile('fluxdata.pvd').write(self.fluxdata)

        return 0
#
# Discrete data to be used in the DDCM problem
class PoissonDDCMData():
    def __init__(self, domain):
        # The DDCM data is associated to a domain
        self.domain = domain
        # The complete sampled data set 
        self.sampled_grad = None
        self.sampled_flux = None
        '''
        Data distributed over the domain to be used by the DDCM solver:
        by option we assign the data to the elements.
        Therefore, the natural choice is to use elementwise discontinuous spaces,
        this way, the induced domain decompostion from the data
        will conform to the element edges
        '''
        self.graddata = fd.Function(domain.VDG0)
        self.fluxdata = fd.Function(domain.VDG0)

    #----------------------------------------------------------------------------
    # Read the (Flux-Grad) data from a file and accommodate it in
    # the appropriate format to be consumed by ddcm_solvers
    def ReadDataFromFile(self, datafile):
        dataset = np.loadtxt(datafile)
        # We expect the data in this format:
        self.sampled_gradu = dataset[:,0:2] 
        self.sampled_flux  = dataset[:,2:4] 
        return 0

    def SetSampledData(self, gradu_vals, flux_vals, add_origin=False, data_file=None):
        self.sampled_grad = gradu_vals 
        self.sampled_flux = flux_vals
        if(add_origin):
            self.sampled_grad = np.append(self.sampled_grad, [[0.0, 0.0]], axis=0)
            self.sampled_flux = np.append(self.sampled_flux, [[0.0, 0.0]], axis=0)
        if(data_file != None):
            np.savetxt(data_file, np.hstack((gradu_vals, flux_vals)))
        return 0
    
    def AddNoisetoData(self, noise_parameters: dict):
        #self.sampled_gradu += GaussianNoise()
        #self.sampled_gradu += GaussianNoise()
        return 0

    def Distribute(self, ref_solution, flag_project='Project-VDG0', flag_plot=False):

        ti = time.time()
        # Something reasonable would be to ask that g and s to be in the same space
        if(flag_project == 'Project-VDG0'):
            refgrad = fd.project(ref_solution.g, self.domain.VDG0).dat.data
            refflux = fd.project(ref_solution.s, self.domain.VDG0).dat.data
        elif(flag_project == 'Interpolate-VDG0'): 
            x = fd.SpatialCoordinate(self.domain.msh)
            # This is typically used to intepolate a manufactured solution onto a vector DG0 space
            refgrad = fd.assemble(interpolate(ref_solution.g, self.domain.VDG0)).dat.data
            refflux = fd.assemble(interpolate(ref_solution.s(x), self.domain.VDG0)).dat.data
        else:
            raise ValueError(flag_project+' option not recognized in method Distribuite()')
        
        reference_state = np.dstack([refgrad[:,0], refgrad[:,1], refflux[:,0], refflux[:,1]])[0]

        # These are only vectors with data, not associated yet to any domain
        sampgrad = self.sampled_grad
        sampflux = self.sampled_flux
        sampled_state = np.dstack([sampgrad[:,0], sampgrad[:,1], sampflux[:,0], sampflux[:,1]])[0]

        print(' . Proj-Int= ', time.time() - ti)

        # Choose those values that are closest to the reference solution
        # and distribute over the domain (this induces a domain decomposition)
        ti = time.time()
        mytree = scipy.spatial.KDTree(sampled_state)
        distances, indexes = mytree.query(reference_state)
        self.graddata.dat.data[:] = self.sampled_grad[indexes,0:2]
        self.fluxdata.rename('GradDataDistrib')
        self.fluxdata.dat.data[:] = self.sampled_flux[indexes,0:2]
        self.fluxdata.rename('FluxDataDistrib')
        print(' . KDTree and query= ', time.time() - ti)

        if(flag_plot):
            ti = time.time()
            index_field = fd.Function(self.domain.DG0)
            index_field.dat.data[:] = indexes.astype(float)
            index_field.rename('IndexDataDistrib')
            VTKFile('mydatafields/indexdata.pvd').write(index_field)
            VTKFile('mydatafields/graddata.pvd').write(self.graddata)
            VTKFile('mydatafields/fluxdata.pvd').write(self.fluxdata)
            print(' . Plot= ', time.time() - ti)

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

    def __init__(self, ddcm_data, ObjF, dObjF, setting_type_flag: str, forcings, ddcm_formulation, orders):
        self.ddcm_data = ddcm_data
        self.J = ObjF        # Function to be minimized
        self.dJ = dObjF      # Jacobian of function to be minimized (can be given or compute by the program)
        self.setting_type = setting_type_flag
        self.ddcm_formulation = ddcm_formulation
        self.forcings = forcings
        self.orders = orders
        self.ddcm_solution = DDCMSolution()

    def solve(self, dir_outs=None):

        print("|> Solving with:\n   - %s() DDCM-formulation" %(self.ddcm_formulation.__name__))
        print("   - Setting type: %s" %(self.setting_type))
        print("   ... Solving ...")

        s, u, g, lmd, mu = self.ddcm_formulation(self.ddcm_data.domain,
                                                 self.J, self.dJ,
                                                 self.ddcm_data.graddata, self.ddcm_data.fluxdata,
                                                 self.ddcm_data.domain.flag_quads, 
                                                 self.setting_type, 
                                                 self.forcings.volumetric_source,
                                                 self.forcings.boundary_flux,
                                                 self.forcings.boundary_vals,
                                                 self.orders,
                                                 dir_outs)
        
        self.ddcm_solution.u   = u
        self.ddcm_solution.s   = s 
        self.ddcm_solution.g   = g
        self.ddcm_solution.lmd = lmd
        self.ddcm_solution.mu  = mu 
    
        return self.ddcm_solution

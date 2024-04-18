import firedrake as fd
from firedrake import *
from firedrake.output import VTKFile
from firedrake.__future__ import interpolate
#
#----------------------------------------------------------------
#
# This file contains user defined utility functions to set your
# standard Poisson model, including constitutive and numerical 
# parameters, flux laws, fem formulations, etc.
# ADD ANYTHING YOU NEED BELOW
#
#----------------------------------------------------------------
# Different petsc option solvers that might be used by
# the Poisson solvers below
#
petsc_opt1 = {
    'snes_type': 'newtonls',
    'snes_monitor': None,
    'snes_converged_reason': None,
    'snes_linesearch_type': 'basic',
    'snes_max_it': 100,
    'snes_divergence_tolerance': 1e20,
    'snes_rtol': 1e-9,
    'snes_atol': 1e-11,
    'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
}
#
petsc_opt2 = {
    'snes_type': 'newtonls',
    'snes_monitor': None,
    'snes_converged_reason': None,
    'snes_linesearch_type': 'basic',
    'snes_max_it': 100,
    'snes_divergence_tolerance': 1e20,
    'snes_rtol': 1e-10,
    'snes_atol': 1e-12,
    'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'superlu-dist'
}
#
#----------------------------------------------------------------
#
# Program your Flux Law: (Mixed or Primal): 
# Follow format for the names: FLX_mymodel()
#
def FLM_DarcyLinear(potential, flux, params):
    kappa = params[0]
    return flux + kappa * grad(potential)

def FLM_ForcheimerReg(potential, flux, params):
    eta = params[0]
    muokap = self.params[1]
    betarho = self.params[2]
    modreg = eta*(sqrt(1.0 + inner(flux,flux)/eta**2) - 1.0)
    return grad(potential) + muokap*flux + betarho * modreg * flux

def FLM_Klinkenberg(potential, flux, params):
    kapomu = params[0]
    DK = params[1]
    return flux + kapomu * grad(potential) + (DK/potential)*grad(potential)
    
def FLM_DarcyNonLinear(potential, flux, params):
    def Kofu(u, params):
        kappa0 = params[0]
        viol = params[1]
        return kappa0*(1 + 10*0.5 * (1.0 + tanh(viol*(u-2.0))))
    return flux + Kofu(potential,params) * grad(potential)
    
def FLP_DarcyLinear(potential, params):
    kappa = params[0]
    return -kappa * grad(potential)

#-------------------------------------------------------------------------------------
# Program your FEM formulation (Mixed or Primal):
# Use representative names (e.g. PoissonPrimalCG(), PoissonDualMixedHdiv(), etc.)
#
def PoissonDualMixedHdiv(domain, flag_quads, params, flag_setting,
                         volumetric_source, boundary_flux, boundary_vals,
                         FluxLaw, orders, dir_outs):
      
    mesh = domain.msh

    orderF, orderU = orders

    if(flag_quads):
        HD = fd.FunctionSpace(mesh, "RTCF", orderF)
        CG = fd.FunctionSpace(mesh, "CG", orderU)
    else:
        HD = fd.FunctionSpace(mesh, "DRT", orderF)
        CG = fd.FunctionSpace(mesh, "CG", orderU)

    W = HD * CG

    w = fd.Function(W)
    sigma, u = fd.split(w)
    tau, v = fd.TestFunctions(W)
    help(sigma)

    x = fd.SpatialCoordinate(mesh)
    n = fd.FacetNormal(mesh)

    if(flag_setting == 'Setting-A'):
        
        q = volumetric_source(x)
        bcall = [fd.DirichletBC(W.sub(1), boundary_vals(x), 'on_boundary')]

    elif(flag_setting == 'Uniaxial'):
        
        # By definition, the uniaxial test consists of this:
        q = fd.Constant(0.0)
        pbottom, ptop = fd.Constant(1.0), boundary_vals(x)
        flag_bottom, flag_top = 3, 4
        bcall = [fd.DirichletBC(W.sub(1), pbottom, flag_bottom), fd.DirichletBC(W.sub(1), ptop, flag_top)]

    else:
        
        raise ValueError('Unrecognized problem setting option')
    
    Res  = (inner(FluxLaw(u, sigma, params),tau))*fd.dx
    Res += inner(sigma,grad(v))*fd.dx
    Res += q*v*fd.dx
    fd.solve(Res == 0, w, bcs=bcall, solver_parameters=petsc_opt1)

    # It may be reasonable that the output be consistent with the choices above
    if(dir_outs != None):
        filename = dir_outs+'/flux.pvd'
        VTKFile(filename).write(w.sub(0)) #project(w.sub(0), HD, name="Flux"))
        filename = dir_outs+'/u.pvd'
        VTKFile(filename).write(w.sub(1)) #project(w.sub(0), HD, name="Flux"))

    return w.sub(0), w.sub(1), fd.project(grad(u), domain.VDG0)
#-------------------------------------------------------------------------------------

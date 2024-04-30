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
# Nonlinear solver with mumps
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
# Simple direct solver
petsc_opt2 = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_monitor": None
}
#
# # Simple krylov solver
petsc_opt3 = {
    "ksp_type": "gmres",
    "ksp_gmres_restart": 100,
    "ksp_rtol": 1e-8,
    "pc_type": "ilu",
    "ksp_monitor": None
}
#
#----------------------------------------------------------------
#
# Program your Flux Law: (Mixed or Primal): 
# Follow format for the names: FLX_mymodel()
#
def FLM_DarcyLinear(potential, flux, params):
    kappa = params['kappa']
    return flux + kappa * grad(potential)

def FLM_ForcheimerReg(potential, flux, params):
    eta = params[0]
    muokap = params[1]
    betarho = params[2]
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
#-------------------------------------------------------------------------------------

#=====================================================================================

def PoissonDualMixedHdiv(domain, flag_quads, params, flag_setting,
                         volumetric_source, boundary_flux, boundary_vals,
                         FluxLaw, orders, dir_outs):
    
    # In this formulation we need constitutive 
    # parameters and a FluxLaw in which sigma might
    # not be explicity obtained from grad(u)
      
    mesh = domain.msh

    orderF, orderU = orders
    if(flag_quads):
        HD = fd.FunctionSpace(mesh, "RTCF", orderF)
        CG = fd.FunctionSpace(mesh, "CG", orderU)
    else:
        HD = fd.FunctionSpace(mesh, "DRT", orderF)
        CG = fd.FunctionSpace(mesh, "CG", orderU)

    W = HD * CG

    w = fd.Function(W, name='Solution')
    sigma, u = fd.split(w)
    tau, v = fd.TestFunctions(W)

    x = fd.SpatialCoordinate(mesh)
    n = fd.FacetNormal(mesh)

    if(flag_setting == 'Pure-Dirichlet'):
        
        q = volumetric_source(x)

        # u bc (strong in this formulation)
        bcall = [fd.DirichletBC(W.sub(1), boundary_vals(x), 'on_boundary')]

    elif(flag_setting == 'Uniaxial'):
        
        # By definition, the uniaxial test consists of this:
        q = fd.Constant(0.0)

        # u bc (strong in this formulation)
        ubottom, utop = fd.Constant(1.0), boundary_vals(x)
        tag_bottom, tag_top = 3, 4
        bcall = [fd.DirichletBC(W.sub(1), ubottom, tag_bottom), fd.DirichletBC(W.sub(1), utop, tag_top)]

    else:
        
        raise ValueError('Unrecognized problem setting option')
    
    Res  = (fd.inner(FluxLaw(u, sigma, params),tau))*fd.dx(degree=4)
    Res += fd.inner(sigma,fd.grad(v))*fd.dx
    Res += q*v*fd.dx(degree=4)

    fd.solve(Res == 0, w, bcs=bcall, solver_parameters=petsc_opt1)
    g = fd.grad(u)

    # It may be reasonable that the output be consistent with the choices
    # of spaces above. That's why I program this here 
    # e.g., if u \in DG1, g may belong to DG0, but if u \in DG0 then we should 
    # integrate by parts during projection??
    if(dir_outs != None):

        filename = dir_outs+'/flux.pvd'
        w.sub(0).rename('Flux')
        VTKFile(filename).write(w.sub(0))

        filename = dir_outs+'/u.pvd'
        w.sub(1).rename('Potential')
        VTKFile(filename).write(w.sub(1))
        
        filename = dir_outs+'/gradu.pvd'
        gpl = fd.project(g, domain.VDG0, name='Grad(u)')
        VTKFile(filename).write(gpl)

    return w.sub(0), w.sub(1), g

#=====================================================================================

def PoissonMixedHdiv(domain, flag_quads, params, flag_setting,
                     volumetric_source, boundary_flux, boundary_vals,
                     FluxLaw, orders, dir_outs):

    # In this formulation the FluxLaw is not used.
    # We just need the definition of kappa and assume
    # that we can write (1/K)sigma = -grad(u)

    mesh = domain.msh

    orderF, orderU = orders
    if(flag_quads):
        HD = fd.FunctionSpace(mesh, "RTCF", orderF)
        DG = fd.FunctionSpace(mesh, "DQ", orderU)
    else:
        HD = fd.FunctionSpace(mesh, "RT", orderF)
        DG = fd.FunctionSpace(mesh, "DG", orderU)

    W = HD * DG

    w = fd.Function(W, name='Solution')
    sigma, u = fd.split(w)
    tau, v = fd.TestFunctions(W)

    x = fd.SpatialCoordinate(mesh)

    n = fd.FacetNormal(mesh)

    if(flag_setting == 'Pure-Dirichlet'):
        
        q = volumetric_source(x)
        
        # Pressure bc (weak in this formulation)
        pressure_boundary_term = boundary_vals(x)*fd.dot(tau,n)*fd.ds(degree=4)
        
        # Flux bc (strong in this formulation) - empty for this setting
        bcall = []

        # Some non-zero initial guess to avoid singularities
        w.sub(1).interpolate(boundary_vals(x))
        
    elif(flag_setting == 'Uniaxial'):
        
        # By definition, the uniaxial test consists of this:
        q = fd.Constant(0.0)
        
        # Pressure bc (weak in this formulation)
        tag_bottom, tag_top = 3, 4
        ubottom, utop = fd.Constant(1.0), boundary_vals(x)
        pressure_boundary_term = ubottom*fd.dot(tau,n)*fd.ds(tag_bottom) + utop*fd.dot(tau,n)*fd.ds(tag_top)

        # Flux bc (strong in this formulation)
        no_flux = fd.as_vector([fd.Constant(0.0), fd.Constant(0.0)])
        tag_left, tag_right = 1, 2
        bcall = [fd.DirichletBC(W.sub(0), no_flux, tag_left), fd.DirichletBC(W.sub(0), no_flux, tag_right)]

        # Some non-zero initial guess to avoid singularities
        w.sub(1).interpolate(x[1]*(utop-ubottom) + ubottom)

    else:
        
        raise ValueError('Unrecognized problem setting option')

    kappa = params[0]
    Res  = ((1.0/kappa(u))*fd.dot(sigma, tau) - fd.div(tau)*u - fd.div(sigma)*v)*fd.dx(degree=4)
    Res += q*v*fd.dx(degree=4) + pressure_boundary_term

    fd.solve(Res == 0, w, bcs=bcall, solver_parameters=petsc_opt1)

    # Compute g in a consistent way?
    gtr = fd.TrialFunction(HD)
    tau = fd.TestFunction(HD)
    usol = fd.Function(DG)
    usol.assign(w.sub(1))
    a = fd.inner(gtr,tau)*fd.dx
    L = -usol*fd.div(tau)*fd.dx + usol*fd.dot(tau,n)*fd.ds
    g = fd.Function(HD, name='Hdiv-Gradient')
    fd.solve(a == L, g, bcs=[], solver_parameters=petsc_opt2)

    # It may be reasonable that the output be consistent with the choices
    # of spaces above. That's why I program this here
    if(dir_outs != None):

        filename = dir_outs+'/s.pvd'
        w.sub(0).rename('Flux')
        VTKFile(filename).write(w.sub(0))

        filename = dir_outs+'/u.pvd'
        w.sub(1).rename('Potential')
        VTKFile(filename).write(w.sub(1))

        filename = dir_outs+'/ucg.pvd'
        w.sub(1).rename('Potential-CG')
        VTKFile(filename).write(project(w.sub(1), domain.CG1, name='Potential-CG1'))
        
        filename = dir_outs+'/gradu.pvd'
        #gpl = fd.project(g, domain.VDG0, name='Gradient(u)')
        VTKFile(filename).write(g)

    return w.sub(0), w.sub(1), g

#=====================================================================================
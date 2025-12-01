import numpy as np
import matplotlib.pyplot as plt
import firedrake as fd
from firedrake import *
from firedrake.output import VTKFile
#from firedrake.__future__ import interpolate
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
    'snes_max_it': 10,
    'snes_divergence_tolerance': 1e20,
    'snes_rtol': 1e-11,
    'snes_atol': 1e-10,
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
    return flux + kappa * fd.grad(potential)

def FLM_ForcheimerReg(potential, flux, params):
    eta = params['eta']
    muokap = params['mu_over_kappa']
    betarho = params['beta_x_rho']
    modfluxreg = eta*(fd.sqrt(1.0 + fd.inner(flux,flux)/eta**2) - 1.0)
    return fd.grad(potential) + muokap*flux + betarho * modfluxreg * flux

def FLM_ForcheimerRegAni(potential, flux, params):
    eta = params['eta']
    muokap = params['mu_over_kappa']
    betarho = params['beta_x_rho']
    modfluxreg = eta*(fd.sqrt(1.0 + fd.inner(flux,flux)/eta**2) - 1.0)
    Km1 = fd.as_tensor([[muokap[0,0], muokap[0,1]], [muokap[1,0], muokap[1,1]]])
    Km12 = fd.as_tensor([[fd.sqrt(muokap[0,0]), muokap[0,1]], [muokap[1,0], fd.sqrt(muokap[1,1])]])
    return fd.grad(potential) + Km1*flux + betarho * Km12 * modfluxreg * flux

def FLM_Klinkenberg(potential, flux, params):
    kapomu = params['kappaovermu']
    DK = params['DK']
    return flux + kapomu * fd.grad(potential) + (DK/potential)*fd.grad(potential)
    
def FLM_DarcyNonLinear(potential, flux, params):
    def Kofu(u, params):
        kappa0 = params['kappa0']
        viol = params['viol']
        return kappa0*(1 + 10*0.5 * (1.0 + fd.tanh(viol*(u-2.0))))
    return flux + Kofu(potential,params) * fd.grad(potential)
    
def FLP_DarcyLinear(potential, params):
    kappa = params[0]
    return -kappa * fd.grad(potential)

#-------------------------------------------------------------------------------------
# Program your FEM formulation (Mixed or Primal):
# Use representative names (e.g. PoissonPrimalCG(), PoissonDualMixedHdiv(), etc.)
#-------------------------------------------------------------------------------------

#=====================================================================================

def PoissonDualMixedHdiv(domain, flag_quads, params, flag_setting,
                         volumetric_source, boundary_flux, boundary_vals,
                         FluxLaw, orders, tt=0.0, dir_outs='myrefsol'):
    
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

        if(tt == 0.0): # This time acts as a parameter over which we do continuation
            mode = 'w'
        else:
            mode = 'a'

        filename = dir_outs+'/flux.pvd'
        w.sub(0).rename('Reference.Flux')
        VTKFile(filename, mode=mode).write(w.sub(0), time=tt)

        filename = dir_outs+'/u.pvd'
        w.sub(1).rename('Reference.Potential')
        VTKFile(filename, mode=mode).write(w.sub(1), time=tt)
        
        filename = dir_outs+'/gradu.pvd'
        gpl = fd.project(g, domain.VDG0, name='Reference.Grad(u)')
        VTKFile(filename, mode=mode).write(gpl, time=tt)

    return w.sub(0), w.sub(1), g

def PoissonDualMixedHdivAni(domain, flag_quads, params, flag_setting,
                            volumetric_source, boundary_flux, boundary_vals,
                            FluxLaw, orders, tt=0.0, dir_outs='myrefsol'):
    
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

        if(tt == 0.0): # This time acts as a parameter over which we do continuation
            mode = 'w'
        else:
            mode = 'a'

        filename = dir_outs+'/flux.pvd'
        w.sub(0).rename('Reference.Flux')
        VTKFile(filename, mode=mode).write(w.sub(0), time=tt)

        filename = dir_outs+'/u.pvd'
        w.sub(1).rename('Reference.Potential')
        VTKFile(filename, mode=mode).write(w.sub(1), time=tt)
        
        filename = dir_outs+'/gradu.pvd'
        gpl = fd.project(g, domain.VDG0, name='Reference.Grad(u)')
        VTKFile(filename, mode=mode).write(gpl, time=tt)

    return w.sub(0), w.sub(1), g

#=====================================================================================


def PoissonMixedEO(domain, flag_quads, params, flag_setting,
                   volumetric_source, boundary_flux, boundary_vals,
                   FluxLaw, orders, tt=0.0, dir_outs='myrefsol'):

    # In this formulation the FluxLaw is not used.
    # We just need the definition of kappa and assume
    # that we can write (1/K)sigma = -grad(u)

    mesh = domain.msh

    orderF, orderU = orders
    VCG = fd.VectorFunctionSpace(mesh, 'CG', degree=orderF)
    CG = fd.FunctionSpace(mesh, 'CG', degree=orderU)

    W = VCG * CG

    w = fd.Function(W, name='Solution')
    sigma, u = fd.split(w)
    tau, v = fd.TestFunctions(W)

    x = fd.SpatialCoordinate(mesh)

    n = fd.FacetNormal(mesh)

    if(flag_setting == 'Pure-Dirichlet'):
        
        q = volumetric_source(x)
        
        # Pressure bc (weak in this formulation)

        #pressure_boundary_term = v*fd.dot(sigma,n)*fd.ds(degree=4)
        pressure_boundary_term = (boundary_vals(x) - u)*fd.dot(tau,n)*fd.ds(degree=4)
        flux_boundary_term = v*fd.dot(sigma,n)*fd.ds(degree=4)
        #bcall = [fd.DirichletBC(W.sub(1), boundary_vals(x), "on_boundary")]
        #bcall = [fd.DirichletBC(W.sub(0).sub(0), -boundary_flux(x)[0], 1),
        #         fd.DirichletBC(W.sub(0).sub(0),  boundary_flux(x)[0], 2),
        #         fd.DirichletBC(W.sub(0).sub(1), -boundary_flux(x)[1], 3),
        #         fd.DirichletBC(W.sub(0).sub(1),  boundary_flux(x)[1], 4)]
        #bcall = []
        
        # Flux bc (strong in this formulation) - empty for this setting
        #bcall = [fd.DirichletBC(W.sub(1), boundary_vals(x), 1), fd.DirichletBC(W.sub(1), boundary_vals(x), 2)]
        #bcall = [fd.DirichletBC(W.sub(1), boundary_vals(x), "on_boundary")]

        # Some non-zero initial guess to avoid singularities
        #w.sub(1).interpolate(boundary_vals(x))

    elif(flag_setting == 'Pure-Neumann'):
        
        q = volumetric_source(x)
                
        # Flux bc (strong in this formulation) - empty for this setting
        bcall = [fd.DirichletBC(W.sub(0), fd.as_vector([fd.Constant(0.0), fd.Constant(0.0)]), 'on_boundary')]
        
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

    perm = params['perm']
    Res  = (fd.dot(sigma, tau) + fd.inner(perm(x)*fd.grad(u),tau) - fd.inner(sigma,fd.grad(v)))*fd.dx(degree=4)
    Res += fd.Constant(0.25) * fd.inner(sigma - perm(x)*fd.grad(u), tau - perm(x)*fd.grad(v))*fd.dx(degree=4)
    Res -= q*v*fd.dx(degree=4)
    Res += flux_boundary_term  + pressure_boundary_term

    zeda = params['zeda']
    if(zeda != None):
        Res += zeda(x) * u * v * fd.dx(degree=4)

    #v_basis = fd.VectorSpaceBasis(constant=True)
    #nullspace = fd.MixedVectorSpaceBasis(W, [W.sub(0), v_basis])

    #fd.solve(Res == 0, w, bcs=bcall, solver_parameters=petsc_opt1, nullspace=nullspace)
    fd.solve(Res == 0, w, bcs=[], solver_parameters=petsc_opt1)

    # Compute g in a consistent way?
    gtr = fd.TrialFunction(VCG)
    tau = fd.TestFunction(VCG)
    usol = fd.Function(CG)
    usol.assign(w.sub(1))
    a = fd.inner(gtr,tau)*fd.dx
    L = -usol*fd.div(tau)*fd.dx + usol*fd.dot(tau,n)*fd.ds
    g = fd.Function(VCG, name='Reference.VCG-Gradient')
    fd.solve(a == L, g, bcs=[], solver_parameters=petsc_opt2)

    # It may be reasonable that the output be consistent with the choices
    # of spaces above. That's why I program this here
    if(dir_outs != None):

        if(tt == 0.0): # This time acts as a parameter over which we do continuation
            mode = 'w'
        else:
            mode = 'a'

        filename = dir_outs+'/s.pvd'
        w.sub(0).rename('Reference.Flux')
        VTKFile(filename, mode=mode).write(w.sub(0), time=tt)

        filename = dir_outs+'/u.pvd'
        w.sub(1).rename('Reference.Potential')
        VTKFile(filename, mode=mode).write(w.sub(1), time=tt)

        filename = dir_outs+'/ucg.pvd'
        w.sub(1).rename('Reference.Potential-CG')
        VTKFile(filename, mode=mode).write(fd.project(w.sub(1), domain.CG1, name='Potential-CG1'), time=tt)
        
        filename = dir_outs+'/gradu.pvd'
        #gpl = fd.project(g, domain.VDG0, name='Gradient(u)')
        VTKFile(filename, mode=mode).write(g, time=tt)

    return w.sub(0), w.sub(1), g

def PoissonMixedEOdiv(domain, flag_quads, params, flag_setting,
                      volumetric_source, boundary_flux, boundary_vals,
                      FluxLaw, orders, tt=0.0, dir_outs='myrefsol'):

    # In this formulation the FluxLaw is not used.
    # We just need the definition of kappa and assume
    # that we can write (1/K)sigma = -grad(u)

    mesh = domain.msh

    orderF, orderU = orders
    VCG = fd.VectorFunctionSpace(mesh, 'CG', degree=orderF)
    CG = fd.FunctionSpace(mesh, 'CG', degree=orderU)

    W = VCG * CG

    w = fd.Function(W, name='Solution')
    sigma, u = fd.split(w)
    tau, v = fd.TestFunctions(W)

    x = fd.SpatialCoordinate(mesh)

    n = fd.FacetNormal(mesh)

    if(flag_setting == 'Pure-Dirichlet'):
        
        q = volumetric_source(x)
        
        # Pressure bc (weak in this formulation)

        #pressure_boundary_term = v*fd.dot(boundary_flux(x),n)*(fd.ds(3,degree=4)+fd.ds(4,degree=4))
        #pressure_boundary_term = boundary_vals(x)*fd.dot(tau,n)*fd.ds(degree=4)
        pressure_boundary_term = u*fd.dot(tau,n)*fd.ds(degree=4)
        bcall = [fd.DirichletBC(W.sub(1), boundary_vals(x), "on_boundary")]
        #bcall = []
        
        # Flux bc (strong in this formulation) - empty for this setting
        #bcall = [fd.DirichletBC(W.sub(1), boundary_vals(x), 1), fd.DirichletBC(W.sub(1), boundary_vals(x), 2)]
        #bcall = [fd.DirichletBC(W.sub(1), boundary_vals(x), "on_boundary")]

        # Some non-zero initial guess to avoid singularities
        #w.sub(1).interpolate(boundary_vals(x))

    elif(flag_setting == 'Pure-Neumann'):
        
        q = volumetric_source(x)
                
        # Flux bc (strong in this formulation) - empty for this setting
        bcall = [fd.DirichletBC(W.sub(0), fd.as_vector([fd.Constant(0.0), fd.Constant(0.0)]), 'on_boundary')]
        
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

    perm = params['perm']
    Res  = ((1/perm(x))*fd.dot(sigma, tau) - fd.inner(fd.div(tau),u) - fd.inner(fd.div(sigma),v))*fd.dx(degree=4)
    Res += q*v*fd.dx(degree=4)
    Res += pressure_boundary_term

    zeda = params['zeda']
    if(zeda != None):
        Res += zeda(x) * u * v * fd.dx(degree=4)

    fd.solve(Res == 0, w, bcs=bcall, solver_parameters=petsc_opt1)

    # Compute g in a consistent way?
    gtr = fd.TrialFunction(VCG)
    tau = fd.TestFunction(VCG)
    usol = fd.Function(CG)
    usol.assign(w.sub(1))
    a = fd.inner(gtr,tau)*fd.dx
    L = -usol*fd.div(tau)*fd.dx + usol*fd.dot(tau,n)*fd.ds
    g = fd.Function(VCG, name='Reference.VCG-Gradient')
    fd.solve(a == L, g, bcs=[], solver_parameters=petsc_opt2)

    # It may be reasonable that the output be consistent with the choices
    # of spaces above. That's why I program this here
    if(dir_outs != None):

        if(tt == 0.0): # This time acts as a parameter over which we do continuation
            mode = 'w'
        else:
            mode = 'a'

        filename = dir_outs+'/s.pvd'
        w.sub(0).rename('Reference.Flux')
        VTKFile(filename, mode=mode).write(w.sub(0), time=tt)

        filename = dir_outs+'/u.pvd'
        w.sub(1).rename('Reference.Potential')
        VTKFile(filename, mode=mode).write(w.sub(1), time=tt)

        filename = dir_outs+'/ucg.pvd'
        w.sub(1).rename('Reference.Potential-CG')
        VTKFile(filename, mode=mode).write(fd.project(w.sub(1), domain.CG1, name='Potential-CG1'), time=tt)
        
        filename = dir_outs+'/gradu.pvd'
        #gpl = fd.project(g, domain.VDG0, name='Gradient(u)')
        VTKFile(filename, mode=mode).write(g, time=tt)

    return w.sub(0), w.sub(1), g

def PoissonMixedHdiv(domain, flag_quads, params, flag_setting,
                     volumetric_source, boundary_flux, boundary_vals,
                     FluxLaw, orders, tt=0.0, dir_outs='myrefsol'):

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
        #w.sub(1).interpolate(boundary_vals(x))

    elif(flag_setting == 'Pure-Neumann'):
        
        q = volumetric_source(x)
                
        # Flux bc (strong in this formulation) - empty for this setting
        bcall = [fd.DirichletBC(W.sub(0), fd.as_vector([fd.Constant(0.0), fd.Constant(0.0)]), 'on_boundary')]
        
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

    kappa = params['kappa']
    Res  = ((1.0/kappa(u))*fd.dot(sigma, tau) - fd.div(tau)*u - fd.div(sigma)*v)*fd.dx(degree=4)
    Res += q*v*fd.dx(degree=4)
    if(flag_setting == 'Pure-Dirichlet' or flag_setting == 'Uniaxial'):
        Res += pressure_boundary_term

    reac = params['reaction']
    if(reac != None):
        Res -= reac(u) * v * fd.dx(degree=4)

    fd.solve(Res == 0, w, bcs=bcall, solver_parameters=petsc_opt1)

    # Compute g in a consistent way?
    gtr = fd.TrialFunction(HD)
    tau = fd.TestFunction(HD)
    usol = fd.Function(DG)
    usol.assign(w.sub(1))
    a = fd.inner(gtr,tau)*fd.dx
    L = -usol*fd.div(tau)*fd.dx + usol*fd.dot(tau,n)*fd.ds
    g = fd.Function(HD, name='Reference.Hdiv-Gradient')
    fd.solve(a == L, g, bcs=[], solver_parameters=petsc_opt2)

    # It may be reasonable that the output be consistent with the choices
    # of spaces above. That's why I program this here
    if(dir_outs != None):

        if(tt == 0.0): # This time acts as a parameter over which we do continuation
            mode = 'w'
        else:
            mode = 'a'

        filename = dir_outs+'/s.pvd'
        w.sub(0).rename('Reference.Flux')
        VTKFile(filename, mode=mode).write(w.sub(0), time=tt)

        filename = dir_outs+'/u.pvd'
        w.sub(1).rename('Reference.Potential')
        VTKFile(filename, mode=mode).write(w.sub(1), time=tt)

        filename = dir_outs+'/ucg.pvd'
        w.sub(1).rename('Reference.Potential-CG')
        VTKFile(filename, mode=mode).write(fd.project(w.sub(1), domain.CG1, name='Potential-CG1'), time=tt)
        
        filename = dir_outs+'/gradu.pvd'
        #gpl = fd.project(g, domain.VDG0, name='Gradient(u)')
        VTKFile(filename, mode=mode).write(g, time=tt)

    return w.sub(0), w.sub(1), g

#=====================================================================================

#=====================================================================================
def PoissonMixedHdivAni(domain, flag_quads, params, flag_setting,
                        volumetric_source, boundary_flux, boundary_vals,
                        FluxLaw, orders, tt=0.0, dir_outs='myrefsol'):

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

    kappa = params['kappa'] # We expect it to be a tensor (see this better later on)
    Km1 = fd.as_tensor([[fd.Constant(1.0/kappa[0,0]), fd.Constant(0.0)], [fd.Constant(0.0), fd.Constant(1.0/kappa[1,1])]])
    Res  = (fd.dot(Km1*sigma, tau) - fd.div(tau)*u - fd.div(sigma)*v)*fd.dx(degree=4)
    Res += q*v*fd.dx(degree=4) + pressure_boundary_term

    fd.solve(Res == 0, w, bcs=bcall, solver_parameters=petsc_opt1)

    # Compute g in a consistent way?
    gtr = fd.TrialFunction(HD)
    tau = fd.TestFunction(HD)
    usol = fd.Function(DG)
    usol.assign(w.sub(1))
    a = fd.inner(gtr,tau)*fd.dx
    L = -usol*fd.div(tau)*fd.dx + usol*fd.dot(tau,n)*fd.ds
    g = fd.Function(HD, name='Reference.Hdiv-Gradient')
    fd.solve(a == L, g, bcs=[], solver_parameters=petsc_opt2)

    # It may be reasonable that the output be consistent with the choices
    # of spaces above. That's why I program this here
    if(dir_outs != None):

        if(tt == 0.0): # This time acts as a parameter over which we do continuation
            mode = 'w'
        else:
            mode = 'a'

        filename = dir_outs+'/s.pvd'
        w.sub(0).rename('Reference.Flux')
        VTKFile(filename, mode=mode).write(w.sub(0), time=tt)

        filename = dir_outs+'/u.pvd'
        w.sub(1).rename('Reference.Potential')
        VTKFile(filename, mode=mode).write(w.sub(1), time=tt)

        filename = dir_outs+'/ucg.pvd'
        w.sub(1).rename('Reference.Potential-CG')
        VTKFile(filename, mode=mode).write(fd.project(w.sub(1), domain.CG1, name='Potential-CG1'), time=tt)
        
        filename = dir_outs+'/gradu.pvd'
        #gpl = fd.project(g, domain.VDG0, name='Gradient(u)')
        VTKFile(filename, mode=mode).write(g, time=tt)

    return w.sub(0), w.sub(1), g

#=====================================================================================

def PoissonAnalytical(domain, flag_quads, params, flag_setting,
                      volumetric_source, boundary_flux, boundary_vals,
                      FluxLaw, orders, tt=0.0, dir_outs='myrefsol'):

    # This function is essentially called for visualization purposes
    # in turn, the solution that it returns is the same analytical function
    # provided as input

    # The mesh and spaces are intended to represent well the 
    # analytical solution and they will likely be quite accurate
    mesh = domain.msh
    x = fd.SpatialCoordinate(mesh)
    sexact = params['exact-flux']
    uexact = params['exact-potential']
    gradexact = fd.grad(uexact(x))
    lmbdexact = params['exact-lmbd']
    muexact = params['exact-mu']
    # etilde = params['etilde']
    # stilde = params['stilde']
    etilde = params['exact-etilde']
    stilde = params['exact-stilde']


    if(dir_outs != None):

        if(tt == 0.0): # This time acts as a parameter over which we do continuation
            mode = 'w'
        else:
            mode = 'a'

        orderF, orderU = orders
        Vflux = fd.VectorFunctionSpace(mesh, "CG", 1)
        Vpot = fd.FunctionSpace(mesh, "CG", 1)     
        
        # Add an option to interpolate or project the analytical solution
        print('--------')
        print('Poisson Analytical: This may cause problems for the singular solution at the moment of visualization')
        print('--------')

        filename = dir_outs+'/s.pvd'        
        flux = fd.assemble(interpolate(sexact(x), Vflux))
        #flux = fd.project(sexact(x), Vflux)
        flux.rename('Reference.Flux')
        VTKFile(filename, mode=mode).write(flux, time=tt)

        filename = dir_outs+'/u.pvd'
        potential = fd.assemble(interpolate(uexact(x), Vpot),)
        potential.rename('Reference.Potential')
        VTKFile(filename, mode=mode).write(potential, time=tt)
        
        filename = dir_outs+'/gradu.pvd'
        gradu = fd.assemble(interpolate(gradexact, Vflux))
        gradu.rename('Reference.Grad(u)')
        VTKFile(filename, mode=mode).write(gradu, time=tt)
        
        filename = dir_outs+'/l.pvd'
        lmbd = fd.assemble(interpolate(lmbdexact(x), Vpot))
        lmbd.rename('Reference.lambda')
        VTKFile(filename, mode=mode).write(lmbd, time=tt)

        filename = dir_outs+'/m.pvd'
        mu = fd.assemble(interpolate(muexact(x), Vflux))
        mu.rename('Reference.mu')
        VTKFile(filename, mode=mode).write(mu, time=tt)
        
        if(etilde != None):
            filename = dir_outs+'/et.pvd'
            et = fd.assemble(interpolate(etilde(x), Vflux))
            et.rename('Reference.et')
            VTKFile(filename).write(et)

        if(stilde != None):
            filename = dir_outs+'/st.pvd'
            st = fd.assemble(interpolate(stilde(x), Vflux))
            st.rename('Reference.st')
            VTKFile(filename).write(st)

        # marker_style = dict(linestyle = '', marker='o', markersize=7, markerfacecolor='w', markeredgecolor='b')
        # nve = np.linalg.norm(gradu.dat.data[:,:], axis=1)    
        # nvs = np.linalg.norm(flux.dat.data[:,:], axis=1)
        # nvet = np.linalg.norm(et.dat.data[:,:], axis=1)
        # nvst = np.linalg.norm(st.dat.data[:,:], axis=1)
        # # nve = gradu.dat.data[:,0]
        # # nvs = flux.dat.data[:,0]
        # # nvet = et.dat.data[:,0]
        # # nvst = st.dat.data[:,0]
        # plt.figure(figsize=(8,6))
        # plt.plot(nvet, nvst, **marker_style,label=r'$\tilde{\mathbf{s}}\left(\tilde{\mathbf{e}}\right)$')
        # plt.plot(nve, nvs, '.k',label=r'$\mathbf{s}\left(\mathbf{e}\right)$')
        # plt.xlabel(r'$\|\mathbf{e}\|$', fontsize=16)
        # plt.ylabel(r'$\|\mathbf{s}\|$', fontsize=16)
        # # plt.xlabel(r'$e_{x}$', fontsize=12)
        # # plt.ylabel(r'$s_{x}$', fontsize=12)
        # plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        # plt.legend(loc='best', fancybox=True, shadow=True, fontsize=12)
        # plt.tick_params(axis='both', which='major', labelsize=14)
        # plt.tick_params(axis='both', which='minor', labelsize=14)
        # plt.tight_layout()
        # plt.savefig('stilde_vs_etilde.svg', format='svg')
        # plt.show()


    return sexact, uexact, gradexact

def PoissonPrimal(domain, flag_quads, params, flag_setting,
                  volumetric_source, boundary_flux, boundary_vals,
                  FluxLaw, orders, tt=0.0, dir_outs='myrefsol'):

    # This function is essentially called for visualization purposes
    # in turn, the solution that it returns is the same analytical function
    # provided as input

    # The mesh and spaces are intended to represent well the 
    # analytical solution and they will likely be quite accurate
    mesh = domain.msh
    x = fd.SpatialCoordinate(mesh)

    orderU = orders[0]
    CG = fd.FunctionSpace(mesh, "CG", orderU)

    u = fd.Function(CG, name='Solution')
    v = fd.TestFunction(CG)
    n = fd.FacetNormal(mesh)

    difus = params['difus']
    Res  = (difus(u))*fd.dot(fd.grad(u), fd.grad(v))*fd.dx(degree=4)

    if(flag_setting == 'Pure-Dirichlet'):
        
        q = volumetric_source(x)
        
        # Flux bc (strong in this formulation) - empty for this setting
        bcall = [fd.DirichletBC(CG, boundary_vals(x), 'on_boundary')]

    elif(flag_setting == 'Pure-Neumann'):
        
        q = volumetric_source(x)
        Res += boundary_flux(x) * v * fd.ds(degree=4)
        bcall = []
    
    else:
        
        raise ValueError('Unrecognized problem setting option')

    Res -= q*v*fd.dx(degree=4)

    reac = params['reaction']
    if(reac != None):
        Res += reac(u) * v * fd.dx(degree=4)

    fd.solve(Res == 0, u, bcs=bcall, solver_parameters=petsc_opt1)
    s = -difus(u)*fd.grad(u)
    g = fd.grad(u)
    
    if(dir_outs != None):

        if(tt == 0.0): # This time acts as a parameter over which we do continuation
            mode = 'w'
        else:
            mode = 'a'

        filename = dir_outs+'/flux.pvd'
        spl = fd.project(s, domain.VDG0, name='Reference.Flux')
        VTKFile(filename, mode=mode).write(spl, time=tt)

        filename = dir_outs+'/fluxCG1.pvd'
        splc = fd.project(s, domain.VCG1, name='Reference.Flux')
        VTKFile(filename, mode=mode).write(splc, time=tt)

        filename = dir_outs+'/u.pvd'
        u.rename('Reference.Potential')
        VTKFile(filename, mode=mode).write(u, time=tt)
        
        filename = dir_outs+'/gradu.pvd'
        gpl = fd.project(g, domain.VDG0, name='Reference.Grad(u)')
        VTKFile(filename, mode=mode).write(gpl, time=tt)

    return s, u, g

#=====================================================================================
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
# the Mechanical solvers below
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
#----------------------------------------------------------------
#
# Program your Stress-Strain Law: (Primal or Mixed): 
# Follow format for the names: SSLX_mymodel() (where X can be P: primal or M: mixed)
#
def SSLP_LinearElas(displ, Lame):
    mu = Lame['mu']
    lmd = Lame['lmd']
    return (2*mu*fd.sym(fd.grad(displ)) + lmd*fd.div(displ)*fd.Identity(2))

def SSLM_LinearElas(displ, stress, params):
    mu = params['mu']
    lmd = params['lmd']
    return stress - (2*mu*fd.sym(fd.grad(displ)) + lmd*fd.div(displ)*fd.Identity(2))

#-------------------------------------------------------------------------------------
# Program your FEM formulation (Mixed or Primal):
# Use representative names (e.g. MechanicalPrimalCG(), etc.)
#-------------------------------------------------------------------------------------

#=====================================================================================

def MechanicalPrimalCG(domain, flag_quads, params, flag_setting,
                       body_force, boundary_force, boundary_vals, boundary_tags,
                       StressLaw, orders, tt=0.0, dir_outs='myrefsol'):
    
    mesh = domain.msh

    orderU = orders[0]
    W = fd.VectorFunctionSpace(mesh, "CG", orderU)

    u = fd.Function(W, name='Solution')
    v = fd.TestFunction(W)

    EY = params['YoungMod']
    poi = params['PoissonRat']
    mu  = fd.Constant(EY/(2.0*(1.0 + poi)))
    lmd = fd.Constant(EY*poi/((1.0 + poi)*(1.0 - 2.0*poi)))
    Lame = {'mu': mu, 'lmd': lmd}

    x = fd.SpatialCoordinate(mesh)
    n = fd.FacetNormal(mesh)
    
    sigma = StressLaw(u, Lame)
    Res = fd.inner(sigma, fd.grad(v))*fd.dx

    if(flag_setting == 'Indent-Block'):
        # By definition an indentation problem corresponds to:
        # x restricted in the left, y restricted in the bottom
        # No force on the right and a force distribution on the top
        
        f = body_force(x)
        Res -= fd.inner(f, v)*fd.dx

        tag_left = boundary_tags['Left']
        tag_bottom = boundary_tags['Bottom']
        bcall = [fd.DirichletBC(W.sub(0), fd.Constant(0.0), tag_left), 
                 fd.DirichletBC(W.sub(1), fd.Constant(0.0), tag_bottom)]
        
        fs = boundary_force['Top'](x)
        tag_top = boundary_tags['Top']
        Res -= fd.inner(fs, v)*fd.ds(tag_top)

    elif(flag_setting == 'Uniaxial'):
        
        # By definition, the uniaxial test consists of
        # Bottom wall restricted, top wall imposed vertical
        # displacement and free lateral walls:
        f = fd.as_vector([fd.Constant(0.0), fd.Constant(0.0)])
        Res -= fd.inner(f, v)*fd.dx

        uybottom, uytop = fd.Constant(0.0), boundary_vals(x)
        tag_top, tag_bottom = boundary_tags['Top'], boundary_tags['Bottom']
        bcall = [fd.DirichletBC(W.sub(1), uybottom, tag_bottom), 
                 fd.DirichletBC(W.sub(1), uytop, tag_top)]

    else:
        
        raise ValueError('Unrecognized problem setting option')

    fd.solve(Res == 0, u, bcs=bcall, solver_parameters=petsc_opt1)
    g = fd.sym(fd.grad(u))
    sigma = StressLaw(u, Lame)

    # It may be reasonable that the output be consistent with the choices
    # of spaces above. That's why I program this here 
    # e.g., if u \in DG1, g may belong to DG0, but if u \in DG0 then we should 
    # integrate by parts during projection??
    if(dir_outs != None):

        if(tt == 0.0): # This time acts as a parameter over which we do continuation
            mode = 'w'
        else:
            mode = 'a'

        filename = dir_outs+'/stress.pvd'
        sigmapl = fd.project(sigma, domain.TDG0, name='Reference.Stress')
        VTKFile(filename, mode=mode).write(sigmapl, time=tt)

        filename = dir_outs+'/u.pvd'
        u.rename('Reference.Displacement')
        VTKFile(filename, mode=mode).write(u, time=tt)
        
        filename = dir_outs+'/gradu.pvd'
        gpl = fd.project(g, domain.TDG0, name='Reference.SymGrad(u)')
        VTKFile(filename, mode=mode).write(gpl, time=tt)

    return sigma, u, g

    return 

import firedrake as fd

from ddcm_classes import *
#
#----------------------------------------------------------------
#
# This file contains user defined utility functions to set your
# DDCM Poisson model, including numerical parameters, 
# ddcm formulations, etc.
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

def NLSP5FA(domain, ObjFunc, der_ObjFunc, graddata, fluxdata, flag_quads,
            flag_setting, volumetric_source, boundary_flux, boundary_vals,
            orders, dir_outs):

    mesh = domain.msh
    
    orderF, orderU = orders

    DGtype = 'DG'
    if(flag_quads):
        DGtype = 'DQ'
    CG  = fd.FunctionSpace(mesh, family="CG", degree=orderU)
    VDG = fd.VectorFunctionSpace(mesh, family=DGtype, degree=orderF)

    # Mixed space: In this formulation we assume all scalar fields 
    # are approximated in the same space, the same for all vector fields 
    #  flux    u    g  lambda  mu
    W = VDG * CG * VDG * CG * VDG

    x = fd.SpatialCoordinate(mesh)

    # Unknown field
    w = fd.Function(W)
    q, u, g, l, m = fd.split(w)

    # Test functions
    dq, du, dg, dl, dm = fd.TestFunctions(W)

    # Functional to be minimized
    J = ObjFunc(g, graddata, q, fluxdata)

    # First variations
    if(der_ObjFunc == None):
        dqJ = fd.derivative(J, q, dq)
        dgJ = fd.derivative(J, g, dg)
    
    # Residual
    dL  =  dqJ + dgJ
    dL +=  -inner(dq, grad(l)) * dx
    dL += (-inner(q, grad(dl)) - volumetric_source(x) * dl ) * dx
    dL += inner(g-grad(u), dm) * dx
    dL += inner(dg, m) * dx - inner(m, grad(du)) * dx

    # Jacobian
    Jac = fd.derivative(dL, w)

    if(flag_setting == 'Pure-Dirichlet'):
        # Boundary condition for potential field
        bcu = fd.DirichletBC(W.sub(1), boundary_vals(x), "on_boundary")

        # Homogeneous boundary condition for scalar Lagrange multiplier
        bcl = fd.DirichletBC(W.sub(3), fd.Constant(0.0), "on_boundary")

    else:
        
        raise ValueError('Unrecognized problem setting option')

    # Non-linear solve (see solver parameters above)
    fd.solve(dL == 0, w, bcs=[bcu, bcl], J=Jac, solver_parameters=petsc_opt1)
    
    if(dir_outs != None):

        # Split the solution for post-processing
        #qpl, ppl, gpl, lpl, mpl = split(w)

        filename = dir_outs+'/s.pvd'
        w.sub(0).rename('DDCM.Flux')
        VTKFile(filename).write(w.sub(0))

        filename = dir_outs+'/u.pvd'
        w.sub(1).rename('DDCM.Potential')
        VTKFile(filename).write(w.sub(1))
        
        filename = dir_outs+'/g.pvd'
        w.sub(2).rename('DDCM.gradient(u)')
        VTKFile(filename).write(w.sub(2))

        filename = dir_outs+'/l.pvd'
        w.sub(3).rename('DDCM.lambda')
        VTKFile(filename).write(w.sub(3))

        filename = dir_outs+'/m.pvd'
        w.sub(3).rename('DDCM.mu')
        VTKFile(filename).write(w.sub(4))
    
    return w.sub(0), w.sub(1), w.sub(2), w.sub(3), w.sub(4)
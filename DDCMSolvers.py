import firedrake as fd
from firedrake.output import VTKFile

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

def DDCMStab(domain, ObjFunc, der_ObjFunc, graddata, fluxdata, flag_quads,
             flag_setting, volumetric_source, boundary_flux, boundary_vals,
             params, orders, dir_outs):

    mesh = domain.msh
    
    orderS, orderU = orders

    formulation = params['formulation']
    if(formulation == 'Natural'):
        DGtype = 'DG'
        if(flag_quads):
            DGtype = 'DQ'
        SF = fd.FunctionSpace(mesh, family="CG", degree=orderU)
        VF =  fd.VectorFunctionSpace(mesh, family=DGtype, degree=orderS)
    elif(formulation == 'Equal-Order'):
        SF = fd.FunctionSpace(mesh, family="CG", degree=orderU)
        VF = fd.VectorFunctionSpace(mesh, family="CG", degree=orderS)
    else:
        raise ValueError('Formulation not recognized in DDCMStab')

    # Mixed space: 
    # In this formulation we assume all scalar fields are approximated 
    # in the same space SF and all vector fields in the same space VF
    #   s    u    e    lam  mu
    W = VF * SF * VF * SF * VF

    x = fd.SpatialCoordinate(mesh)

    # Unknown field
    w = fd.Function(W)
    s, u, e, lam, mu = fd.split(w)

    # Test function
    vw = fd.TestFunction(W)

    # Functional to be minimized
    J = ObjFunc(e, graddata, s, fluxdata)

    zeda  = params['zeda']
    theta = params['theta']
    kappa = params['kappa']
    alpha = params['alpha']
    beta  = params['beta']
    gamma = params['gamma']
    eta  = params['eta']
    ls    = params['ls']
    lmu   = params['lmu']
    f     = volumetric_source

    # Lagrangian:
    # Proximity function
    L  = J

    # Balance and Compatibility constraint
    L -= fd.inner(s, fd.grad(lam)) * fd.dx
    L += fd.inner(zeda*u - f(x), lam) * fd.dx(degree=4)
    L += fd.inner(e - fd.grad(u), mu) * fd.dx

    # Stabilization terms
    resC = e - fd.grad(u)
    L += 0.5*alpha*fd.inner(resC,resC) * fd.dx

    resE = e + mu - graddata
    L  -= 0.5*gamma*fd.inner(resE,resE) * fd.dx

    resK = kappa*s - fd.grad(lam) - kappa*fluxdata
    L  -= (0.5*eta/kappa)*fd.inner(resK,resK) * fd.dx

    resB = fd.div(s) + zeda*u - f(x)
    L += (0.5*theta*kappa*ls**2)*fd.inner(resB,resB) * fd.dx

    resM = fd.div(mu) + zeda*lam
    L -= (0.5*beta*lmu**2)*fd.inner(resM,resM) * fd.dx

    bc_lst = []
    if(flag_setting == 'Pure-Dirichlet'):
        # Boundary condition for potential field
        bc_lst.append(fd.DirichletBC(W.sub(1), boundary_vals(x), "on_boundary"))

        # Homogeneous boundary condition for scalar Lagrange multiplier
        bc_lst.append(fd.DirichletBC(W.sub(3), fd.Constant(0.0), "on_boundary"))

    elif(flag_setting == 'Pure-Neumann'):

        raise ValueError('Neumann conditions should be include in the Lagrangian')
        #dL += boundary_flux(x) * dl * fd.ds(degree=4)
        
    else:
        
        raise ValueError('Unrecognized problem setting option')

    # Optimallity conditions
    dL = fd.derivative(L, w, vw)

    # Jacobian
    Jac = fd.derivative(dL, w)

    # Non-linear solve (see solver parameters above)
    fd.solve(dL == 0, w, bcs=bc_lst, J=Jac, solver_parameters=petsc_opt1)
    
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
        w.sub(4).rename('DDCM.mu')
        VTKFile(filename).write(w.sub(4))
    
    return w.sub(0), w.sub(1), w.sub(2), w.sub(3), w.sub(4)


def NLSP5FA(domain, ObjFunc, der_ObjFunc, graddata, fluxdata, flag_quads,
            flag_setting, volumetric_source, boundary_flux, boundary_vals,
            params, orders, dir_outs):

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
    dL += -fd.inner(dq, fd.grad(l)) * fd.dx
    dL +=(-fd.inner(q, fd.grad(dl)) - volumetric_source(x) * dl ) * fd.dx
    dL += (fd.inner(g, dm) - fd.inner(fd.grad(u), dm))* fd.dx
    dL += (fd.inner(dg, m) - fd.inner(fd.grad(du), m)) * fd.dx

    bc_lst = []
    if(flag_setting == 'Pure-Dirichlet'):
        # Boundary condition for potential field
        bc_lst.append(fd.DirichletBC(W.sub(1), boundary_vals(x), "on_boundary"))

        # Homogeneous boundary condition for scalar Lagrange multiplier
        bc_lst.append(fd.DirichletBC(W.sub(3), fd.Constant(0.0), "on_boundary"))

    elif(flag_setting == 'Pure-Neumann'):

        dL += boundary_flux(x) * dl * fd.ds(degree=4)
        bc_lst = []

    else:
        
        raise ValueError('Unrecognized problem setting option')
    
    reac = params['reaction']
    if(reac != None):
        dreac = params['dreaction']
        dL += reac(u) * dl * fd.dx
        dL += dreac(u) * du * l * fd.dx

    # Jacobian
    Jac = fd.derivative(dL, w)

    # Non-linear solve (see solver parameters above)
    fd.solve(dL == 0, w, bcs=bc_lst, J=Jac, solver_parameters=petsc_opt1)
    
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
        w.sub(4).rename('DDCM.mu')
        VTKFile(filename).write(w.sub(4))
    
    return w.sub(0), w.sub(1), w.sub(2), w.sub(3), w.sub(4)


def NLSP5FA_EQUOR(domain, ObjFunc, der_ObjFunc, graddata, fluxdata, flag_quads,
                  flag_setting, volumetric_source, boundary_flux, boundary_vals,
                  params, orders, dir_outs):

    mesh = domain.msh
    
    orderF, orderU = orders

    CG  = fd.FunctionSpace(mesh, family="CG", degree=orderU)
    VCG = fd.VectorFunctionSpace(mesh, family="CG", degree=orderF)

    # Mixed space: In this formulation we assume all scalar fields 
    # are approximated in the same space, the same for all vector fields 
    #  flux    u    g  lambda  mu
    W = VCG * CG * VCG * CG * VCG

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
    dL += -fd.inner(dq, fd.grad(l)) * fd.dx
    dL +=(-fd.inner(q, fd.grad(dl)) - volumetric_source(x) * dl) * fd.dx(degree=4)
    dL +=  fd.inner(g-fd.grad(u), dm) * fd.dx
    dL += (fd.inner(dg, m) - fd.inner(fd.grad(du), m)) * fd.dx

    # Jacobian
    Jac = fd.derivative(dL, w)

    bc_lst = []
    if(flag_setting == 'Pure-Dirichlet'):
        # Boundary condition for potential field
        bc_lst.append(fd.DirichletBC(W.sub(1), boundary_vals(x), "on_boundary"))

        # Homogeneous boundary condition for scalar Lagrange multiplier
        bc_lst.append(fd.DirichletBC(W.sub(3), fd.Constant(0.0), "on_boundary"))

    else:
        
        raise ValueError('Unrecognized problem setting option')

    # Non-linear solve (see solver parameters above)
    fd.solve(dL == 0, w, bcs=bc_lst, J=Jac, solver_parameters=petsc_opt1)
    
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
        w.sub(4).rename('DDCM.mu')
        VTKFile(filename).write(w.sub(4))
    
    return w.sub(0), w.sub(1), w.sub(2), w.sub(3), w.sub(4)

def NLSP5FB(domain, ObjFunc, der_ObjFunc, graddata, fluxdata, flag_quads,
            flag_setting, volumetric_source, boundary_flux, boundary_vals,
            params, orders, dir_outs):

    mesh = domain.msh
    
    orderF, orderU = orders

    RTtype, DGtype = 'RT', 'DG'
    if(flag_quads):
        RTtype, DGtype = "RTCF", 'DQ'

    Hdiv  = fd.FunctionSpace(mesh, family=RTtype, degree=orderF)
    DG = fd.FunctionSpace(mesh, family=DGtype, degree=orderU)

    # Mixed space: In this formulation we assume all scalar fields 
    # are approximated in the same space, the same for all vector fields
    # It makes sense that g \in Hdiv, since it is the Hdiv projection of grad(u)?
    #   flux   u     g   lambda  mu
    W = Hdiv * DG * Hdiv * DG * Hdiv

    x = fd.SpatialCoordinate(mesh)
    n = fd.FacetNormal(mesh)

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
    dL +=  fd.inner(fd.div(dq), l) * fd.dx
    dL += (fd.inner(fd.div(q), dl) - volumetric_source(x) * dl) * fd.dx
    dL += (fd.inner(dg, m) + fd.inner(du, fd.div(m))) * fd.dx
    dL += (fd.inner(dm, g) + fd.inner(u, fd.div(dm))) * fd.dx

    # Jacobian
    Jac = fd.derivative(dL, w)

    bc_lst = []
    if(flag_setting == 'Pure-Dirichlet'):
        # Boundary condition for potential field
        dL += -boundary_vals(x) * fd.inner(dm,n) * fd.ds

    else:
        
        raise ValueError('Unrecognized problem setting option')

    # Non-linear solve (see solver parameters above)
    fd.solve(dL == 0, w, bcs=bc_lst, J=Jac, solver_parameters=petsc_opt1)
    
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
        w.sub(4).rename('DDCM.mu')
        VTKFile(filename).write(w.sub(4))
    
    return w.sub(0), w.sub(1), w.sub(2), w.sub(3), w.sub(4)

def NLSP5FC(domain, ObjFunc, der_ObjFunc, graddata, fluxdata, flag_quads,
            flag_setting, volumetric_source, boundary_flux, boundary_vals,
            params, orders, dir_outs):

    mesh = domain.msh
    
    orderF, orderU = orders

    RTtype, DGtype = 'RT', 'DG'
    if(flag_quads):
        RTtype, DGtype = "RTCF", 'DQ'

    Hdiv = fd.FunctionSpace(mesh, family=RTtype, degree=orderF)
    VDG = fd.VectorFunctionSpace(mesh, family=DGtype, degree=orderF-1)
    DG = fd.FunctionSpace(mesh, family=DGtype, degree=orderU)
    CG = fd.FunctionSpace(mesh, family='CG', degree=orderU+1)

    # Mixed space: In this formulation we assume all scalar fields 
    # are approximated in the same space, the same for all vector fields 
    #  flux    u    g   lambda  mu
    W = VDG * DG * Hdiv * CG * Hdiv

    x = fd.SpatialCoordinate(mesh)
    n = fd.FacetNormal(mesh)

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
    dL += -fd.inner(dq, fd.grad(l)) * fd.dx
    dL +=  fd.inner(du, fd.div(m)) * fd.dx
    dL +=  fd.inner(dg, m) * fd.dx
    dL +=(-fd.inner(q, fd.grad(dl)) - volumetric_source(x) * dl) * fd.dx(degree=4)
    dL += (fd.inner(g, dm) + fd.inner(u, fd.div(dm))) * fd.dx

    # Jacobian
    Jac = fd.derivative(dL, w)

    bc_lst = []
    if(flag_setting == 'Pure-Dirichlet'):
        # Boundary condition for potential field
        dL += -boundary_vals(x) * fd.inner(dm,n) * fd.ds(degree=4)

        # Homogeneous boundary condition for scalar Lagrange multiplier
        bc_lst.append(fd.DirichletBC(W.sub(3), fd.Constant(0.0), "on_boundary"))

    else:
        
        raise ValueError('Unrecognized problem setting option')

    # Non-linear solve (see solver parameters above)
    fd.solve(dL == 0, w, bcs=bc_lst, J=Jac, solver_parameters=petsc_opt1)
    
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
        w.sub(4).rename('DDCM.mu')
        VTKFile(filename).write(w.sub(4))
    
    return w.sub(0), w.sub(1), w.sub(2), w.sub(3), w.sub(4)

def NLSP5FD(domain, ObjFunc, der_ObjFunc, graddata, fluxdata, flag_quads,
            flag_setting, volumetric_source, boundary_flux, boundary_vals,
            params, orders, dir_outs):

    mesh = domain.msh
    
    orderF, orderU = orders

    RTtype, DGtype = 'RT', 'DG'
    if(flag_quads):
        RTtype, DGtype = "RTCF", 'DQ'

    Hdiv = fd.FunctionSpace(mesh, family=RTtype, degree=orderF)
    VDG = fd.VectorFunctionSpace(mesh, family=DGtype, degree=orderF-1)
    DG = fd.FunctionSpace(mesh, family=DGtype, degree=orderU)
    CG = fd.FunctionSpace(mesh, family='CG', degree=orderU+1)

    # Mixed space: In this formulation we assume all scalar fields 
    # are approximated in the same space, the same for all vector fields 
    #   flux    u    g  lambda  mu
    W = Hdiv * CG * VDG * DG * VDG

    x = fd.SpatialCoordinate(mesh)
    n = fd.FacetNormal(mesh)

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
    dL +=  fd.inner(fd.div(dq),l) * fd.dx
    dL += -fd.inner(fd.grad(du),m) * fd.dx
    dL +=  fd.inner(dg, m) * fd.dx
    dL += (fd.inner(fd.div(q), dl) - volumetric_source(x) * dl ) * fd.dx(degree=4)
    dL += (fd.inner(g, dm) -  fd.inner(fd.grad(u), dm)) * fd.dx

    # Jacobian
    Jac = fd.derivative(dL, w)

    bc_lst = []
    if(flag_setting == 'Pure-Dirichlet'):
        # Boundary condition for potential field
        bc_lst.append(fd.DirichletBC(W.sub(1), boundary_vals(x), "on_boundary"))

    else:
        
        raise ValueError('Unrecognized problem setting option')

    # Non-linear solve (see solver parameters above)
    fd.solve(dL == 0, w, bcs=bc_lst, J=Jac, solver_parameters=petsc_opt1)
    
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
        w.sub(4).rename('DDCM.mu')
        VTKFile(filename).write(w.sub(4))
    
    return w.sub(0), w.sub(1), w.sub(2), w.sub(3), w.sub(4)


def NLSP5FA_MECH(domain, ObjFunc, der_ObjFunc, symgraddata, stressdata, flag_quads,
                 flag_setting, body_force, boundary_force, boundary_vals, boundary_tags,
                 params, orders, dir_outs):
    
    mesh = domain.msh
    
    orderS, orderU = orders

    DGtype = 'DG'
    if(flag_quads):
        DGtype = 'DQ'
    VCG = fd.VectorFunctionSpace(mesh, family="CG", degree=orderU)
    TDG = fd.TensorFunctionSpace(mesh, family=DGtype, degree=orderS)

    # Mixed space: In this formulation we assume all scalar fields 
    # are approximated in the same space, the same for all vector fields 
    #  stress  u  strain lambda  mu
    W = TDG * VCG * TDG * VCG * TDG

    x = fd.SpatialCoordinate(mesh)

    n = fd.FacetNormal(mesh)

    # Unknown field
    w = fd.Function(W)
    si, u, g, l, m = fd.split(w)

    # Test functions
    dsi, du, dg, dl, dm = fd.TestFunctions(W)

    # Functional to be minimized
    J = ObjFunc(g, symgraddata, si, stressdata)

    # First variations
    if(der_ObjFunc == None):
        dgJ = fd.derivative(J, g, dg)
        dsiJ = fd.derivative(J, si, dsi)        
    else:
        aux = der_ObjFunc(g, symgraddata, dg, si, stressdata, dsi)
        dgJ  = aux[0]
        dsiJ = aux[1]

    # Residual
    NL_model = 'SmallDef'
    if(NL_model == 'SmallDef'):
        Hofu = fd.sym(fd.grad(u))
        duHofu = fd.sym(fd.grad(du))
        hoflu = fd.sym(fd.grad(l))
        duhoflu = fd.as_tensor([[fd.Constant(0.0), fd.Constant(0.0)], 
                                [fd.Constant(0.0), fd.Constant(0.0)]])
        dlhoflu = fd.sym(fd.grad(dl))
    elif(NL_model == 'LargeDef'):
        raise ValueError('Large deformations not implemented yet')

    dsiL =  fd.inner(dsi, hoflu) * fd.dx
    duL  = -fd.inner(m, duHofu) * fd.dx # -fd.inner(si, duhoflu) * fd.dx 
    dgL  =  fd.inner(dg, m) * fd.dx
    dlL  =  fd.inner(si, dlhoflu) * fd.dx # check this sign
    dmL  =  fd.inner((g - Hofu), dm) * fd.dx

    # Residual, put all together 
    dL = dsiJ + dgJ + dsiL + duL + dgL + dlL + dmL

    bc_lst = []
    if(flag_setting == 'Indent-Block'):
        # By definition an indentation problem corresponds to:
        # x restricted in the left, y restricted in the bottom
        # No force on the right and a force distribution on the top

        tag_left = boundary_tags['Left']
        tag_bottom = boundary_tags['Bottom']
        tag_top = boundary_tags['Top']
        tag_right = boundary_tags['Right']

        fb = body_force(x)
        dL -= fd.inner(fb, dl) * fd.dx # check this sign
        
        fs = boundary_force['Top'](x)
        #dL += fd.inner(fd.dot(dsi,n),l) * fd.ds(tag_top)
        #dL -= fd.inner(fd.dot(si, n) - fs, dl) * fd.ds(tag_top, degree=4)
        dL -= fd.inner(fs, dl) * fd.ds(tag_top, degree=4)

        bc_lst = [fd.DirichletBC(W.sub(1).sub(0), fd.Constant(0.0), tag_left), 
                  fd.DirichletBC(W.sub(1).sub(1), fd.Constant(0.0), tag_bottom),
                  fd.DirichletBC(W.sub(3).sub(0), fd.Constant(0.0), tag_left), 
                  fd.DirichletBC(W.sub(3).sub(1), fd.Constant(0.0), tag_bottom)
                  ] 
    elif(flag_setting == 'Stretch-Block'):
        # By definition an indentation problem corresponds to:
        # x restricted in the left, y restricted in the bottom
        # and stretching displacements on the other sides

        tag_left = boundary_tags['Left']
        tag_bottom = boundary_tags['Bottom']
        tag_top = boundary_tags['Top']
        tag_right = boundary_tags['Right']

        fb = body_force(x)
        dL -= fd.inner(fb, dl) * fd.dx # check this sign
        
        #dL += fd.inner(fd.dot(dsi,n),l) * fd.ds(tag_top)
        #dL -= fd.inner(fd.dot(si, n) - fs, dl) * fd.ds(tag_top, degree=4)

        bc_lst = [fd.DirichletBC(W.sub(1).sub(0), fd.Constant(0.0), tag_left), 
                  fd.DirichletBC(W.sub(1).sub(1), fd.Constant(0.0), tag_bottom),
                  fd.DirichletBC(W.sub(1).sub(0), fd.Constant(0.1), tag_right), 
                  fd.DirichletBC(W.sub(1).sub(1), fd.Constant(0.1), tag_top), 
                  fd.DirichletBC(W.sub(3).sub(0), fd.Constant(0.0), tag_left), 
                  fd.DirichletBC(W.sub(3).sub(1), fd.Constant(0.0), tag_bottom),
                  fd.DirichletBC(W.sub(3).sub(0), fd.Constant(0.0), tag_right), 
                  fd.DirichletBC(W.sub(3).sub(1), fd.Constant(0.0), tag_top)
                  ] 

    elif(flag_setting == 'Beam'):
        # Restricted at both ends

        tag_left = boundary_tags['Left']
        tag_top = boundary_tags['Top']
        tag_right = boundary_tags['Right']

        fb = body_force(x)
        dL += fd.inner(fb, dl) * fd.dx
        
        fs = boundary_force['Top'](x)
        dL += fd.inner(fd.dot(dsi,n),l) * fd.ds(tag_top)
        dL -= fd.inner(fd.dot(si, n) - fs, dl) * fd.ds(tag_top, degree=4)

        bc_lst = [fd.DirichletBC(W.sub(1), fd.as_vector([fd.Constant(0.0), fd.Constant(0.0)]), tag_left), 
                  fd.DirichletBC(W.sub(1), fd.as_vector([fd.Constant(0.0), fd.Constant(0.0)]), tag_right),
                  fd.DirichletBC(W.sub(3), fd.as_vector([fd.Constant(0.0), fd.Constant(0.0)]), tag_left), 
                  fd.DirichletBC(W.sub(3), fd.as_vector([fd.Constant(0.0), fd.Constant(0.0)]), tag_right)
                  ]

    else:
        raise ValueError('Setting option not recognized')
    
    # Jacobian
    Jac = fd.derivative(dL, w)

    # Non-linear solve (see solver parameters above)
    fd.solve(dL == 0, w, bcs=bc_lst, J=Jac, solver_parameters=petsc_opt1)
    
    if(dir_outs != None):

        # Split the solution for post-processing
        #qpl, ppl, gpl, lpl, mpl = split(w)

        filename = dir_outs+'/s.pvd'
        w.sub(0).rename('DDCM.Stress')
        VTKFile(filename).write(w.sub(0))

        filename = dir_outs+'/u.pvd'
        w.sub(1).rename('DDCM.Displ')
        VTKFile(filename).write(w.sub(1))
        
        filename = dir_outs+'/g.pvd'
        w.sub(2).rename('DDCM.SymGrad(u)')
        VTKFile(filename).write(w.sub(2))

        filename = dir_outs+'/l.pvd'
        w.sub(3).rename('DDCM.lambda')
        VTKFile(filename).write(w.sub(3))

        filename = dir_outs+'/m.pvd'
        w.sub(4).rename('DDCM.mu')
        VTKFile(filename).write(w.sub(4))

    return w.sub(0), w.sub(1), w.sub(2), w.sub(3), w.sub(4)
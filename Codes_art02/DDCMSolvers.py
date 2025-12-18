import firedrake as fd
import numpy as np
#from firedrake.output import VTKFile
from ddcm_classes import *

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
    'snes_max_it': 2,
    'snes_divergence_tolerance': 1e20,
    'snes_rtol': 1e-3,
    'snes_atol': 1e-3,
    'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
}

def DDCMStab(domain, ObjFunc, der_ObjFunc, graddata, fluxdata, flag_quads,
             flag_setting, volumetric_source, s_boundary_vals, u_boundary_vals,
             lmbd_boundary_vals, mu_boundary_vals, dual_volumetric_source,
             params, orders, tt=0.0, dir_outs='myddcmsol'):
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
    n = fd.FacetNormal(mesh)

    # Unknown field
    w = fd.Function(W)
    s, u, e, lam, mu = fd.split(w)

    # Test function
    vw = fd.TestFunction(W)

    # Functional to be minimized
    J = ObjFunc(e, graddata, s, fluxdata)

    zeta  = params['zeta']
    theta = params['theta']
    kappa = params['kappa']
    alpha = params['alpha']
    beta  = params['beta']
    gamma = params['gamma']
    eta   = params['eta']
    ls    = params['ls']
    lmu   = params['lmu']
    q     = volumetric_source
    f     = dual_volumetric_source
    # Lagrangian:
    # Proximity function
    L = J

    # Balance and Compatibility constraint
    L -= fd.inner(s, fd.grad(lam)) * fd.dx
    L += fd.inner(zeta*u - q(x), lam) * fd.dx
    L += fd.inner(e - fd.grad(u), mu) * fd.dx
    
    # Stabilization terms
    resC = e - fd.grad(u)
    L += 0.5*alpha*fd.inner(resC,resC) * fd.dx

    resE = e + mu - graddata
    L  -= 0.5*gamma*fd.inner(resE,resE) * fd.dx

    resK = kappa*s - fd.grad(lam) - kappa*fluxdata
    L  -= (0.5*eta/kappa)*fd.inner(resK,resK) * fd.dx

    resB = fd.div(s) + zeta*u - q(x)
    L += (0.5*theta*kappa*ls**2)*fd.inner(resB,resB) * fd.dx

    resM = fd.div(mu) + zeta*lam - f(x)
    L -= (0.5*beta*lmu**2)*fd.inner(resM,resM) * fd.dx
    
    L -= u*f(x)*fd.dx
    
    bc_lst = []
    if(flag_setting == 'Pure-Dirichlet'):
        
        # Boundary condition for potential field
        bc_lst.append(fd.DirichletBC(W.sub(1), u_boundary_vals(x), "on_boundary"))
        # Homogeneous boundary condition for scalar Lagrange multiplier
        bc_lst.append(fd.DirichletBC(W.sub(3), lmbd_boundary_vals(x), "on_boundary"))

    elif(flag_setting == 'Pure-Neumann'):
        
        L += fd.inner(s_boundary_vals(x),n)*lam*fd.ds
        L -= fd.inner(mu_boundary_vals(x),n)*u*fd.ds
        
    elif(flag_setting == 'Mixed-Dirichlet-Neumann'):
        
        # Dirichlet boundary conditions for potential field on left and right
        bc_lst.append(fd.DirichletBC(W.sub(1), u_boundary_vals(x), 1))
        bc_lst.append(fd.DirichletBC(W.sub(1), u_boundary_vals(x), 2))
        
        bc_lst.append(fd.DirichletBC(W.sub(3), lmbd_boundary_vals(x), 1))
        bc_lst.append(fd.DirichletBC(W.sub(3), lmbd_boundary_vals(x), 2))

        # bottom and top
        # L += fd.inner(s_boundary_vals(x),n)*lam*(fd.ds(3)+fd.ds(4))
        # L -= fd.inner(mu_boundary_vals(x),n)*u*(fd.ds(3)+fd.ds(4))
        
    else:
        
        raise ValueError('Unrecognized problem setting option')

    # Optimallity conditions
    dL = fd.derivative(L, w, vw)

    # Jacobian
    Jac = fd.derivative(dL, w)

    # Non-linear solve (see solver parameters above)
    fd.solve(dL == 0, w, bcs=bc_lst, J=Jac, solver_parameters=petsc_opt1)

    if(dir_outs != None):

        if(tt == 0.0): # This time acts as a parameter over which we do continuation
            mode = 'w'
        else:
            mode = 'a'

        filename = dir_outs+'/s.pvd'
        w.sub(0).rename('DDCM.Flux')
        VTKFile(filename, mode=mode).write(w.sub(0), time=tt)

        filename = dir_outs+'/u.pvd'
        w.sub(1).rename('DDCM.Potential')
        VTKFile(filename, mode=mode).write(w.sub(1), time=tt)
        
        filename = dir_outs+'/g.pvd'
        w.sub(2).rename('DDCM.gradient(u)')
        VTKFile(filename, mode=mode).write(w.sub(2), time=tt)

        filename = dir_outs+'/l.pvd'
        w.sub(3).rename('DDCM.lambda')
        VTKFile(filename, mode=mode).write(w.sub(3), time=tt)

        filename = dir_outs+'/m.pvd'
        w.sub(4).rename('DDCM.mu')
        VTKFile(filename, mode=mode).write(w.sub(4), time=tt)

    return w.sub(0), w.sub(1), w.sub(2), w.sub(3), w.sub(4)


def DDCMStabCod(domain, ObjFunc, der_ObjFunc, graddata, fluxdata, flag_quads,
                flag_setting, volumetric_source, s_boundary_vals, u_boundary_vals,
                lmbd_boundary_vals, mu_boundary_vals, dual_volumetric_source,
                params, orders, tt=0.0, dir_outs='myddcmsol'):

    # Define the mesh
    mesh = domain.msh

    # Extract the orders
    orderS, orderU = orders

    # Get the formulation we want to employ and the projection we want to consider
    formulation = params['formulation']
    Pprime = params['Pprime']
    print('------')
    print(formulation, Pprime)
    print('------')

    # Set the function spaces we have to consider according to the formulation
    """
    P' = I: proceed with a five-field formulation, since we can directly substitute the sub-grid scales into the 
    stabilized weak formulation. This method is called algebraic VMS;
    P' = L2 or OSS: consider the L2 orthogonal projection of the residuals onto the same function spaces selected 
    for the primal fields. We end up with a ten-field formulation, which is solved monolithically.            
    """
    if formulation == 'VMS-Primal' or formulation == 'VMS-Dual':
        if Pprime == 'Identity':
            SF = fd.FunctionSpace(mesh, family="CG", degree=orderU)
            VF = fd.VectorFunctionSpace(mesh, family="CG", degree=orderS)
            print(orderU, orderS)
        elif Pprime == 'L2' or Pprime == 'OSS':
            SF = fd.FunctionSpace(mesh, family="CG", degree=orderU)
            VF = fd.VectorFunctionSpace(mesh, family="CG", degree=orderS)
            VscS = fd.FunctionSpace(mesh, family="CG", degree=orderU)
            VscV = fd.VectorFunctionSpace(mesh, family="CG", degree=orderS)
        else:
            raise ValueError('Projection operator not recognized')
    else:
        raise ValueError('Formulation not recognized')

    # Define the mixed space:
    """
    All scalar fields use the same space SF, and all vector fields use the same space VF. The extra fields used when the 
    L2 or OSS operator is selected are defined according to the function spaces introduced so far. 
    s - u - e - lam - mu - s' - u' - e' - lam' - mu'
    """
    if Pprime == 'Identity':
        W = VF * SF * VF * SF * VF 
    elif Pprime == 'L2' or Pprime == 'OSS':
        W = VF * SF * VF * SF * VF * VscV * VscS * VscV * VscS * VscV
    else:
        raise ValueError('Projection operator not recognized')
    
    total_dofs = W.dim()
    print(f"Total DOFs in mixed space: {total_dofs}")
    
    # Coordinates and unit outward normal vector 
    x = fd.SpatialCoordinate(mesh)
    n = fd.FacetNormal(mesh)

    # Unknown field and Test functions
    w = fd.Function(W)
    if Pprime == 'Identity':
        s, u, e, lam, mu = fd.split(w)
        dels, delu, dele, dellam, delmu = fd.TestFunctions(W)
    elif Pprime == 'L2' or Pprime == 'OSS':
        s, u, e, lam, mu, PL2res_s, PL2res_u, PL2res_e, PL2res_l, PL2res_m = fd.split(w)
        dels, delu, dele, dellam, delmu, delsub_s, delsub_u, delsub_e, delsub_l, delsub_m = fd.TestFunctions(W)
    else:
        raise ValueError('Projection operator not recognized')

    # Functional to be minimized
    J = ObjFunc(e, graddata, s, fluxdata)

    # Get parameters and physical constants 
    zeta  = params['zeta']
    kappa = params['kappa']
    cu  = params['cu']
    ce  = params['ce']
    cs  = params['cs']
    cl  = params['cl']
    cm  = params['cm']
    L0   = params['L0']
    L1   = params['L1']
    
    # Sources
    q = volumetric_source
    f = dual_volumetric_source

    # Linear form (standard terms)
    etilde, stilde = graddata, fluxdata
    L = ( fd.inner(etilde,dele) + kappa*fd.inner(stilde,dels) ) * fd.dx

    # Set the stabilization parameters
    if formulation == 'VMS-Primal':
        tauu = cu*L0**2
        taue = ce
        taus = cs/kappa
        taul = -cl/(zeta*L1)**2
        taum = -cm
    elif formulation == 'VMS-Dual':
        h = fd.CellDiameter(mesh)
        tauu = cu*L0**2
        taue = ce
        taus = (cs/kappa)*(h**2)
        taul = -cl/(zeta*L1)**2
        taum = -cm*(h**2)
    else:
        raise Exception('Invalid VMS option')
 
    # Physical and dual source term (just for the manufactured solutions that have a rationale in DDCM)
    L += ( q(x) * dellam + f(x) * delu ) * fd.dx

    if formulation == 'VMS-Primal':

        # Bilinear form (Standard Galerkin terms - Primal)
        B  = ( zeta*fd.inner(lam,delu) - fd.inner(mu,fd.grad(delu)) ) * fd.dx
        B += ( fd.inner(e,dele) + fd.inner(mu,dele)                 ) * fd.dx
        B += ( kappa*fd.inner(s,dels) - fd.inner(fd.grad(lam),dels) ) * fd.dx
        B += ( zeta*fd.inner(u, dellam) - fd.inner(s,fd.grad(dellam))  ) * fd.dx
        B += ( -fd.inner(fd.grad(u), delmu) + fd.inner(e,delmu)     ) * fd.dx

        # Set the boundary conditions accordingly
        bc_lst = []
        if flag_setting == 'Pure-Dirichlet':

            bc_lst.append(fd.DirichletBC(W.sub(1), u_boundary_vals(x), "on_boundary"))
            bc_lst.append(fd.DirichletBC(W.sub(3), lmbd_boundary_vals(x), "on_boundary"))

            #bc_lst.append(fd.DirichletBC(W.sub(1), u_boundary_vals(x), (1,2,3,4,5,6)))
            #bc_lst.append(fd.DirichletBC(W.sub(3), lmbd_boundary_vals(x), (1,2,3,4,5,6)))

        elif flag_setting == 'Pure-Neumann':

            L -= ( fd.inner(s_boundary_vals(x), n) * dellam ) * fd.ds
            L -= ( fd.inner(mu_boundary_vals(x),n) * delu  ) * fd.ds
            
        elif flag_setting == 'Mixed-Dirichlet-Neumann':
            
            # Left and right (1,2)
            bc_lst.append(fd.DirichletBC(W.sub(1), u_boundary_vals(x), 1))
            bc_lst.append(fd.DirichletBC(W.sub(1), u_boundary_vals(x), 2))
            bc_lst.append(fd.DirichletBC(W.sub(3), lmbd_boundary_vals(x), 1))
            bc_lst.append(fd.DirichletBC(W.sub(3), lmbd_boundary_vals(x), 2))

            # Bottom and top (3,4)
            # L -= ( fd.inner(s_boundary_vals(x), n) * dellam ) * fd.ds(3)
            # L -= ( fd.inner(s_boundary_vals(x), n) * dellam ) * fd.ds(4)
            #
            # L -= ( fd.inner(mu_boundary_vals(x),n) * delu  ) * fd.ds(3)
            # L -= ( fd.inner(mu_boundary_vals(x),n) * delu  ) * fd.ds(4)
            
        else:
            raise ValueError('Unrecognized problem setting option')
        
    elif formulation == 'VMS-Dual':

        # Bilinear form (Standard Galerkin terms - Dual)
        B  = ( zeta*fd.inner(lam,delu) + fd.inner(fd.div(mu),delu)   ) * fd.dx
        B += ( fd.inner(e,dele) + fd.inner(mu,dele)                  ) * fd.dx
        B += ( kappa*fd.inner(s,dels) + fd.inner(lam,fd.div(dels))   ) * fd.dx
        B += ( zeta*fd.inner(u, dellam) + fd.inner(fd.div(s),dellam) ) * fd.dx
        B += ( fd.inner(u, fd.div(delmu)) + fd.inner(e,delmu)        ) * fd.dx

        # Set the boundary conditions accordingly 
        bc_lst = []
        if flag_setting == 'Pure-Dirichlet':

            print('Hello 1')

            L += u_boundary_vals(x)*fd.inner(delmu,n)   * fd.ds
            L += lmbd_boundary_vals(x)*fd.inner(dels,n) * fd.ds
        
        elif flag_setting == 'Pure-Neumann':


            # --- Remove DirichletBCs for s and mu (use weak imposition instead) ---
            '''
            bc_lst.append(fd.DirichletBC(W.sub(0), s_boundary_vals(x),  'on_boundary'))
            bc_lst.append(fd.DirichletBC(W.sub(4), mu_boundary_vals(x), 'on_boundary'))
            '''

            '''
            # bottom
            bc_lst.append(fd.DirichletBC(W.sub(0).sub(1), s_boundary_vals(x)[1], 1))
            bc_lst.append(fd.DirichletBC(W.sub(4).sub(1), mu_boundary_vals(x)[1], 1))

            # top
            bc_lst.append(fd.DirichletBC(W.sub(0).sub(1), s_boundary_vals(x)[1], 5))
            bc_lst.append(fd.DirichletBC(W.sub(4).sub(1), mu_boundary_vals(x)[1], 5))

            # left
            bc_lst.append(fd.DirichletBC(W.sub(0).sub(0), s_boundary_vals(x)[0], 6))
            bc_lst.append(fd.DirichletBC(W.sub(4).sub(0), mu_boundary_vals(x)[0], 6))

            # right
            bc_lst.append(fd.DirichletBC(W.sub(0).sub(0), s_boundary_vals(x)[0], 4))
            bc_lst.append(fd.DirichletBC(W.sub(4).sub(0), mu_boundary_vals(x)[0], 4))

            # corner horizontal
            def cornerhor(x):
                eps = fd.Constant(1.0e-9)
                return fd.near(x[1], 0) and 'on_boundary' and fd.conditional(fd.gt(x[0], eps), True, False)

            bc_lst.append(fd.DirichletBC(W.sub(0).sub(1), s_boundary_vals(x)[1], cornerhor))
            bc_lst.append(fd.DirichletBC(W.sub(4).sub(1), mu_boundary_vals(x)[1], cornerhor))

            # corner vertical
            def cornerver(x):
                eps = fd.Constant(1.0e-9)
                return fd.near(x[0], 0) and 'on_boundary' and fd.conditional(fd.lt(x[1], -eps), True, False)

            bc_lst.append(fd.DirichletBC(W.sub(0).sub(0), s_boundary_vals(x)[0], cornerver))
            bc_lst.append(fd.DirichletBC(W.sub(4).sub(0), mu_boundary_vals(x)[0], cornerver))
            '''

            # Ractangle domain

            print('Hello 2')
            bc_lst.append(fd.DirichletBC(W.sub(0).sub(0), s_boundary_vals(x)[0],  1))
            bc_lst.append(fd.DirichletBC(W.sub(0).sub(0), s_boundary_vals(x)[0],  2))
            bc_lst.append(fd.DirichletBC(W.sub(0).sub(1), s_boundary_vals(x)[1],  3))
            bc_lst.append(fd.DirichletBC(W.sub(0).sub(1), s_boundary_vals(x)[1],  4))
            bc_lst.append(fd.DirichletBC(W.sub(4).sub(0), mu_boundary_vals(x)[0],  1))
            bc_lst.append(fd.DirichletBC(W.sub(4).sub(0), mu_boundary_vals(x)[0],  2))
            bc_lst.append(fd.DirichletBC(W.sub(4).sub(1), mu_boundary_vals(x)[1],  3))
            bc_lst.append(fd.DirichletBC(W.sub(4).sub(1), mu_boundary_vals(x)[1],  4))


        elif flag_setting == 'Mixed-Dirichlet-Neumann':
            L += (u_boundary_vals(x))*fd.inner(delmu,n)   * (fd.ds(1)+fd.ds(2))
            L += (lmbd_boundary_vals(x))*fd.inner(dels,n) * (fd.ds(1)+fd.ds(2))

            bc_lst.append(fd.DirichletBC(W.sub(0), s_boundary_vals(x),  (3,4)))
            bc_lst.append(fd.DirichletBC(W.sub(4), mu_boundary_vals(x), (3,4)))

        else:
            raise ValueError('Unrecognized problem setting option')

    else:
        raise ValueError('Unrecognized formulation')

    res_u = f(x) - zeta*lam - fd.div(mu)
    res_e = etilde - e - mu 
    res_s = kappa*(stilde - s) + fd.grad(lam)
    res_l = q(x) - zeta*u - fd.div(s)
    res_m = fd.grad(u) - e

    if Pprime == 'L2' or Pprime == 'OSS': # Assemble L2 projections onto W
        B += ( fd.inner(PL2res_u, delsub_u) - fd.inner(res_u, delsub_u) ) * fd.dx
        B += ( fd.inner(PL2res_e, delsub_e) - fd.inner(res_e, delsub_e) ) * fd.dx
        B += ( fd.inner(PL2res_s, delsub_s) - fd.inner(res_s, delsub_s) ) * fd.dx
        B += ( fd.inner(PL2res_l, delsub_l) - fd.inner(res_l, delsub_l) ) * fd.dx
        B += ( fd.inner(PL2res_m, delsub_m) - fd.inner(res_m, delsub_m) ) * fd.dx

    if Pprime == 'Identity':
        sub_u = tauu*res_u
        sub_e = taue*res_e
        sub_s = taus*res_s 
        sub_l = taul*res_l
        sub_m = taum*res_m
     
    elif Pprime == 'L2':
        sub_u = tauu*PL2res_u
        sub_e = taue*PL2res_e
        sub_s = taus*PL2res_s
        sub_l = taul*PL2res_l
        sub_m = taum*PL2res_m

    elif Pprime == 'OSS':
        sub_u = tauu*(res_u - PL2res_u)
        sub_e = taue*(res_e - PL2res_e)
        sub_s = taus*(res_s - PL2res_s)
        sub_l = taul*(res_l - PL2res_l)
        sub_m = taum*(res_m - PL2res_m)

    else:
        raise ValueError('Invalid operator.')
    
    # Stabilization terms
    B += ( zeta*fd.inner(sub_l,delu) - fd.inner(sub_m,fd.grad(delu))     ) * fd.dx
    B += ( fd.inner(sub_e, dele) + fd.inner(sub_m, dele)                 ) * fd.dx
    B += ( kappa*fd.inner(sub_s,dels) +  fd.inner(sub_l,fd.div(dels))    ) * fd.dx
    B += ( zeta*fd.inner(sub_u,dellam) - fd.inner(sub_s,fd.grad(dellam)) ) * fd.dx
    B += ( fd.inner(sub_u, fd.div(delmu)) + fd.inner(sub_e, delmu)       ) * fd.dx

    # if formulation == 'VMS-Dual' and flag_setting == 'Pure-Neumann':
    #    L += sub_u*fd.inner(delmu,n)* fd.ds
    #    L += sub_l*fd.inner(dels,n) * fd.ds
    #    L -= dellam*fd.inner(sub_s,n)* fd.ds
    #    L -= delu*fd.inner(sub_m,n)* fd.ds

    # Total residual (LHS - RHS)
    Res = B - L

    # Jacobian
    Jac = fd.derivative(Res, w)

    # Solve
    fd.solve(Res == 0, w, bcs=bc_lst, J=Jac, solver_parameters=petsc_opt1)

    # udif = fd.Function(SF)
    # udif.interpolate(w.sub(1) - u_boundary_vals(x))

    # Save
    if dir_outs is not None:

        print(tt)
        if tt == 0.0: # This time acts as a parameter over which we do continuation
            mode = 'w'
        else:
            mode = 'a'

        filename = dir_outs+'/s.pvd'
        w.sub(0).rename('DDCM.Flux')
        VTKFile(filename, mode=mode).write(w.sub(0), project_output = False, time=tt)

        filename = dir_outs+'/u.pvd'
        w.sub(1).rename('DDCM.Potential')
        VTKFile(filename, mode=mode).write(w.sub(1), project_output = False, time=tt)

        filename = dir_outs+'/g.pvd'
        w.sub(2).rename('DDCM.gradient(u)')
        VTKFile(filename, mode=mode).write(w.sub(2), project_output = False, time=tt)

        filename = dir_outs+'/l.pvd'
        w.sub(3).rename('DDCM.lambda')
        VTKFile(filename, mode=mode).write(w.sub(3), project_output = False, time=tt)

        filename = dir_outs+'/m.pvd'
        w.sub(4).rename('DDCM.mu')
        VTKFile(filename, mode=mode).write(w.sub(4), project_output = False, time=tt)

        # filename = dir_outs+'/udif.pvd'
        # udif.rename('DDCM.udif')
        # VTKFile(filename, mode=mode).write(udif, project_output = False, time=tt)

    return w.sub(0), w.sub(1), w.sub(2), w.sub(3), w.sub(4)

    #wf = fd.Function(W)
    #wf.assign(w)
    #s_f, u_f, e_f, lam_f, mu_f = wf.sub(0), wf.sub(1), wf.sub(2), wf.sub(3), wf.sub(4)
    #
    #
    #if (True):
    #     ut = u + sub_u
    #     et = e + sub_e
    #     st = s + sub_s
    #     lt = lam + sub_l
    #     mt = mu + sub_m
    
    #     if (True):
    #         DGk = fd.FunctionSpace(mesh, 'DG', degree=orderU)
    #         VDGk = fd.VectorFunctionSpace(mesh, 'DG', degree=orderS)
    #         s_f = fd.assemble(interpolate(st, VDGk))
    #         u_f = fd.assemble(interpolate(ut, DGk))
    #         e_f = fd.assemble(interpolate(et, VDGk))
    #         lam_f = fd.assemble(interpolate(lt, DGk))
    #         mu_f = fd.assemble(interpolate(mt, VDGk))
    #     else:
    #         s_f = fd.project(st, W.sub(0))
    #         u_f = fd.project(ut, W.sub(1))
    #         e_f = fd.project(et, W.sub(2))
    #         lam_f = fd.project(lt, W.sub(3))
    #         mu_f = fd.project(mt, W.sub(4))
    #else:
    #     s_f, u_f, e_f, lam_f, mu_f = w.sub(0), w.sub(1), w.sub(2), w.sub(3), w.sub(4)
    #
    # if dir_outs is not None:
    #
    #     if tt == 0.0: # This time acts as a parameter over which we do continuation
    #         mode = 'w'
    #     else:
    #         mode = 'a'
    #     filename = dir_outs + '/s.pvd'
    #     s_f.rename('DDCM.Flux')
    #     VTKFile(filename, mode=mode).write(s_f, time=tt)
    #
    #     filename = dir_outs + '/u.pvd'
    #     u_f.rename('DDCM.Potential')
    #     VTKFile(filename, mode=mode).write(u_f, time=tt)
    #
    #     filename = dir_outs + '/g.pvd'
    #     e_f.rename('DDCM.gradient(u)')
    #     VTKFile(filename, mode=mode).write(e_f, time=tt)
    #
    #     filename = dir_outs + '/l.pvd'
    #     lam_f.rename('DDCM.lambda')
    #     VTKFile(filename, mode=mode).write(lam_f, time=tt)
    #
    #     filename = dir_outs + '/m.pvd'
    #     mu_f.rename('DDCM.mu')
    #     VTKFile(filename, mode=mode).write(mu_f, time=tt)

    #return s_f, u_f, e_f, lam_f, mu_f




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

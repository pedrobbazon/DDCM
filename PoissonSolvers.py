solpet = {
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

def SolvePoissonCG(mesh, kappa, source, flag_quads, HomDirVal, orderU=1):

    print( "|> Solving with CG (primal) formulation\n")

    W = FunctionSpace(mesh, "CG", orderU)

    u = TrialFunction(W)
    v = TestFunction(W)

    a = inner(kappa*grad(u), grad(v))*dx
    L = source*v*dx

    # Pressure (natural) boundary conditions
    uboundary = Constant(HomDirVal)
    bcall = DirichletBC(W, uboundary, "on_boundary")

    w = Function(W)
    solve(a == L, w, bcs=[bcall])

    return w



def FluxLaw(sigma, u, params):

  if(NL_model == 'Darcy'):

    kappa0 = params[0]
    return sigma + kappa0 * grad(u)

  elif(NL_model == 'mine1'):

    pert = params[0]
    return as_vector([(sigma[0] + pert*tanh(sin(14*2*np.pi*sigma[0])) + u.dx(0)),
                      (sigma[1] + pert*tanh(sin(14*2*np.pi*sigma[1])) + u.dx(1))])

  elif(NL_model == 'Forchheimer-Reg'):

    eta = params[0]
    muokap = params[1]
    betarho = params[2]
    modreg = eta*(sqrt(1 + inner(sigma,sigma)/eta**2) - 1.0)
    return grad(u) + muokap*sigma + betarho * modreg * sigma

  elif(NL_model == 'Klinkenberg'):

    kapomu = params[0]
    DK = params[1]
    return sigma + kapomu * grad(u) + (DK/u)*grad(u)


def SolvePoissonInvDualHdiv(mesh, flag_quads, ubcval, orderF=2, orderU=1):

    print( "|> Solving with Dual-Mixed formulation\n")

    if(flag_quads):
        HD = FunctionSpace(mesh, "RTCF", orderF)
        CG = FunctionSpace(mesh, "CG", orderU)
    else:
        HD = FunctionSpace(mesh, "DRT", orderF)
        CG = FunctionSpace(mesh, "CG", orderU)

    W = HD * CG

    w = Function(W)
    sigma, u = split(w)
    tau, v = TestFunctions(W)

    x, y = SpatialCoordinate(mesh)

    f = Constant(1.0)
    n = FacetNormal(mesh)

    # Pressure (natural) boundary conditions
    uboundary = Constant(ubcval)

    bcall = DirichletBC(W.sub(1), uboundary, "on_boundary")

    params = []

    if(NL_model == 'Darcy'):

      params.append(Constant(1.0))  # kappa0
      params.append(Constant(0.01)) # pert
      Res  = (inner(FluxLaw(sigma,u, params),tau))*dx
      Res += inner(sigma,grad(v))*dx
      Res += f*v*dx
      solve(Res == 0, w, bcs=[bcall], solver_parameters=solpet)

    elif(NL_model == 'Forchheimer-Reg'):

      params.append(Constant(1.0))     # muokap
      params.append(Constant(10.0))    # betarho
      params.append(Constant(1.0e-10)) # eta
      Res  = (inner(FluxLaw(sigma,u, params),tau))*dx
      Res += inner(sigma,grad(v))*dx
      Res += f*v*dx
      solve(Res == 0, w, bcs=[bcall], solver_parameters=solpet)

    elif(NL_model == 'Klinkenberg'):

      params.append(Constant(1.0))  # kapomu
      params.append(Constant(1.0))  # DK
      w.sub(1).assign(Constant(1.0))
      for dk in [0.0, 0.2, 0.4, 0.6]:
        print('Solving for DK = %f' %(dk))
        DK = Constant(dk)
        params[1] = DK
        Res  = (inner(FluxLaw(sigma,u, params),tau))*dx
        Res += inner(sigma,grad(v))*dx
        Res += f*v*dx
        solve(Res == 0, w, bcs=[bcall], solver_parameters=solpet)

    sigma, u = w.sub(0), w.sub(1)
    return sigma, u

def SolvePoissonInvDualHdiv1D(mesh, flag_quads, ubcval, orderF=2, orderU=1):

    print( "|> Solving with Dual-Mixed formulation\n")

    if(flag_quads):
        HD = FunctionSpace(mesh, "RTCF", orderF)
        CG = FunctionSpace(mesh, "CG", orderU)
    else:
        HD = FunctionSpace(mesh, "DRT", orderF)
        CG = FunctionSpace(mesh, "CG", orderU)

    W = HD * CG

    w = Function(W)
    sigma, u = split(w)
    tau, v = TestFunctions(W)

    x, y = SpatialCoordinate(mesh)

    f = Constant(0.0)
    n = FacetNormal(mesh)

    # Pressure (natural) boundary conditions
    uboundary = Constant(ubcval)

    # Boundary Conditions
    pbottom = Constant(0.01)
    ptop = Constant(1.01)

    bc_bottom = DirichletBC(W.sub(1),pbottom, 3)  # Bottom boundary condition
    bc_top = DirichletBC(W.sub(1), ptop, 4)       # Top boundary condition
    bcall = [bc_bottom, bc_top]

    NL_model = 'Klinkenberg'
    params = []

    if(NL_model == 'Darcy'):

      params.append(Constant(1.0))  # kappa0
      params.append(Constant(0.01)) # pert
      Res  = (inner(FluxLaw(sigma,u, params),tau))*dx
      Res += inner(sigma,grad(v))*dx
      Res += f*v*dx
      solve(Res == 0, w, bcs=bcall, solver_parameters=solpet)

    elif(NL_model == 'Forchheimer-Reg'):

      params.append(Constant(1.0))     # muokap
      params.append(Constant(10.0))    # betarho
      params.append(Constant(1.0e-10)) # eta
      Res  = (inner(FluxLaw(sigma,u, params),tau))*dx
      Res += inner(sigma,grad(v))*dx
      Res += f*v*dx
      solve(Res == 0, w, bcs=bcall, solver_parameters=solpet)

    elif(NL_model == 'Klinkenberg'):

      params.append(Constant(1.0))  # kapomu
      params.append(Constant(1.0))  # DK
      w.sub(1).interpolate(y*(ptop-pbottom) + pbottom)
      for dk in [0.1, 0.2, 0.4, 0.8]:
        print('Solving for DK = %f' %(dk))
        DK = Constant(dk)
        params[1] = DK
        Res  = (inner(FluxLaw(sigma,u, params),tau))*dx
        Res += inner(sigma,grad(v))*dx
        Res += f*v*dx
        solve(Res == 0, w, bcs=bcall, solver_parameters=solpet)

    sigma, u = w.sub(0), w.sub(1)
    return sigma, u


def SolvePoissonInvDualHdivStd(mesh, flag_quads, sourceval, ubcval, orderF=2, orderU=1):

    print( "|> Solving with Dual-Mixed formulation\n")

    if(flag_quads):
        HD = FunctionSpace(mesh, "RTCF", orderF)
        CG = FunctionSpace(mesh, "CG", orderU)
    else:
        HD = FunctionSpace(mesh, "DRT", orderF)
        CG = FunctionSpace(mesh, "CG", orderU)

    W = HD * CG

    w = Function(W)
    sigma, u = split(w)
    tau, v = TestFunctions(W)

    x, y = SpatialCoordinate(mesh)

    f = Constant(sourceval)
    n = FacetNormal(mesh)

    # Pressure (natural) boundary conditions
    uboundary = Constant(ubcval)

    bc_all = DirichletBC(W.sub(1), uboundary, "on_boundary")  # Bottom boundary condition
    bcall = [bc_all]

    NL_model = 'Klinkenberg'
    params = []

    if(NL_model == 'Darcy'):

      params.append(Constant(1.0))  # kappa0
      params.append(Constant(0.01)) # pert
      Res  = (inner(FluxLaw(sigma,u, params),tau))*dx
      Res += inner(sigma,grad(v))*dx
      Res += f*v*dx
      solve(Res == 0, w, bcs=bcall, solver_parameters=solpet)

    elif(NL_model == 'Forchheimer-Reg'):

      params.append(Constant(1.0))     # muokap
      params.append(Constant(10.0))    # betarho
      params.append(Constant(1.0e-10)) # eta
      Res  = (inner(FluxLaw(sigma,u, params),tau))*dx
      Res += inner(sigma,grad(v))*dx
      Res += f*v*dx
      solve(Res == 0, w, bcs=bcall, solver_parameters=solpet)

    elif(NL_model == 'Klinkenberg'):

      params.append(Constant(1.0))  # kapomu
      params.append(Constant(1.0))  # DK
      w.sub(1).interpolate(Constant(ubcval))
      for dk in [0.1, 0.2, 0.4, 0.8]:
        print('Solving for DK = %f' %(dk))
        DK = Constant(dk)
        params[1] = DK
        Res  = (inner(FluxLaw(sigma,u, params),tau))*dx
        Res += inner(sigma,grad(v))*dx
        Res += f*v*dx
        solve(Res == 0, w, bcs=bcall, solver_parameters=solpet)

    sigma, u = w.sub(0), w.sub(1)
    return sigma, u

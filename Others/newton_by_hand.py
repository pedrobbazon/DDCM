def q(u):
        return 1 + u ** 2

    x = ufl.SpatialCoordinate(mesh)
    u_ufl = 1 + x[0] + 2 * x[1]
    f = - ufl.div(q(u_ufl) * ufl.grad(u_ufl))

    V = FunctionSpace(mesh, "Lagrange", 1)
    uh = Function(V)
    v = TestFunction(V)

    F = q(uh) * ufl.dot(ufl.grad(uh), ufl.grad(v)) * ufl.dx - f * v * ufl.dx

    u_D = Function(V).interpolate(u_ufl)
    bc = DirichletBC(V, u_D, (1, 2, 3, 4))

    parameters = {
        "snes_monitor": None,
        "snes_rtol": 1e-6,
        "snes_atol": 1e-7,
        "ksp_type": "preonly",
        "mat_type": "aij",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    bcs = [bc]

    print("********* Firedrake non-linear solver *******")
    solve(F == 0, uh, bcs=bcs, solver_parameters=parameters)

    uh.assign(0.0)
    max_iterations = 10

    print("********* Custom non-linear solver *******")
    i = 0
    while i < max_iterations:
        # assemble Jacobian and residual
        J = assemble(derivative(F, uh), bcs=bcs)
        R = assemble(-F, bcs=bcs)

        du = Function(V)
        ksp = LinearSolver(J, solver_parameters=parameters)
        ksp.solve(du, R)
        uh.assign(uh + du)

        Fres = assemble(F, bcs=bcs)
        with Fres.dat.vec_ro as Fres_vec:
            resNR = Fres_vec.norm()
            print(f"Iteration {i}: Correction norm {resNR}")
            if resNR < 1e-10:
                break
        i += 1

from firedrake import *

# Use a triangular mesh
base = UnitCubeMesh(10, 10, 10)
mh = MeshHierarchy(base, 1)
mesh = mh[-1]
print(mesh.num_cells())
n = FacetNormal(mesh)
(x, y, z) = SpatialCoordinate(mesh)

# Define Taylor-Hood function space W
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = MixedFunctionSpace([V, Q])

# Define Reynolds number and bcs
Re = Constant(1)
bcs = [DirichletBC(W.sub(0), Constant((0, 0, 0)), (1, 2, 3, 4, 5)),
       DirichletBC(W.sub(0), as_vector([16 * x**2 * (1-x)**2, 16 * y**2 * (1-y)**2, 0]), (6,))]

w = Function(W, name="Solution")
#print(w.sub(0).dat.data.shape)
#print(w.sub(1).dat.data.shape)
(u, p) = split(w)


# Augmented Lagrangian term
gamma = Constant(100000.0)

# Define Lagrangian
L = (
      0.5 * inner(2/Re * sym(grad(u)), sym(grad(u)))*dx
    -       inner(p, div(u))*dx
    + 0.5 * gamma * inner(div(u), div(u))*dx
    )

# Optimality conditions
F = derivative(L, w)

sp0 = {
    'mat_type': 'nest',
    'snes_monitor': None,
    'snes_converged_reason': None,
    'snes_max_it': 20,
    'snes_atol': 1e-8,
    'snes_rtol': 1e-12,
    'snes_stol': 1e-06,
    'ksp_type': 'preonly',
    'ksp_converged_reason': None,
    'ksp_monitor_true_residual': None,
    'ksp_max_it': 500,
    'ksp_atol': 1e-08,
    'ksp_rtol': 1e-10,
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
    }

sp = {
    'mat_type': 'nest',
    'snes_monitor': None,
    'snes_converged_reason': None,
    'snes_max_it': 20,
    'snes_atol': 1e-8,
    'snes_rtol': 1e-12,
    'snes_stol': 1e-06,
    'ksp_type': 'fgmres',
    'ksp_converged_reason': None,
    'ksp_monitor_true_residual': None,
    'ksp_max_it': 500,
    'ksp_atol': 1e-08,
    'ksp_rtol': 1e-10,
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_factorization_type': 'full',
    'fieldsplit_0': {'ksp_convergence_test': 'skip',
                     'ksp_max_it': 1,
                     'ksp_norm_type': 'unpreconditioned',
                     'ksp_richardson_self_scale': False,
                     'ksp_type': 'richardson',
                     'pc_type': 'mg',
                     'pc_mg_type': 'full',
                     'mg_coarse_assembled_pc_type': 'lu',
                     'mg_coarse_assembled_pc_factor_mat_solver_type': 'mumps',
                     'mg_coarse_pc_python_type': 'firedrake.AssembledPC',
                     'mg_coarse_pc_type': 'python',
                     'mg_levels': {'ksp_convergence_test': 'skip',
                                   'ksp_max_it': 5,
                                   'ksp_type': 'fgmres',
                                   'pc_python_type': 'firedrake.ASMStarPC',
                                   'pc_type': 'python'},
                    },
    'fieldsplit_1': {'ksp_type': 'richardson',
                     'ksp_max_it': 1,
                     'ksp_convergence_test': 'skip',
                     'ksp_richardson_scale': 1,
                     'pc_type': 'python',
                     'pc_python_type': 'firedrake.MassInvPC',
                     'Mp_pc_type': 'jacobi'},
}

stokes_petsc_options = {
    # Outer solver (Stokes system)
    "ksp_type": "fgmres",               # Flexible GMRES for saddle-point problems
    "ksp_rtol": 1e-6,                   # Relative tolerance
    "ksp_max_it": 200,                  # Max iterations
    "ksp_monitor_true_residual": None,   # Monitor true residual (optional)
    
    # Preconditioner setup (FieldSplit for Schur complement)
    "pc_type": "fieldsplit",            # Use field decomposition
    "pc_fieldsplit_type": "schur",      # Schur complement approach
    "pc_fieldsplit_schur_fact_type": "full",  # Full factorization (A, B, B^T)
    "pc_fieldsplit_schur_precondition": "self",  # Approximate Schur complement
    
    # Velocity block (A block, P2 space)
    "fieldsplit_0_ksp_type": "preonly",  # Only apply preconditioner
    "fieldsplit_0_pc_type": "hypre",     # Hypre/BoomerAMG for velocity
    "fieldsplit_0_pc_hypre_type": "boomeramg",  # AMG method
    
    # Schur complement block (Pressure block, P1 space)
    "fieldsplit_1_ksp_type": "preonly",  # Only apply preconditioner
    "fieldsplit_1_pc_type": "jacobi",    # Simple diagonal (or 'hypre' if needed)
    
    # Optional: Use if BoomerAMG struggles with large 3D problems
    # "fieldsplit_0_pc_hypre_boomeramg_max_iter": 1,
    # "fieldsplit_0_pc_hypre_boomeramg_strong_threshold": 0.7,
}

stokes_petsc_options = {
    "ksp_type": "fgmres",
    "ksp_rtol": 1e-6,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_schur_precondition": "self",  # ← Replace with "a11" or "user"
    "fieldsplit_0": {
        "ksp_type": "preonly",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
    },
    "fieldsplit_1": {
        "ksp_type": "preonly",
        "pc_type": "none",  # ← Disable unsupported diagonal extraction
    },
}

stokes_petsc_options = {
    'snes_monitor': None,
    'snes_converged_reason': None,
    'snes_max_it': 20,
    'snes_atol': 1e-8,
    'snes_rtol': 1e-12,
    'snes_stol': 1e-06,
    'ksp_type': 'fgmres',
    'ksp_converged_reason': None,
    'ksp_monitor_true_residual': None,
    'ksp_max_it': 1000,
    'ksp_atol': 1e-07,                  
    'ksp_rtol': 1e-07,    
    "ksp_type": "fgmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "additive",  # ← Diagonal blocks only
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "hypre",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "jacobi",
}

stokes_solver_parameters = {
    'snes_monitor': None,
    'snes_converged_reason': None,
    'snes_max_it': 20,
    'snes_atol': 1e-8,
    'snes_rtol': 1e-12,
    'snes_stol': 1e-06,
    'ksp_converged_reason': None,
    'ksp_monitor_true_residual': None,
    'ksp_max_it': 1000,
    'ksp_atol': 1e-07,
    'ksp_rtol': 1e-07,
    "mat_type": "aij",  # CSR format (AIJ = default sparse matrix format in PETSc)
    "ksp_type": "gmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",  # or "lower", "upper" for triangular
    "pc_fieldsplit_schur_precondition": "selfp",  # or "a11" (try both)
    "fieldsplit_velocity_ksp_type": "preonly",
    "fieldsplit_velocity_pc_type": "hypre",  # or "gamg" or "ilu" (if serial/small)
    "fieldsplit_pressure_ksp_type": "preonly",
    "fieldsplit_pressure_pc_type": "jacobi",  # "none" or "sor" can work too
}

petsc_options={
        'snes_monitor': None,
        'snes_converged_reason': None,
        'snes_max_it': 20,
        'snes_atol': 1e-8,
        'snes_rtol': 1e-12,
        'snes_stol': 1e-06,
        'ksp_converged_reason': None,
        'ksp_monitor_true_residual': None,
        'ksp_max_it': 1000,
        'ksp_atol': 1e-07,
        'ksp_rtol': 1e-07,
        "mat_type": "aij",  # CSR format (AIJ = default sparse matrix format in PETSc)
        "ksp_type": "gmres",
        "ksp_rtol": 1e-9,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "hypre",
        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "jacobi",
        }

sp = {
    'mat_type': 'nest',
    'snes_monitor': None,
    'snes_converged_reason': None,
    'snes_max_it': 20,
    'snes_atol': 1e-8,
    'snes_rtol': 1e-12,
    'snes_stol': 1e-06,
    'ksp_type': 'fgmres',
    'ksp_converged_reason': None,
    'ksp_monitor_true_residual': None,
    'ksp_max_it': 500,
    'ksp_atol': 1e-08,
    'ksp_rtol': 1e-10,
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_factorization_type': 'full',
    'fieldsplit_0': {'ksp_convergence_test': 'skip',
                     'ksp_max_it': 1,
                     'ksp_norm_type': 'unpreconditioned',
                     'ksp_richardson_self_scale': False,
                     'ksp_type': 'richardson',
                     'pc_type': 'mg',
                     'pc_mg_type': 'full',
                     'mg_coarse_assembled_pc_type': 'lu',
                     'mg_coarse_assembled_pc_factor_mat_solver_type': 'mumps',
                     'mg_coarse_pc_python_type': 'firedrake.AssembledPC',
                     'mg_coarse_pc_type': 'python',
                     'mg_levels': {'ksp_convergence_test': 'skip',
                                   'ksp_max_it': 5,
                                   'ksp_type': 'fgmres',
                                   'pc_python_type': 'firedrake.ASMStarPC',
                                   'pc_type': 'python'},
                    },
    'fieldsplit_1': {'ksp_type': 'richardson',
                     'ksp_max_it': 1,
                     'ksp_convergence_test': 'skip',
                     'ksp_richardson_scale': 1,
                     'pc_type': 'python',
                     'pc_python_type': 'firedrake.MassInvPC',
                     'Mp_pc_type': 'jacobi'},
}


# Solve problem
sp['fieldsplit_1']['ksp_richardson_scale'] = -(2/float(Re) + float(gamma))

import time

ti = time.time()

lin_var = NonlinearVariationalProblem(F, w, bcs)
solver = NonlinearVariationalSolver(lin_var,solver_parameters=sp0)

solver.solve()
#solve(F == 0, w, bcs, solver_parameters=sp0)
print(time.time() - ti)

w.sub(0).dat.data[:] = 0.0
w.sub(1).dat.data[:] = 0.0
ti = time.time()
solver.solve()
#solve(F == 0, w, bcs, solver_parameters=sp0)
print(time.time() - ti)


# Monitor incompressibility
print(f"||div u||: {norm(div(u), 'L2'):.2e}")

# Save solutions
(u_, p_) = w.subfunctions
u_.rename("Velocity")
p_.rename("Pressure")
VTKFile("output/stokes.pvd").write(u_, p_)

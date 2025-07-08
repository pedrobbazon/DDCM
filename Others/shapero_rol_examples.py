import firedrake
mesh = firedrake.UnitSquareMesh(32, 32, diagonal='crossed')
Q = firedrake.FunctionSpace(mesh, 'CG', 2)
V = firedrake.FunctionSpace(mesh, 'CG', 2)

import numpy as np
from numpy import random, pi as π
x = firedrake.SpatialCoordinate(mesh)

rng = random.default_rng(seed=1)
def random_fourier_series(std_dev, num_modes, exponent):
    from firedrake import sin, cos
    A = std_dev * rng.standard_normal((num_modes, num_modes))
    B = std_dev * rng.standard_normal((num_modes, num_modes))
    return sum([(A[k, l] * sin(π * (k * x[0] + l * x[1])) +
                 B[k, l] * cos(π * (k * x[0] + l * x[1])))
                / (1 + (k**2 + l**2)**(exponent/2))
                for k in range(num_modes)
                for l in range(int(np.sqrt(num_modes**2 - k**2)))])

from firedrake import Function
g = Function(V).interpolate(random_fourier_series(1.0, 6, 1))

from firedrake import inner, max_value, conditional, Constant
a = -Constant(4.5)
r = Constant(1/4)
ξ = Constant((0.4, 0.5))
q_true = Function(Q).interpolate(a * conditional(inner(x - ξ, x - ξ) < r**2, 1, 0))

b = Constant(6.)
R = Constant(1/4)
η = Constant((0.7, 0.5))
f = Function(V).interpolate(b * max_value(0, 1 - inner(x - η, x - η) / R**2))


from firedrake import exp, grad, dx, ds
k = Constant(1.)
h = Constant(10.)
u_true = Function(V)
v = firedrake.TestFunction(V)
F = (
    (k * exp(q_true) * inner(grad(u_true), grad(v)) - f * v) * dx +
    h * (u_true - g) * v * ds
)


opts = {
    'solver_parameters': {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
    },
}
firedrake.solve(F == 0, u_true, **opts)

ξ = Function(V)
n = len(ξ.dat.data_ro)
ξ.dat.data[:] = rng.standard_normal(n)

from firedrake import assemble
from firedrake.petsc import PETSc
ϕ, ψ = firedrake.TrialFunction(V), firedrake.TestFunction(V)
m = inner(ϕ, ψ) * dx
M = assemble(m, mat_type='aij').M.handle
ksp = PETSc.KSP().create()
ksp.setOperators(M)
ksp.setUp()
pc = ksp.pc
pc.setType(pc.Type.CHOLESKY)
pc.setFactorSolverType(PETSc.Mat.SolverType.PETSC)
pc.setFactorSetUpSolverType()
L = pc.getFactorMatrix()
pc.setUp()

area = assemble(Constant(1) * dx(mesh))
z = Function(V)
z.dat.data[:] = rng.standard_normal(n)
with z.dat.vec_ro as Z:
    with ξ.dat.vec as Ξ:
        L.solveBackward(Z, Ξ)
        Ξ *= np.sqrt(area / n)

û = u_true.dat.data_ro[:]
signal = û.max() - û.min()
signal_to_noise = 50
σ = firedrake.Constant(signal / signal_to_noise)

u_obs = u_true.copy(deepcopy=True)
u_obs += σ * ξ

from firedrake.adjoint import (
    Control, ReducedFunctional, MinimizationProblem, ROLSolver
)

firedrake.adjoint.continue_annotation()
q = Function(Q)
u = Function(V)
F = (
    (k * exp(q) * inner(grad(u), grad(v)) - f * v) * dx +
    h * (u - g) * v * ds
)
forward_problem = firedrake.NonlinearVariationalProblem(F, u)
forward_solver = firedrake.NonlinearVariationalSolver(forward_problem, **opts)
forward_solver.solve()


Δ = firedrake.VectorFunctionSpace(mesh, 'CG', 1)
v = Function(Δ)
μ = Function(Δ)

α = Constant(5e-2)
ρ = Constant(1.0)

rol_options = {
    "Step": {
        "Type": "Line Search",
        "Line Search": {"Descent Method": {"Type": "Quasi-Newton Step"}},
    },
    "Status Test": {
        "Gradient Tolerance": 1e-4,
        "Step Tolerance": 1e-4,
        "Iteration Limit": 500,
    },
    "General": {
        "Print Verbosity": 0,
        "Secant": {"Type": "Limited-Memory BFGS", "Maximum Storage": 10},
    },
}

from firedrake import sqrt
from tqdm.notebook import trange

qs = [q.copy(deepcopy=True)]
vs = [v.copy(deepcopy=True)]
μs = [μ.copy(deepcopy=True)]

num_steps = 15
for step in range(num_steps):
    # Step 1: Solve the inverse problem for q.
    J = assemble(
        0.5 * ((u - u_obs) / σ)**2 * dx +
        0.5 * ρ * α**2 * inner(grad(q) + μ - v, grad(q) + μ - v) * dx
    )

    q̂ = Control(q)
    Ĵ = ReducedFunctional(J, q̂)
    inverse_problem = MinimizationProblem(Ĵ)
    inverse_solver = ROLSolver(inverse_problem, rol_options, inner_product="L2")
    q_opt = inverse_solver.solve()
    q.assign(q_opt)
    forward_solver.solve()

    # Step 2: soft thresholding for v.
    z = grad(q) + μ
    expr = conditional(
        (ρ * α) ** 2 * inner(z, z) < 1,
        Constant((0, 0)),
        (1 - 1 / (ρ * α * sqrt(inner(z, z)))) * z
    )
    v.project(expr)

    # Step 3: gradient ascent for μ.
    μ.project(μ + grad(q) - v)

    qs.append(q.copy(deepcopy=True))
    vs.append(v.copy(deepcopy=True))

firedrake.adjoint.pause_annotation()

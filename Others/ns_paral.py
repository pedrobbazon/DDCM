import firedrake as fd

mesh = fd.RectangleMesh(100, 50, 2, 1, comm=fd.COMM_WORLD)
V = fd.VectorFunctionSpace(mesh, "CG", 2)
Q = fd.FunctionSpace(mesh, "CG", 1)
Z = V * Q
print(f"Number of dofs: {V.dof_count + Q.dof_count}")

Re = fd.Constant(100)
delta = fd.Constant(1e-3)
h = fd.CellVolume(mesh)

up = fd.Function(Z)
u, p = fd.split(up)
v, q = fd.TestFunctions(Z)
F = (
    1 / Re * fd.inner(fd.grad(u), fd.grad(v)) * fd.dx
    - p * fd.div(v) * fd.dx
    + fd.div(u) * q * fd.dx
)

# Create boundary conditions
x = fd.SpatialCoordinate(mesh)
val = [2 * x[0] - x[0] ** 2, 0]
bcs = [
    fd.DirichletBC(Z.sub(0), val, (4,)),
    fd.DirichletBC(Z.sub(0), fd.Constant((0.0, 0.0)), (1, 2, 3)),
]

F += fd.inner(fd.dot(fd.grad(u), u), v) * fd.dx

nullspace = fd.MixedVectorSpaceBasis(
    Z, [Z.sub(0), fd.VectorSpaceBasis(constant=True, comm=fd.COMM_WORLD)]
)

J = fd.assemble(fd.derivative(F, up), bcs=fd.homogenize(bcs))
ls = fd.LinearSolver(J, solver_parameters={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})

resNR = 1.0  # just a custom value to start
i = 0
dup = fd.Function(up.function_space()).assign(0.0)

while i < 20 and resNR >= 1e-7:
    for bc in bcs:
        bc.apply(up)

    Fres = fd.assemble(F, bcs=fd.homogenize(bcs))
    with Fres.dat.vec_ro as Fres_vec:
        resNR = Fres_vec.norm()
    if fd.COMM_WORLD.rank == 0:
        print(f"Iteration {i}: Correction norm {resNR}")

    J = fd.assemble(fd.derivative(F, up), bcs=fd.homogenize(bcs))
    R = fd.assemble(-F, bcs=fd.homogenize(bcs))
    ls.ksp.setOperators(J.M.handle)
    ls.solve(dup, R)
    up.assign(up + dup) 
    i += 1

file = fd.VTKFile("output.pvd")
u, p = up.subfunctions
file.write(u, p)

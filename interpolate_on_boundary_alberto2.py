from firedrake import *

mesh = UnitSquareMesh(10, 10)
space = FunctionSpace(mesh, "Lagrange", 1)
test = TestFunction(space)
trial = TrialFunction(space)

u = Function(space, name="u")
mat = assemble(Constant(0) * inner(trial, test) * dx).petscmat

u_bc = Function(space, name="u_bc")
DirichletBC(space, Constant(1), "on_boundary").apply(u_bc)

with u_bc.dat.vec_ro as u_bc_v:
    mat.setDiagonal(u_bc_v)
mat.assemblyBegin()
mat.assemblyEnd()



u_test = Function(space).interpolate(Constant(1))
u_test_bc = Function(space)
with u_test.dat.vec_ro as u_test_v, u_test_bc.dat.vec_wo as u_test_bc_v:
    mat.mult(u_test_v, u_test_bc_v)

# Zero on interior
interior = Function(space).assign(u_test_bc)
DirichletBC(space, Constant(0), "on_boundary").apply(interior)
print(f"{norm(interior)=}")

# Equal on boundary
print(f"{sqrt(abs(assemble(inner(u_test - u_test_bc, u_test - u_test_bc) * ds)))=}")

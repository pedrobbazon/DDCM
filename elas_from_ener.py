from firedrake import *

mesh = RectangleMesh(8, 8, 0.35, 0.02, quadrilateral=True)

D = VectorFunctionSpace(mesh, "CG", 2)

d = Function(D)

d_test = TestFunction(D)

rhos = Constant(1e3)
nus = Constant(0.4)
mus = Constant(0.5*1e6)

lam = Constant(2.0*mus*nus/(1.0-2.0*nus))

I = Identity(2)

def def_stress(d):
    F = I + grad(d) # Deformation gradient
    C = F.T * F # Right Cauchy-Green tensor
    E = 0.5*(C - I)
    J = det(F)
    Finv = inv(F)
    return (F, C, E, J, Finv)

def PK2(E):
    return lam*tr(E)*I+2.0*mus*E

(F, C, E, J, Finv) = def_stress(d)

R = 0.0

R += inner(F * PK2(E), grad(d_test)) * dx
R -= rhos * inner(as_vector((0.0, -2.0)), d_test) * dx

bcs = [
    DirichletBC(D, as_vector((0.0, 0.0)), (1,)),
]

problem = NonlinearVariationalProblem(R, d, bcs=bcs)

solver = NonlinearVariationalSolver(problem,
    solver_parameters={
        "snes_monitor": None,
        "snes_rtol": 1e-6,
        "mat_type": "aij",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
    }
)

d.rename("displacement")

solver.solve()

outfile = VTKFile("output/csm1.pvd")
outfile.write(d)


from firedrake.petsc import PETSc
import firedrake.solving_utils as solving_utils
from firedrake.matrix import MatrixBase
from firedrake import *
import numpy as np

class MatrixFreeB(object):

    def __init__(self, A, u, v):
        self.A = A
        self.u = u
        self.v = v

    def mult(self, mat, x, y):
        # y <- A x
        self.A.mult(x, y)

        # alpha <- v^T x
        alpha = self.v.dot(x)

        # y <- y + alpha*u
        y.axpy(alpha, self.u)


def setup_problem(mesh, t, kappa):
    gdim = mesh.geometric_dimension()
    dx = Measure("dx", mesh)

    RT = FiniteElement("RT", mesh.ufl_cell(), 1)
    DG  = FiniteElement("DG", mesh.ufl_cell(), 0)
    W = FunctionSpace(mesh, MixedElement([RT, DG]))
    CG2 = FunctionSpace(mesh, "CG", 2)

    x,y = SpatialCoordinate(mesh)
    f = Function(CG2).interpolate(sin(2*pi*x)*cos(2*pi*y) + 1)

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    a = 1/kappa*inner(u,v)*dx + p*div(v)*dx \
        + div(u)*q*dx - (t + 1 - t)*p*q*dx
    
    L = -f*q*dx

    b = 1/kappa*inner(u,v)*dx + 1/(kappa + t)*div(u)*div(v)*dx \
        + (kappa + t)*p*q*dx
    
    bcs = [DirichletBC(W.sub(0), Constant([0]*gdim), "on_boundary")]
    
    return a, b, L, W, bcs

mesh = UnitSquareMesh(20,20)
gdim = mesh.geometric_dimension()
mesh.coordinates.dat.data[:] *= 1
t = Constant(1e-6)
kappa = Constant(1e-2)
a, b, L, W, bcs = setup_problem(mesh, t, kappa)

# setup system matrix
B = PETSc.Mat().create()
A = assemble(a, bcs=bcs).M.handle

weight = (Constant(1) - t)
domain_size = assemble(Constant(1)*dx(domain=mesh))

zv = Constant([0]*gdim)
v,q = TestFunctions(W)
vecform = (inner(zv, v) + q)
cv = assemble(vecform*dx)
cvn = assemble(- Constant(domain_size)*weight*vecform*dx)

with cv.dat.vec_ro as u_vec:
    u = u_vec.copy()
with cvn.dat.vec_ro as v_vec:
    v = v_vec.copy()


# Build the matrix "context"
Bctx = MatrixFreeB(A, u, v)
B.setSizes(A.getSizes())
B.setType(B.Type.PYTHON)
B.setPythonContext(Bctx)
B.setDM(W.dm)
B.setUp()

class Op(MatrixBase):

    def __init__(self, a, bcs, mat_type, petscmat, *args, **kwargs):
        # sets self._a, self._bcs, and self._mat_type
        MatrixBase.__init__(self, a, bcs, mat_type)
        options_prefix = kwargs.pop("options_prefix")
        self.petscmat = petscmat
        self.petscmat.setOptionsPrefix(options_prefix)
        self.mat_type = mat_type
        self.appctx = dict()

BM = Op(a, bcs, "python", B, options_prefix="bm")

lu_precond = {"ksp_type":"preonly", "pc_type":"lu"}

parameters = {
        "ksp_type": "minres",
        "ksp_rtol": 1e-18,
        "ksp_atol": 1e-30,
        "ksp_norm_type":"preconditioned",
        "ksp_minres_radius":1e30,
        "ksp_monitor": None,
        #"ksp_view": None,
        "ksp_view_singularvalues":None,
        'ksp_converged_reason': None,
        "pc_type": "fieldsplit",
        "ksp_initial_guess_nonzero":1,
        "pc_fieldsplit_type": "additive",
        "fieldsplit_0":lu_precond,
        "fieldsplit_1":lu_precond,
    }


P = assemble(b, bcs=bcs)
sol = Function(W)

solver = LinearSolver(BM, P=P, solver_parameters=parameters)        

rhs = assemble(L, bcs=bcs)

lvp = LinearVariationalProblem(a=BM.a, L=0, u=sol, bcs=bcs)
mat_type = BM.mat_type
ctx = solving_utils._SNESContext(lvp,
                                mat_type=mat_type,
                                pmat_type=mat_type,
                                appctx=dict(),
                                options_prefix="ctx")
dm = solver.ksp.dm
with dmhooks.add_hooks(W.dm, solver, appctx=ctx):
    solver.solve(sol,rhs)

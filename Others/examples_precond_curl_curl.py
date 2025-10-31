from firedrake import *

first_msh = UnitCubeMesh(5,5,5)
meshes = MeshHierarchy(first_msh, 1)

k = 1.0
kappa = Constant(k)

class myPC(AuxiliaryOperatorPC):
    def form(self, pc, test, trial):
        a = (inner(curl(trial), curl(test)) * dx
                - kappa**2 * inner(trial, test) * dx)
        return a, None

solver_parameters = {
        "snes_type": "ksponly",
        "mat_type": "matfree",
        "ksp_type": "gmres",
        "ksp_max_it": 100,
        "ksp_gmres_restart": 50,
        "ksp_monitor": None,
        "pc_type": "python",
        "pc_python_type": "__main__.myPC",
        "aux": {
            "pc_type": "mg",
            "mg_levels": {
                "ksp_type": "gmres",
                "ksp_max_it": 2,
                "ksp_convergence_test": "skip",
                "pc_type": "python",
                "pc_python_type": "firedrake.ASMStarPC",
                "pc_star_construct_dim": 0,
                },
            "mg_coarse": {
                "ksp_type": "preonly",
                "pc_type": "lu"
                }
            }
        }

for msh in meshes[1:]:
    # U = FunctionSpace(msh, "CG", fe_deg)
    # V = VectorFunctionSpace(msh, "CG", fe_deg)
    W = FunctionSpace(msh, "N1curl", 1)

    u = Function(W)
    v = TestFunction(W)

    F = inner(curl(u), curl(v)) * dx - kappa**2 * inner(u, v) * dx
    problem = NonlinearVariationalProblem(F, u)
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)
    solver.solve()



from firedrake import *

m = UnitIntervalMesh(10)
mh = MeshHierarchy(m, refinement_levels=1)
hierarchy = ExtrudedMeshHierarchy(mh, 1, base_layer=20, refinement_ratio=3, extrusion_type='uniform')
mesh = hierarchy[-1]

RT = FunctionSpace(mesh, 'RTCF', 1)
DG = FunctionSpace(mesh, 'DG', 0)
W = RT * DG
U = Function(W)
u, p = split(U)
w, q = TestFunctions(W)

VDG = VectorFunctionSpace(mesh, 'DQ', 1)
pcg = PCG64(seed=1234567)
rg = Generator(pcg)
f = rg.normal(VDG, 1.0, 2.0)

bc1 = DirichletBC(W.sub(0), as_vector([0., 0.]), "top")
bc2 = DirichletBC(W.sub(0), as_vector([0., 0.]), "bottom")
bc3 = DirichletBC(W.sub(0), as_vector([0., 0.]), "on_boundary")
bcs = [bc1, bc2, bc3]

eqn = (inner(u,w) - div(w)*p + div(u)*q)*dx - inner(f, w) * dx

v_basis = VectorSpaceBasis(constant=True, comm=COMM_WORLD)
nullspace = MixedVectorSpaceBasis(W, [W.sub(0), v_basis])


params_direct = {
        'snes_type':'ksponly',
        'ksp_type': 'gmres',
        'snes_monitor':None,
        'ksp_monitor':None,
        'mat_type': 'aij',
        'pc_type':'lu',
        'pc_factor_mat_solver_type': 'mumps',
}

nprob = NonlinearVariationalProblem(eqn, U, bcs=bcs)
nsolver = NonlinearVariationalSolver(nprob, nullspace=nullspace, solver_parameters=params_direct)
nsolver.solve()

print(U.sub(0).dat.data[:].shape)
print(U.sub(1).dat.data[:].shape)

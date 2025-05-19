verb = 1
Nmesh = 50
k = 6
epsilon_tol = 0.1           
bigC = 0.0025 
xi = 1.0e-6

mesh = UnitSquareMesh(Nmesh,Nmesh)
x, y = SpatialCoordinate(mesh)

print("Define the 3 points to be deformed and their expected images:")
xref = np.array([
    [1/2, 1/2],
    [1/4, 1/4],
    [3/4, 1/4]
])
Z = np.array([
    [1/4, 3/4],
    [1/16, 1/16],
    [1/2, 1/4]
])

V = VectorFunctionSpace(mesh, "CG", k)

print("Define the boundary condition :")
RravT_space = FunctionSpace(mesh, "RT", k)
bc = DirichletBC(RravT_space, as_vector([0.0,0.0]), "on_boundary")  
RTfct = Function(RravT_space)
bc.apply(RTfct)

phi = Function(V)
phi.interpolate(RTfct) 

idd = Function(V)
idd.interpolate(as_vector([x, y]))  

Psi = Function(V)
Psi = idd + phi

print("Define the functional : ")
J = AdjFloat(0.0)
J += Func_jac(mesh, Psi, bigC, epsilon_tol)
J += Func_H2(mesh, Psi, k, 2)

Bphi_array = np.array([phi.at(xref[i]) for i in range(len(xref))])
C = np.linalg.norm(np.abs(Z - xref - Bphi_array))

Functional = xi * J + 0.5 * C**2

parameters = {
    "tao_type": "lmvm",     
    "tao_max_it": 100,       
    "tao_monitor": None,  
    "tao_max_funcs": 1000,  
    "tao_gatol": 1e-8,      
    "tao_grtol": 1e-8,      
    "tao_stol": 1e-8,       
}

m = Control(phi)
rf = ReducedFunctional(Functional, m)
problem = MinimizationProblem(rf)
solver = TAOSolver(problem, parameters=parameters)

phi_opt = solver.solve() 

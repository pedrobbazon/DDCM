from firedrake import *
mesh = UnitSquareMesh(10, 10) 
p = 1 
V = FunctionSpace(mesh, "DG", p)
u = Function(V, name="u_DG")
v = TestFunction(V)
sigma = 10*p**2/CellDiameter(mesh)
n = FacetNormal(mesh)
B = (dot(grad(u), grad(v)) * dx
   - dot(avg(grad(u)),jump(v, n)) * dS
   - dot(avg(grad(v)),jump(u, n)) * dS
   + sigma*dot(jump(u, n), jump(v, n)) * dS
   - dot(avg(grad(u)),jump(v, n)) * ds
   - dot(avg(grad(v)),jump(u, n)) * ds
   + sigma*dot(jump(u, n), jump(v, n)) * ds) 
L = - Constant(1.) * (dot(grad(v), n) - sigma*v) * ds
F = B - L 
solve(F == 0, u)

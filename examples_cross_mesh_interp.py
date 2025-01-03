# Taylor-Hood elements
mesh_f = Mesh('fluid_domain.msh')
Vf_velocity = VectorFunctionSpace(mesh_f, "CG", 2)
Vf_pressure = FunctionSpace(mesh_f, "CG", 1)
Vf = Vf_velocity * Vf_pressure

# weak form
z = Function(Vf)
w = TestFunction(Vf)
u, p = split(z)
v, q = split(w)
F = inner(grad(u), grad(v))*dx - p*div(v)*dx + div(u)*q*dx

# boundary conditions
x, y = SpatialCoordinate(mesh_f)
uin = as_vector([4-y**2, 0]) 
bc1 = DirichletBC(Vf.sub(0), uin, 12) # inlet velocity
bc2 = DirichletBC(Vf.sub(0), 0, [11, 13]) # no slip condition
solve(F == 0, z, bcs=[bc1, bc2])
u_, p_ = z.subfunctions
u_.rename("velocity")
p_.rename("pressure")
VTKFile("fluid_sol.pvd").write(p_, u_) 


# interpolate on different mesh
mesh_h = Mesh('heat_domain.msh')
Vh_p = FunctionSpace(mesh_h, "CG", 1)
ph = Function(Vh_p, name="pressure")
ph.interpolate(p, allow_missing_dofs=True)
Vh_u = VectorFunctionSpace(mesh_h, "CG", 2)
bh = Function(Vh_u, name="velocity")
bh.interpolate(u, allow_missing_dofs=True)
VTKFile("heat_sol.pvd").write(ph, bh)

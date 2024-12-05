mesh = UnitIcosahedralSphereMesh(refinement_level=3)
plex = mesh.topology_dm
pVertexStart, pVertexEnd = plex.getHeightStratum(2)
V = FunctionSpace(mesh, "CG", degree=1)
u = Function(V)
for vertex in range(pVertexStart, pVertexEnd):
    u[dof_index] = f(vertex)



section = V.dm.getDefaultSection()

for vertex in range(...):
    ndof = section.getDof(vertex)
    assert ndof == 1
    offset = section.getOffset(vertex)
    u[offset] = f(vertex)

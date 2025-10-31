V = FunctionSpace(mesh,"CG",1)
u = Function(V)
plex = mesh.topology_dm
pVertexStart, pVertexEnd = plex.getHeightStratum(2)
section = V.dm.getDefaultSection()
for vertex in range(pVertexStart, pVertexEnd):
    offset = section.getOffset(vertex)
    u.dat.data[offset] = vertex

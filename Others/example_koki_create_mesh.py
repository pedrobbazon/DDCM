vertices = [
        [0., 0., 0.],
        [0., 0., 1.],
        [0., 1., 0.],
        [0., 1., 1.],
        [0., 2., 0.],
        [0., 2., 1.],
        [1., 0., 0.],
        [1., 0., 1.],
        [1., 1., 0.],
        [1., 1., 1.],
        [1., 2., 0.],
        [1., 2., 1.],
        [2., 0., 0.],
        [2., 0., 1.],
        [2., 1., 0.],
        [2., 1., 1.],
    ]
    cell_vert_map = [
        [0, 2, 8, 6, 1, 7, 9, 3],
        [2, 4, 10, 8, 3, 9, 11, 5],
        [6, 8, 14, 12, 7, 13, 15, 9],
    ]
    plex = plex_from_cell_list(3, cell_vert_map, vertices, COMM_WORLD)
    mesh = Mesh(
        plex,
        distribution_parameters={
            "partition": True,
            "partitioner_type": "simple",
            "overlap_type": (DistributedMeshOverlapType.RIDGE, 1),
        },
    )
    x, y, z = SpatialCoordinate(mesh)
    facet0 = And(And(x > 0.9, x < 1.1), y > 0.9)
    facet1 = And(And(y > 0.9, y < 1.1), x > 0.9)
    V = FunctionSpace(mesh, "Q", 2)
    f = Function(V).interpolate(conditional(Or(facet0, facet1), 1, 0))
    mesh = RelabeledMesh(mesh, [f], [999])
    subm = Submesh(mesh, mesh.topological_dimension() - 1, 999)
    v = assemble(Constant(1.) * dx(domain=subm))
    assert abs(v - 2.) < 2.e-15

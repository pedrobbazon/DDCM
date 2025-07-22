    # Create a vertex only mesh using these coordinates and use the
    # input ordering property to copy in the field values
    vom = VertexOnlyMesh(src_mesh, coordinates)
    P0DG_io = FunctionSpace(vom.input_ordering, "DG", 0)
    field_vomio = Function(P0DG_io)
    field_vomio.dat.data_wo[:] = field_list

    # We now interpolate onto the vertex only mesh that does not have
    # the input ordering
    P0DG = FunctionSpace(vom, "DG", 0)
    field_vom = interpolate(field_vomio, P0DG)
    print(field_vom.dat.data.min(), field_vom.dat.data.max())

    Vsrc = FunctionSpace(src_mesh, "CG", 1)
    I = Interpolator(TestFunction(Vsrc), P0DG)
    f_star = field_vom.riesz_representation(riesz_map="l2")
    f_data_star = Cofunction(Vsrc.dual())
    I.interpolate(f_star, adjoint=True, output=f_data_star)
    field = f_data_star.riesz_representation(riesz_map="l2")
    print(field.dat.data.min(), field.dat.data.max())

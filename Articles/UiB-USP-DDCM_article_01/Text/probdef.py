 Nx, Ny = 20, 20
 Lx, Ly = 1.0, 1.0
 msh = RectangleMesh(Nx, Ny, Lx, Ly, quadrilateral=False)
 h = CellDiameter(msh)
 pars = {'zeda': Constant(1.0), 'kappa': Constant(1.0),
         'alpha': Constant(0.125), 'gamma': Constant(0.125),
         'eta': Constant(0.5), 
         'beta': Constant(0.5), 'lmu': h
         'theta': Constant(0.5), 'ls': h,
         'formulation': "Equal-Order"}
 x = SpatialCoordinate(msh)

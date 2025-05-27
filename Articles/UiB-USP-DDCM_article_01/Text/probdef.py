# Geometry
Nx, Ny = 20, 20
Lx, Ly = 1.0, 1.0
msh = RectangleMesh(Nx, Ny, Lx, Ly, quadrilateral=False)
x = SpatialCoordinate(msh) 
h = CellDiameter(msh)

# Parameters and Manufacuted fields  
pars = {'zeta':  fd.Constant(1.0), 
        'kappa': fd.Constant(1.0),
        'alpha': fd.Constant(0.125),   
        'gamma': fd.Constant(0.125),
        'eta':   fd.Constant(0.5), 
        'beta':  fd.Constant(0.5), 'lmu': h,
        'theta': fd.Constant(0.5), 'ls': h,
        'formulation': 'Equal-Order'
    }

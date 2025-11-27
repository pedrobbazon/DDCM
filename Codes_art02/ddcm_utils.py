import firedrake as fd
import gmsh as gm
import matplotlib.pyplot as plt

def PlotPoissonData(lst_data_files, flag_type_plot = 'magnitude'):
    
    plt.figure(figsize= (10,6))
    plt.rcParams['font.size'] = '10'
    plt.axes(facecolor = "lightgray")

    c = ['r', 'b', 'g', 'm', 'y']

    for i, data_file in enumerate(lst_data_files):

        data = np.loadtxt(data_file)
        nc = data.shape[1]
        nsd = int(nc/2)

        gradu = data[:,0:nsd]
        flux  = data[:,nsd:2*nsd]
        
        if(flag_type_plot == 'magnitude'):
        
            plt.xlabel('$||grad(u)||_2$', fontsize=10)
            plt.ylabel('$||\sigma||_2$',fontsize=10)
            magflux = np.linalg.norm(flux, axis=1)
            maggrad = np.linalg.norm(gradu, axis=1)
            l = plt.plot(maggrad, magflux, 'o'+c[i])
        
        elif(flag_type_plot == 'x'):
        
            plt.xlabel('$\partial_x{u}$', fontsize=10)
            plt.ylabel('$\sigma_x$',fontsize=10)
            l = plt.plot(gradu[:,0], flux[:,0],'or')
        
        elif(flag_type_plot == 'y'):
            plt.xlabel('$\partial_y{u}$', fontsize=10)
            plt.ylabel('$\sigma_y$',fontsize=10)
            l = plt.plot(gradu[:,1], flux[:,1],'or')

        elif(flag_type_plot == 'z'):
            plt.xlabel('$\partial_z{u}$', fontsize=10)
            plt.ylabel('$\sigma_z$',fontsize=10)
            l = plt.plot(gradu[:,2], flux[:,2],'or')
        
    plt.setp(l, 'markersize', 4)
    plt.setp(l, 'markerfacecolor', 'white')
    plt.grid()
    plt.show()


def RotationMatrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], 
                     [np.sin(theta), np.cos(theta)]])

def RotateData2D(angi, angf, Nang, EP, gradu_sampled_vals, flux_sampled_vals, gradu_samples, flux_samples):
    
    aux1 = np.empty(shape=(1,2), dtype=np.float64)
    for angle in np.linspace(angi, angf, Nang, endpoint=EP):
        R = RotationMatrix(angle)
        gradu_rot_values = np.reshape([R.dot(vec) for vec in gradu_sampled_vals], (-1, 2))
        gradu_samples.append(gradu_rot_values)
        flux_rot_values = np.reshape([R.dot(vec) for vec in flux_sampled_vals], (-1, 2))
        flux_samples.append(flux_rot_values)
    
    return gradu_samples, flux_samples

def FitForcheimerParams(data_file):
    
    data = np.loadtxt(data_file)
    nc = data.shape[1]
    nsd = int(nc/2)

    gradu = data[:,0:nsd]
    flux  = data[:,nsd:2*nsd]
    modgrad = np.linalg.norm(gradu,axis=1)
    modflux = np.linalg.norm(flux,axis=1)
    

    n = np.size(modgrad)
    degree = 2
    A = np.zeros(shape=(n,2), dtype=np.float64)
    A[:,0] = modflux
    A[:,1] = modflux**2

    if(False):
        p = np.polyfit(modflux, modgrad, degree)
    else:
        paux = np.linalg.lstsq(A, modgrad, rcond=None)[0]
        p = np.array([paux[1], paux[0], 0.0])
        print(p)

    fluxeval = np.linspace(min(modflux),max(modflux))
    gradeval = np.polyval(p, fluxeval)

    plt.figure(figsize= (9,6))
    plt.rcParams['font.size'] = '16'
    plt.rcParams['axes.linewidth'] = 2.5
    plt.axes(facecolor = "lightgray")
    plt.xlabel('{\$g\$}',fontsize=20)
    plt.ylabel('{\$s\$}', fontsize=20)
    l = plt.plot(gradeval, fluxeval, '-r', modgrad, modflux, 'ob')
    plt.setp(l, 'markersize', 10)
    plt.setp(l, 'markerfacecolor', 'white')
    plt.legend(['{\$\\textbf{Linear regression}\$}', '{\$\\textbf{Experimenal data}\$}'])
    plt.grid()
    plt.savefig('data_vs_fit.svg')
    plt.show()
    print(p)
    plt.show()

    return 0

def FitDarcyParams(data_file):
    
    data = np.loadtxt(data_file)
    nc = data.shape[1]
    nsd = int(nc/2)

    gradu = data[:,0:nsd]
    flux  = data[:,nsd:2*nsd]
    modgrad = np.linalg.norm(gradu,axis=1)
    modflux = np.linalg.norm(flux,axis=1)
    
    n = np.size(modgrad)
    degree = 1
    A = np.zeros(shape=(n,1), dtype=np.float64)
    A[:,0] = modflux

    if(True):
        #p = np.polyfit(modflux, modgrad, degree)
        paux = np.linalg.lstsq(A, modgrad, rcond=None)[0]
        p = np.array([paux[0], 0.0])
        print(p)
    else:
        paux = np.linalg.lstsq(A, modgrad, rcond=None)[0]
        p = np.array([paux[1], paux[0], 0.0])
        print(p)

    fluxeval = np.linspace(min(modflux),max(modflux))
    gradeval = np.polyval(p, fluxeval)

    plt.figure(figsize= (9,6))
    plt.rcParams['font.size'] = '16'
    plt.rcParams['axes.linewidth'] = 2.5
    plt.axes(facecolor = "lightgray")
    plt.xlabel('\$g\$',fontsize=20)
    plt.ylabel('\$s\$', fontsize=20)
    l = plt.plot(gradeval, fluxeval, '-r', modgrad, modflux, 'ob')
    plt.setp(l, 'markersize', 10)
    plt.setp(l, 'markerfacecolor', 'white')
    plt.legend(['\$\\textbf{Linear regression}\$', '\$\\textbf{Experimenal data}\$'])
    plt.grid()
    plt.savefig('data_vs_fit.svg')
    plt.show()
    print(p)
    plt.show()

    return 0

import gmsh 
import numpy as np

#--------------------------------------------------------------------
#--- Preprocess: Mesh generation, boundary and region identification

def CreateMesh3Inclusions(meshname, hsize):

    # User defined data
    bottom_wall = 3
    right_wall  = 2
    top_wall    = 4
    left_wall   = 1

    inclusion_marker = 3
    background_marker = 2

    ninclusions = 3
    Lx = 2.0
    Ly = 2.0
    R1 = 0.2
    R2 = 0.3
    R3 = 0.25
    c1x, c1y = 0.85, 1.65
    c2x, c2y = 1.5, 1.05
    c3x, c3y = 0.5, 0.55

    gmsh.initialize()

    mass1 = np.pi*R1**2
    mass2 = np.pi*R2**2
    mass3 = np.pi*R3**2
    mass_inc = mass1 + mass2 + mass3

    # We create one rectangle and the circular inclusion
    background = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
    inclusion1 = gmsh.model.occ.addDisk(c1x, c1y, 0, R1, R1)
    inclusion2 = gmsh.model.occ.addDisk(c2x, c2y, 0, R2, R2)
    inclusion3 = gmsh.model.occ.addDisk(c3x, c3y, 0, R3, R3)
    #gmsh.model.occ.synchronize()
    all_inclusions = [(2, inclusion1)]
    all_inclusions.extend([(2, inclusion2)])
    all_inclusions.extend([(2, inclusion3)])
    whole_domain = gmsh.model.occ.fragment([(2, background)], all_inclusions)
    gmsh.model.occ.synchronize()

    background_surfaces = []
    other_surfaces = []
    tags = {'bottom_wall': bottom_wall, 'right_wall': right_wall, 'top_wall': top_wall, 'left_wall': left_wall, 'inclusion_marker': 3, 'background_marker': 2}
    for domain in whole_domain[0]:
        com = gmsh.model.occ.getCenterOfMass(domain[0], domain[1])
        mass = gmsh.model.occ.getMass(domain[0], domain[1])
        print(com, mass)
        # Identify the square by its mass
        if np.isclose(mass, (Lx*Ly - mass_inc)):
            #print('0:', com)
            gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=background_marker)
            background_surfaces.append(domain)
        # Identify the inner circles by its center of mass
        elif np.isclose(np.linalg.norm(com), np.sqrt((c1x)**2 + (c1y)**2)):
            #print('1:', com)
            gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=inclusion_marker)
            other_surfaces.append(domain)
        elif np.isclose(np.linalg.norm(com), np.sqrt((c2x)**2 + (c2y)**2)):
            #print('2:', com)
            gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=inclusion_marker+1)
            other_surfaces.append(domain)
        elif np.isclose(np.linalg.norm(com), np.sqrt((c3x)**2 + (c3y)**2)):
            #print('3:', com)
            gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=inclusion_marker+2)
            other_surfaces.append(domain)
        
    # Tag the left and right boundaries
    left = []
    right = []
    bottom = []
    top = []
    for line in gmsh.model.getEntities(dim=1):
        com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
        if np.isclose(com[0], 0.0):
            left.append(line[1])
        if np.isclose(com[0], Lx):
            right.append(line[1])
        if np.isclose(com[1], 0.0):
            bottom.append(line[1])
        if np.isclose(com[1], Ly):
            top.append(line[1])
    gmsh.model.addPhysicalGroup(1, left, left_wall)
    gmsh.model.addPhysicalGroup(1, right, right_wall)
    gmsh.model.addPhysicalGroup(1, bottom, bottom_wall)
    gmsh.model.addPhysicalGroup(1, top, top_wall)

    if(False):
        gmsh.model.mesh.field.add("Distance", 1)
        edges = gmsh.model.getBoundary(other_surfaces, oriented=False)
        gmsh.model.mesh.field.setNumbers(1, "EdgesList", [e[1] for e in edges])
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        r = 0.02
        gmsh.model.mesh.field.setNumber(2, "LcMin", r / 4)
        gmsh.model.mesh.field.setNumber(2, "LcMax", 3 * r)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 3 * r)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 3 * r)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)
        # Generate mesh
        gmsh.option.setNumber("Mesh.Algorithm", 2)
        gmsh.model.mesh.generate(2) 
    else:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), hsize)
        gmsh.model.mesh.generate(2)
        
    filename = meshname+".msh"
    gmsh.write(filename)
    gmsh.finalize()

    return filename, tags

def CreateMesh4Inclusions(meshname, hsize):

    # User defined data
    bottom_wall = 3
    right_wall  = 2
    top_wall    = 4
    left_wall   = 1

    inclusion_marker = 6
    background_marker = 5

    ninclusions = 3
    Lx = 1.0
    Ly = 1.0
    R1 = 0.2
    R2 = 0.2
    R3 = 0.2
    R4 = 0.2
    c1x, c1y = 0.255, 0.25
    c2x, c2y = 0.75, 0.255
    c3x, c3y = 0.75, 0.75
    c4x, c4y = 0.25, 0.75

    gmsh.initialize()

    mass1 = np.pi*R1**2
    mass2 = np.pi*R2**2
    mass3 = np.pi*R3**2
    mass4 = np.pi*R4**2
    mass_inc = mass1 + mass2 + mass3 + mass4

    # We create one rectangle and the circular inclusion
    background = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
    inclusion1 = gmsh.model.occ.addDisk(c1x, c1y, 0, R1, R1)
    inclusion2 = gmsh.model.occ.addDisk(c2x, c2y, 0, R2, R2)
    inclusion3 = gmsh.model.occ.addDisk(c3x, c3y, 0, R3, R3)
    inclusion4 = gmsh.model.occ.addDisk(c4x, c4y, 0, R4, R4)
    #gmsh.model.occ.synchronize()
    all_inclusions = [(2, inclusion1)]
    all_inclusions.extend([(2, inclusion2)])
    all_inclusions.extend([(2, inclusion3)])
    all_inclusions.extend([(2, inclusion4)])
    whole_domain = gmsh.model.occ.fragment([(2, background)], all_inclusions)
    gmsh.model.occ.synchronize()

    background_surfaces = []
    other_surfaces = []
    tags = {'bottom_wall': bottom_wall, 'right_wall': right_wall, 'top_wall': top_wall, 'left_wall': left_wall, 
            'inclusion_marker': inclusion_marker, 'background_marker': background_marker}
    for domain in whole_domain[0]:
        com = gmsh.model.occ.getCenterOfMass(domain[0], domain[1])
        mass = gmsh.model.occ.getMass(domain[0], domain[1])
        print(com, mass)
        # Identify the square by its mass
        if np.isclose(mass, (Lx*Ly - mass_inc)):
            gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=background_marker)
            background_surfaces.append(domain)
        # Identify the inner circles by its center of mass
        elif np.isclose(np.linalg.norm(com), np.sqrt((c1x)**2 + (c1y)**2)):
            gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=inclusion_marker)
            other_surfaces.append(domain)
        elif np.isclose(np.linalg.norm(com), np.sqrt((c2x)**2 + (c2y)**2)):
            gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=inclusion_marker+1)
            other_surfaces.append(domain)
        elif np.isclose(np.linalg.norm(com), np.sqrt((c3x)**2 + (c3y)**2)):
            gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=inclusion_marker+2)
            other_surfaces.append(domain)
        elif np.isclose(np.linalg.norm(com), np.sqrt((c4x)**2 + (c4y)**2)):
            gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=inclusion_marker+3)
            other_surfaces.append(domain)
        
    # Tag the left and right boundaries
    left = []
    right = []
    bottom = []
    top = []
    for line in gmsh.model.getEntities(dim=1):
        com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
        if np.isclose(com[0], 0.0):
            left.append(line[1])
        if np.isclose(com[0], Lx):
            right.append(line[1])
        if np.isclose(com[1], 0.0):
            bottom.append(line[1])
        if np.isclose(com[1], Ly):
            top.append(line[1])
    gmsh.model.addPhysicalGroup(1, left, left_wall)
    gmsh.model.addPhysicalGroup(1, right, right_wall)
    gmsh.model.addPhysicalGroup(1, bottom, bottom_wall)
    gmsh.model.addPhysicalGroup(1, top, top_wall)

    if(False):
        gmsh.model.mesh.field.add("Distance", 1)
        edges = gmsh.model.getBoundary(other_surfaces, oriented=False)
        gmsh.model.mesh.field.setNumbers(1, "EdgesList", [e[1] for e in edges])
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        r = 0.02
        gmsh.model.mesh.field.setNumber(2, "LcMin", r / 4)
        gmsh.model.mesh.field.setNumber(2, "LcMax", 3 * r)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 3 * r)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 3 * r)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)
        # Generate mesh
        gmsh.option.setNumber("Mesh.Algorithm", 2)
        gmsh.model.mesh.generate(2) 
    else:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), hsize)
        gmsh.model.mesh.generate(2)
        
    filename = meshname+".msh"
    gmsh.write(filename)
    gmsh.finalize()

    return filename, tags

def CreateMesh1Inclusions(meshname):

    ninclusions = 1
    Lx = 2.0
    Ly = 2.0
    R1 = 0.5

    gmsh.initialize()

    mass1 = np.pi*R1**2
    mass_inc = mass1

    proc = 0
    if proc == 0:
        # We create one rectangle and the circular inclusion
        background = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
        inclusion1 = gmsh.model.occ.addDisk(1.0, 1.0, 0, R1, R1)
        gmsh.model.occ.synchronize()
        all_inclusions = [(2, inclusion1)]
        whole_domain = gmsh.model.occ.fragment([(2, background)], all_inclusions)
        gmsh.model.occ.synchronize()

        background_surfaces = []
        other_surfaces = []
        tags = {'bottom_wall': 0, 'right_wall': 1, 'top_wall': 2, 'left_wall': 3, 'inclusion_marker': 3, 'background_marker': 2}
        for domain in whole_domain[0]:
            com = gmsh.model.occ.getCenterOfMass(domain[0], domain[1])
            mass = gmsh.model.occ.getMass(domain[0], domain[1])
            #print(mass, com)
            # Identify the square by its mass
            if np.isclose(mass, (Lx*Ly - mass_inc)):
                gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=background_marker)
                background_surfaces.append(domain)
            # Identify the inner circle by its center of mass
            elif np.isclose(np.linalg.norm(com), np.sqrt((1.0)**2 + (1.0)**2)):
                gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=inclusion_marker)
    
        # Tag the left and right boundaries
        left = []
        right = []
        for line in gmsh.model.getEntities(dim=1):
            com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
            if np.isclose(com[0], 0.0):
                left.append(line[1])
            if np.isclose(com[0], Lx):
                right.append(line[1])
        gmsh.model.addPhysicalGroup(1, left, left_wall)
        gmsh.model.addPhysicalGroup(1, right, right_wall)

    if(False):
        gmsh.model.mesh.field.add("Distance", 1)
        edges = gmsh.model.getBoundary(other_surfaces, oriented=False)
        gmsh.model.mesh.field.setNumbers(1, "EdgesList", [e[1] for e in edges])
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        r = 0.04
        gmsh.model.mesh.field.setNumber(2, "LcMin", r / 2)
        gmsh.model.mesh.field.setNumber(2, "LcMax", 2 * r)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 1 * r)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 2 * r)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)
        # Generate mesh
        gmsh.option.setNumber("Mesh.Algorithm", 2)
        gmsh.model.mesh.generate(2) 
    else:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.02)
        gmsh.model.mesh.generate(2)
        
    filename = meshname+".msh"
    gmsh.write(filename)
    gmsh.finalize()

    return filename, tags

def CreateMeshReentrantCornerOriginal(meshname, case=1, lc=0.1, flag_singularity_refinement=True):
    # Cases 1, 2 and 3 consider non-convex geometries. 
    # Case 4, on the other hand, consider the only convex geometry
    gmsh.initialize()
    if(case == 0 or case == 1):
        gmsh.model.geo.addPoint(-1, -1, 0, lc, 1)
        if(case == 0):
            gmsh.model.geo.addPoint(1, -1, 0, lc, 2)
        elif(case == 1):
            gmsh.model.geo.addPoint(0, -1, 0, lc, 2)
        gmsh.model.geo.addPoint(0, 0, 0, lc, 3)
        gmsh.model.geo.addPoint(1, 0, 0, lc, 4)
        gmsh.model.geo.addPoint(1, 1, 0, lc, 5)
        gmsh.model.geo.addPoint(-1, 1, 0, lc, 6)
        gmsh.model.geo.addLine(1, 2, 1)
        gmsh.model.geo.addLine(2, 3, 2)
        gmsh.model.geo.addLine(3, 4, 3)
        gmsh.model.geo.addLine(4, 5, 4)
        gmsh.model.geo.addLine(5, 6, 5)
        gmsh.model.geo.addLine(6, 1, 6)
        gmsh.model.geo.addCurveLoop([1, 2, 3, 4, 5, 6], 1)
        gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4, 5, 6], 101)
    elif(case == 2):
        gmsh.model.geo.addPoint(-1, -1, 0, lc, 1)
        gmsh.model.geo.addPoint(0, 0, 0, lc, 2)
        gmsh.model.geo.addPoint(1, 0, 0, lc, 3)
        gmsh.model.geo.addPoint(1, 1, 0, lc, 4)
        gmsh.model.geo.addPoint(-1, 1, 0, lc, 5)

        gmsh.model.geo.addLine(1, 2, 1)
        gmsh.model.geo.addLine(2, 3, 2)
        gmsh.model.geo.addLine(3, 4, 3)
        gmsh.model.geo.addLine(4, 5, 4)
        gmsh.model.geo.addLine(5, 1, 5)
        gmsh.model.geo.addCurveLoop([1, 2, 3, 4, 5], 1)
        gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4, 5], 101)
    elif(case == 3):
        gmsh.model.geo.addPoint(-1, -1, 0, lc, 1)
        gmsh.model.geo.addPoint(1, -1, 0, lc, 2)
        gmsh.model.geo.addPoint(1, 1, 0, lc, 3)
        gmsh.model.geo.addPoint(-1, 1, 0, lc, 4)
        gmsh.model.geo.addLine(1, 2, 1)
        gmsh.model.geo.addLine(2, 3, 2)
        gmsh.model.geo.addLine(3, 4, 3)
        gmsh.model.geo.addLine(4, 1, 4)
        gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
        gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4], 101)
    else:
        raise ValueError('case not recognized in CreateMeshReentrantCorner\n\n')
    
    ps = gmsh.model.addPhysicalGroup(2, [1])
    gmsh.model.setPhysicalName(2, ps, "My surface") 

    # Determine the tag of the boundaries of the domain
    onboundary = []
    for line in gmsh.model.getEntities(dim=1):
        onboundary.append(line[1])
    gmsh.model.addPhysicalGroup(1, onboundary, 1)
    tags = {'on_boundary': 1}
    
    # Determine the tag of the singular point
    singular_marker = []
    for point in gmsh.model.getEntities(dim=0):
        dim, tag = point[0], point[1]
        coords = gmsh.model.getValue(dim, tag, [])
        if np.all(np.isclose(coords, [0, 0, 0])):
            singular_marker.append(tag)

    if flag_singularity_refinement:
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "NodesList", singular_marker)
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        rmin, rmax = 0.1, 0.25
        gmsh.model.mesh.field.setNumber(2, "LcMin", lc/2)
        gmsh.model.mesh.field.setNumber(2, "LcMax", 2*lc)
        gmsh.model.mesh.field.setNumber(2, "DistMin", rmin)
        gmsh.model.mesh.field.setNumber(2, "DistMax", rmax)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)
    
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    
    filename = meshname+".msh"
    gmsh.write(filename)
    gmsh.write(meshname+".vtk") # For visualization in Paraview
    gmsh.finalize()

    return filename, tags


def CreateMeshReentrantCorner(meshname, case=0, lc=0.1, flag_singularity_refinement=True):
    # Cases 1, 2 and 3 consider non-convex geometries.
    # Case 4, on the other hand, consider the only convex geometry
    gm.initialize()
    gm.model.add("my_model")
    if (case == 0 or case == 1):
        gm.model.geo.addPoint(-1, -1, 0, lc, 1)
        if (case == 0):
            gm.model.geo.addPoint(1, -1, 0, lc, 2)
        elif (case == 1):
            gm.model.geo.addPoint(0, -1, 0, lc, 2)
        gm.model.geo.addPoint(0, 0, 0, lc, 3)
        gm.model.geo.addPoint(1, 0, 0, lc, 4)
        gm.model.geo.addPoint(1, 1, 0, lc, 5)
        gm.model.geo.addPoint(-1, 1, 0, lc, 6)
        gm.model.geo.addLine(1, 2, 1)
        gm.model.geo.addLine(2, 3, 2)
        gm.model.geo.addLine(3, 4, 3)
        gm.model.geo.addLine(4, 5, 4)
        gm.model.geo.addLine(5, 6, 5)
        gm.model.geo.addLine(6, 1, 6)
        gm.model.geo.addCurveLoop([1, 2, 3, 4, 5, 6], 1)
        gm.model.geo.addPlaneSurface([1], 1)
        gm.model.geo.synchronize()
    elif (case == 2):
        gm.model.geo.addPoint(-1, -1, 0, lc, 1)
        gm.model.geo.addPoint(0, 0, 0, lc, 2)
        gm.model.geo.addPoint(1, 0, 0, lc, 3)
        gm.model.geo.addPoint(1, 1, 0, lc, 4)
        gm.model.geo.addPoint(-1, 1, 0, lc, 5)
        gm.model.geo.addLine(1, 2, 1)
        gm.model.geo.addLine(2, 3, 2)
        gm.model.geo.addLine(3, 4, 3)
        gm.model.geo.addLine(4, 5, 4)
        gm.model.geo.addLine(5, 1, 5)
        gm.model.geo.addCurveLoop([1, 2, 3, 4, 5], 1)
        gm.model.geo.addPlaneSurface([1], 1)
        gm.model.geo.synchronize()
    else:
        raise ValueError('case not recognized in CreateMeshReentrantCorner\n')

    ps = gm.model.addPhysicalGroup(2, [1])
    gm.model.setPhysicalName(2, ps, "My surface")

    # Determine the tag of the boundaries of the domain
    tags = {}

    for dim, tag in gm.model.getEntities(dim=1):
        # pega os pontos extremos da linha
        boundary_points = gm.model.getBoundary([(dim, tag)], oriented=False)
        if len(boundary_points) == 2:
            p_start, p_end = boundary_points
        else:
            continue
        # obtém coordenadas de cada ponto
        x1, y1, z1 = gm.model.getValue(0, p_start[1], [])
        x2, y2, z2 = gm.model.getValue(0, p_end[1], [])
        # classifica cada fronteira
        if np.isclose(x1,-1.0) and np.isclose(x2,-1.0):
            tags.update({"left":tag})
        elif np.isclose(y1,-1.0) and np.isclose(y2,-1.0):
            tags.update({"bottom":tag})
        elif np.isclose(y1,1.0) and np.isclose(y2,1.0):
            tags.update({"top": tag})
        elif np.isclose(y1,0.0) and np.isclose(y2,0.0):
            tags.update({"corner_horizontal":tag})
        elif np.isclose(x1,1.0) and np.isclose(x2,1.0):
            tags.update({"right":tag})
        else:
            tags.update({"corner_slope":tag})

    for name, tag in tags.items():
        gm.model.addPhysicalGroup(1, [tag], tag)
        gm.model.setPhysicalName(1, tag, name)

    # Determine the tag of the singular point
    singular_marker = []
    for point in gm.model.getEntities(dim=0):
        dim, tag = point[0], point[1]
        coords = gm.model.getValue(dim, tag, [])
        if np.all(np.isclose(coords, [0, 0, 0])):
            singular_marker.append(tag)

    gmsh.model.addPhysicalGroup(0, singular_marker, tag=99)
    gmsh.model.setPhysicalName(0, 99, "SingularPoint")

    if flag_singularity_refinement:
        gm.model.mesh.field.add("Distance", 1)
        gm.model.mesh.field.setNumbers(1, "NodesList", singular_marker)
        gm.model.mesh.field.add("Threshold", 2)
        gm.model.mesh.field.setNumber(2, "IField", 1)
        rmin, rmax = 0.1, 0.25
        gm.model.mesh.field.setNumber(2, "LcMin", lc / 4)
        gm.model.mesh.field.setNumber(2, "LcMax", 2 * lc)
        gm.model.mesh.field.setNumber(2, "DistMin", rmin)
        gm.model.mesh.field.setNumber(2, "DistMax", rmax)
        gm.model.mesh.field.setAsBackgroundMesh(2)

    gm.option.setNumber("Mesh.Algorithm", 6)
    gm.model.geo.synchronize()
    gm.model.mesh.generate(2)

    filename = meshname + ".msh"
    gm.write(filename)
    gm.write(meshname + ".vtk")  # For visualization in Paraview
    gm.finalize()

    return filename, tags

# def CreateMeshReentrantCorner(meshname, case=1, lc=0.1, flag_singularity_refinement=True):
#     # Cases 1, 2 and 3 consider non-convex geometries.
#     # Case 4, on the other hand, consider the only convex geometry
#     gmsh.initialize()
#     if (case == 0 or case == 1):
#         gmsh.model.geo.addPoint(-1, -1, 0, lc, 1)
#         if (case == 0):
#             gmsh.model.geo.addPoint(1, -1, 0, lc, 2)
#         elif (case == 1):
#             gmsh.model.geo.addPoint(0, -1, 0, lc, 2)
#         gmsh.model.geo.addPoint(0, 0, 0, lc, 3)
#         gmsh.model.geo.addPoint(1, 0, 0, lc, 4)
#         gmsh.model.geo.addPoint(1, 1, 0, lc, 5)
#         gmsh.model.geo.addPoint(-1, 1, 0, lc, 6)
#         gmsh.model.geo.addLine(1, 2, 1)
#         gmsh.model.geo.addLine(2, 3, 2)
#         gmsh.model.geo.addLine(3, 4, 3)
#         gmsh.model.geo.addLine(4, 5, 4)
#         gmsh.model.geo.addLine(5, 6, 5)
#         gmsh.model.geo.addLine(6, 1, 6)
#         gmsh.model.geo.addCurveLoop([1, 2, 3, 4, 5, 6], 1)
#         gmsh.model.geo.addPlaneSurface([1], 1)
#         gmsh.model.geo.synchronize()
#         gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4, 5, 6], 101)
#     elif (case == 2):
#         gmsh.model.geo.addPoint(-1, -1, 0, lc, 1)
#         gmsh.model.geo.addPoint(0, 0, 0, lc, 2)
#         gmsh.model.geo.addPoint(1, 0, 0, lc, 3)
#         gmsh.model.geo.addPoint(1, 1, 0, lc, 4)
#         gmsh.model.geo.addPoint(-1, 1, 0, lc, 5)
#
#         gmsh.model.geo.addLine(1, 2, 1)
#         gmsh.model.geo.addLine(2, 3, 2)
#         gmsh.model.geo.addLine(3, 4, 3)
#         gmsh.model.geo.addLine(4, 5, 4)
#         gmsh.model.geo.addLine(5, 1, 5)
#         gmsh.model.geo.addCurveLoop([1, 2, 3, 4, 5], 1)
#         gmsh.model.geo.addPlaneSurface([1], 1)
#         gmsh.model.geo.synchronize()
#         gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4, 5], 101)
#     elif (case == 3):
#         gmsh.model.geo.addPoint(-1, -1, 0, lc, 1)
#         gmsh.model.geo.addPoint(1, -1, 0, lc, 2)
#         gmsh.model.geo.addPoint(1, 1, 0, lc, 3)
#         gmsh.model.geo.addPoint(-1, 1, 0, lc, 4)
#         gmsh.model.geo.addLine(1, 2, 1)
#         gmsh.model.geo.addLine(2, 3, 2)
#         gmsh.model.geo.addLine(3, 4, 3)
#         gmsh.model.geo.addLine(4, 1, 4)
#         gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
#         gmsh.model.geo.addPlaneSurface([1], 1)
#         gmsh.model.geo.synchronize()
#         gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4], 101)
#     else:
#         raise ValueError('case not recognized in CreateMeshReentrantCorner\n\n')
#
#     ps = gmsh.model.addPhysicalGroup(2, [1])
#     gmsh.model.setPhysicalName(2, ps, "My surface")
#
#     # Determine the tag of the boundaries of the domain
#     onboundary = []
#     for line in gmsh.model.getEntities(dim=1):
#         onboundary.append(line[1])
#     gmsh.model.addPhysicalGroup(1, onboundary, 1)
#     tags = {'on_boundary': 1}
#
#     # Determine the tag of the singular point
#     singular_marker = []
#     for point in gmsh.model.getEntities(dim=0):
#         dim, tag = point[0], point[1]
#         coords = gmsh.model.getValue(dim, tag, [])
#         if np.all(np.isclose(coords, [0, 0, 0])):
#             singular_marker.append(tag)
#
#     if flag_singularity_refinement:
#         gmsh.model.mesh.field.add("Distance", 1)
#         gmsh.model.mesh.field.setNumbers(1, "NodesList", singular_marker)
#         gmsh.model.mesh.field.add("Threshold", 2)
#         gmsh.model.mesh.field.setNumber(2, "IField", 1)
#         rmin, rmax = 0.1, 0.25
#         gmsh.model.mesh.field.setNumber(2, "LcMin", lc / 2)
#         gmsh.model.mesh.field.setNumber(2, "LcMax", 2 * lc)
#         gmsh.model.mesh.field.setNumber(2, "DistMin", rmin)
#         gmsh.model.mesh.field.setNumber(2, "DistMax", rmax)
#         gmsh.model.mesh.field.setAsBackgroundMesh(2)
#
#     gmsh.option.setNumber("Mesh.Algorithm", 6)
#     gmsh.model.geo.synchronize()
#     gmsh.model.mesh.generate(2)
#
#     filename = meshname + ".msh"
#     gmsh.write(filename)
#     gmsh.write(meshname + ".vtk")  # For visualization in Paraview
#     gmsh.finalize()
#
#     return filename, tags


def CreateMeshSquare(meshname, lc=0.1):
    gmsh.initialize()
    gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
    gmsh.model.geo.addPoint(1, 0, 0, lc, 2)
    gmsh.model.geo.addPoint(1, 1, 0, lc, 3)
    gmsh.model.geo.addPoint(0, 1, 0, lc, 4)
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4], 101)
    
    ps = gmsh.model.addPhysicalGroup(2, [1])
    gmsh.model.setPhysicalName(2, ps, "My surface") 

    # Determine the tag of the boundaries of the domain
    onboundary = []
    for line in gmsh.model.getEntities(dim=1):
        onboundary.append(line[1])
    gmsh.model.addPhysicalGroup(1, onboundary, 1)
    tags = {'on_boundary': 1}

    gmsh.option.setNumber("Mesh.Algorithm", 2)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    
    filename = meshname+".msh"
    gmsh.write(filename)
    gmsh.write(meshname+".vtk") # For visualization in Paraview
    gmsh.finalize()

    return filename, tags


def CreateMeshTaylorSheet(meshname, lc=0.1):
    L = 6.0
    H = 2.0
    L1 = 1.0
    L2 = L - 2.0*L1
    kk = 1.5
    a = 0.35
    lc1 = lc
    lc2 = 2*lc
    deltax = lc1
    Nparts = int(L2 / deltax)

    tags = {'right_wall': 1, 'top_wall': 2, 'left_wall': 3, 'bottom_left_wall': 4, 'sheet_wall': 5, 'bottom_right_wall': 6}
    gmsh.initialize()
    gmsh.model.geo.addPoint(L, 0, 0, lc1, 1)
    gmsh.model.geo.addPoint(L, H, 0, lc2, 2)
    gmsh.model.geo.addPoint(0.0, H, 0, lc2, 3)
    gmsh.model.geo.addPoint(0, 0, 0, lc1, 4)
    xx = L1
    npoints = 4
    for n in range(Nparts+1):
        yy = a*np.sin(2*kk*np.pi*(xx-L1)/L2)
        gmsh.model.geo.addPoint(xx, yy, 0, lc1, 5+n)
        xx += deltax
        npoints += 1
        
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 5, 4)
    right = [1]
    top = [2]
    left = [3]
    bottom_left = [4]
    lst_lines = [1,2,3,4]
    nlines = 4
    sheet = []
    for n in range(Nparts):
        gmsh.model.geo.addLine(5+n, 5+n+1, 5+n)
        lst_lines.append(5+n)
        sheet.append(5+n)
        nlines += 1

    gmsh.model.geo.addLine(npoints, 1, nlines+1)
    lst_lines.append(nlines+1)
    bottom_right = [nlines+1]
    nlines += 1
    gmsh.model.geo.addCurveLoop(lst_lines, 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()
    
    gmsh.model.addPhysicalGroup(1, left, tags['left_wall'])
    gmsh.model.addPhysicalGroup(1, right, tags['right_wall'])
    gmsh.model.addPhysicalGroup(1, top, tags['top_wall'])
    gmsh.model.addPhysicalGroup(1, bottom_left, tags['bottom_left_wall'])
    gmsh.model.addPhysicalGroup(1, bottom_right, tags['bottom_right_wall'])
    gmsh.model.addPhysicalGroup(1, sheet, tags['sheet_wall'])

    ps = gmsh.model.addPhysicalGroup(2, [1])
    gmsh.model.setPhysicalName(2, ps, "My surface") 

    gmsh.option.setNumber("Mesh.Algorithm", 2)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    
    filename = meshname+".msh"
    gmsh.write(filename)
    gmsh.write(meshname+".vtk") # For visualization in Paraview
    gmsh.finalize()

    mymesh = fd.Mesh(filename)
    return mymesh, tags




def create_mesh_internal_boundary_layer(Lx, Ly, h, flag_refine_interface):
    gmsh.initialize()
    gm = gmsh.model

    tags = {
        'leftdom_marker': 88, 'rightdom_marker': 99,
        'leftwall_marker': 1, 'rightwall_marker': 2,
        'bottomwall_marker': 3, 'topwall_marker': 4,
        'interface_marker': 5,
    }

    gm.occ.addRectangle(0, 0, 0, Lx/2, Ly, tag=1)
    gm.occ.addRectangle(Lx/2, 0, 0, Lx/2, Ly, tag=2)
    whole_domain = gm.occ.fragment([(2, 1)], [(2, 2)])
    gm.occ.synchronize()

    # Mark subdomains
    # --- Identify the fragmented surfaces ---
    surfaces = gm.getEntities(dim=2)
    leftdom, rightdom = [], []
    for dim, tag in surfaces:
        com = gm.occ.getCenterOfMass(dim, tag)
        if com[0] < Lx / 2:
            leftdom.append(tag)
        else:
            rightdom.append(tag)

    # --- Assign physical groups ---
    gm.addPhysicalGroup(2, leftdom, tags['leftdom_marker'])
    gm.addPhysicalGroup(2, rightdom, tags['rightdom_marker'])

    # Boundaries
    leftwall, rightwall, interface, bottomwall, topwall = [], [], [], [], []
    for dim, tag in gm.getEntities(dim=1):
        com = gm.occ.getCenterOfMass(dim, tag)
        if np.isclose(com[0], 0):    leftwall.append(tag)
        if np.isclose(com[0], Lx):   rightwall.append(tag)
        if np.isclose(com[0], Lx/2): interface.append(tag)
        if np.isclose(com[1], 0):    bottomwall.append(tag)
        if np.isclose(com[1], Ly):   topwall.append(tag)

    gm.addPhysicalGroup(1, leftwall, tags['leftwall_marker'])
    gm.addPhysicalGroup(1, rightwall, tags['rightwall_marker'])
    gm.addPhysicalGroup(1, bottomwall, tags['bottomwall_marker'])
    gm.addPhysicalGroup(1, topwall, tags['topwall_marker'])
    gm.addPhysicalGroup(1, interface, tags['interface_marker'])

    # Refinement
    if flag_refine_interface:
        gm.mesh.field.add("Distance", 1)
        gm.mesh.field.setNumbers(1, "EdgesList", interface)
        gm.mesh.field.add("Threshold", 2)
        gm.mesh.field.setNumber(2, "IField", 1)
        r = 0.01
        gm.mesh.field.setNumber(2, "LcMin", r / 2)
        gm.mesh.field.setNumber(2, "LcMax", 4 * r)
        gm.mesh.field.setNumber(2, "DistMin", 2 * r)
        gm.mesh.field.setNumber(2, "DistMax", 4 * r)
        gm.mesh.field.setAsBackgroundMesh(2)
    else:
        gm.mesh.setSize(gm.getEntities(0), h)

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gm.mesh.generate(2)

    meshname = "diffusion_reaction_mesh"
    gmsh.write(meshname + ".msh")
    gmsh.write(meshname + ".vtk")

    gmsh.finalize()
    return meshname + ".msh", tags


def error_seminorm(uex: fd.Function, uh: fd.Function, norm_type: str):
    error = uex - uh
    if norm_type == 'H1':
        return fd.sqrt(fd.assemble(fd.inner(fd.grad(error), fd.grad(error)) * fd.dx))
    elif norm_type == 'Hdiv':
        return fd.sqrt(fd.assemble(fd.inner(fd.div(error), fd.div(error)) * fd.dx))
    elif norm_type == 'Hcurl':
        return fd.sqrt(fd.assemble(fd.inner(fd.curl(error), fd.curl(error)) * fd.dx))
    else:
        raise ValueError(f"Semi-norm '{norm_type}' not implemented yet")

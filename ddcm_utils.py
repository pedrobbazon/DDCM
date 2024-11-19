import numpy as np
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

# User defined data
bottom_wall = 0
right_wall  = 1
top_wall    = 2
left_wall   = 3

inclusion_marker = 3
background_marker = 2

ninclusions = 3
Lx = 2.0
Ly = 2.0
R1 = 0.25
R2 = 0.15
R3 = 0.25

#--------------------------------------------------------------------
#--- Preprocess: Mesh generation, boundary and region identification

def CreateMeshInclusions(meshname):

    gmsh.initialize()

    mass1 = np.pi*R1**2
    mass2 = np.pi*R2**2
    mass3 = np.pi*R3**2
    mass_inc = mass1 + mass2 + mass3

    proc = 0
    if proc == 0:
        # We create one rectangle and the circular inclusion
        background = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
        inclusion1 = gmsh.model.occ.addDisk(0.5, 1.0, 0, R1, R1)
        inclusion2 = gmsh.model.occ.addDisk(1.0, 1.5, 0, R2, R2)
        inclusion3 = gmsh.model.occ.addDisk(1.5, 1.0, 0, R3, R3)
        gmsh.model.occ.synchronize()
        all_inclusions = [(2, inclusion1)]
        all_inclusions.extend([(2, inclusion2)])
        all_inclusions.extend([(2, inclusion3)])
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
            elif np.isclose(np.linalg.norm(com), np.sqrt((0.5)**2 + (1.0)**2)):
                gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=inclusion_marker)
                other_surfaces.append(domain)
            elif np.isclose(np.linalg.norm(com), np.sqrt((1.0)**2 + (1.5)**2)) and com[1] > 1.0:
                gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=inclusion_marker+1)
                other_surfaces.append(domain)
            elif np.isclose(np.linalg.norm(com), np.sqrt((1.5)**2 + (1.0)**2)):
                gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=inclusion_marker+2)
                other_surfaces.append(domain)
    
        # Tag the left and right boundaries
        left = []
        right = []
        for line in gmsh.model.getEntities(dim=1):
            com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
            if np.isclose(com[0], 0.0):
                #print('L', line)
                left.append(line[1])
            if np.isclose(com[0], Lx):
                #print('R', line)
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
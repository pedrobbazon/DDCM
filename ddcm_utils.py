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
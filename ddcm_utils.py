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
#-----------------------------------
# General libraries
#-----------------------------------
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

'''
This class has the following attributes: it reads the .npy file 
and extracts the characteristic mesh size, the errors, and the 
convergence rates. Also, it has three functions to produce all 
the convergence rate graphs. The first initialize all the plots, 
given the number of figures you want. The second plots the curves, 
given the figure, the data, the line type, and the label. The last 
one just saves the figures given the filename.
'''

class DDCM_visualization:
    
    def __init__(self, npy_file):
        self.npy_file = npy_file
        self.figures = {}

    def read_and_get_values(self):
        data = np.load(self.npy_file, allow_pickle=True).item()
        return data['norms'], data['errors'], data['conv_rates']

    def initialize_figures(self, num_plots):
        for fig_num in range(1, num_plots + 1):
            fig = plt.figure(fig_num, figsize=(8, 6))
            ax = fig.gca()
            ax.set_xlabel('$h$', fontsize=14)
            ax.set_ylabel('Error', fontsize=14)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.tick_params(axis='both', which='minor', labelsize=14)
            self.figures[fig_num] = fig

    def plot_lines(self, fig_num, hhs, data, marker, label,
                   title_legend = None, title_font = None):
        fig = self.figures[fig_num]
        ax = fig.gca()
        ax.loglog(hhs, data, marker, label=label)
        if title_legend:
            ax.legend(loc='upper left', fontsize=11, ncol=2, fancybox=True, shadow=True,
                      title=title_legend, title_fontsize=title_font or 12)
        else:
            ax.legend(loc='upper left', fontsize=11, ncol=2, fancybox=True, shadow=True)

    def save_figure(self, fig_num, fig_name, folder):
        fig = self.figures[fig_num]
        fig_path = os.path.join(folder, fig_name)
        fig.savefig(fig_path)
                      
# ------------------------------
# VMS-Primal
# ------------------------------
def run_plots_test_case1(foldername: str, order_poly: str):
    
    folder = os.path.join(foldername, order_poly)
    
    
    # Get .npy files from the folder
    for f in os.listdir(folder):
        if f.endswith('.npy'):
            file = f
    
    npy_file = os.path.join(folder, file)
    viz = DDCM_visualization(npy_file)
    norms, error, conv_rates = viz.read_and_get_values()
        
    # Set the number of figures
    viz.initialize_figures(num_plots=2)
    
    # Set the plot config according to the norm
    scalar_fields = {
        'L2u':   {'marker': 'g:^',   'label': r'$\|u - u_h\|_{L^2}$'},
        'H1u':   {'marker': 'r--o',  'label': r'$\|u - u_h\|_{H^1}$'},
        'L2l':   {'marker': 'b-.s',  'label': r'$\|\lambda - \lambda_h\|_{L^2}$'},
        'H1l':   {'marker': 'm--x',  'label': r'$\|\lambda - \lambda_h\|_{H^1}$'},
    }
    vector_fields = {
        'L2s':   {'marker': 'c-.*',  'label': r'$\|\mathbf{s} - \mathbf{s}_h\|_{L^2}$'},
        'L2e':   {'marker': 'y:d',   'label': r'$\|\mathbf{e} - \mathbf{e}_h\|_{L^2}$'},
        'L2m':   {'marker': 'k--v',  'label':  r'$\|\boldsymbol{\mu} - \boldsymbol{\mu}_h\|_{L^2}$'},
    }
    
    # Loop over figures
    for key, value in scalar_fields.items():
        viz.plot_lines(1, error['h'], error[key], **value)

    for key, value in vector_fields.items():
        viz.plot_lines(2, error['h'], error[key], **value)
        
    plt.show()
    
# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    main_selector = int(sys.argv[1])

    if main_selector == 1:
        run_plots_test_case1('01_VMS-Primal', order_poly = 'Linear')

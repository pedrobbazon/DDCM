#-----------------------------------
# General libraries
#-----------------------------------
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

'''
This class has the following attributes: it reads the .npy file and extracts the characteristic 
mesh size, the errors, and the convergence rates. Also, it has three functions to produce all the convergence rate
graphs. First, initialize all the plots, given the number of figures you want. The second plots the curves, given 
the figure, the data, the line type, and the label. The last one just saves the figures given the filename.
'''
class DDCM_visualization():
    
    def __init__(self, filename):
        self.filename = filename
        
    def read_and_get_values(self):
        data = np.load(f'{self.filename}.npy', allow_pickle=True).item()
        h_vals = data['h']
        errors = data['errors']
        conv_rates = data['conv_rates']
        return h_vals, errors, conv_rates

    def initialize_figures(self, num_plots):
        for fig_num in range(1, num_plots + 1):
            plt.figure(fig_num, figsize=(8, 6))
            plt.xlabel('$h$', fontsize=14)
            plt.ylabel('Error', fontsize=14)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            plt.tick_params(axis='both', which='major', labelsize=14)
            plt.tick_params(axis='both', which='minor', labelsize=14)

    def plot_lines(self, fig_num, hhs, data, line, legend):
        plt.figure(fig_num)
        plt.loglog(hhs, data, line, label=legend)
        plt.legend(fontsize=12)
        
    def save_figures(self, fig_num, fig_name, foldername):
        plt.figure(fig_num)
        plt.legend(loc='upper left', fontsize=11, ncol=2, fancybox=True, shadow=True)
        fig_path = os.path.join(foldername, fig_name)
        plt.savefig(fig_path)

'''
Program this however you want
'''

def run_plots_test_case1(foldername, order_poly):
    
    subfolder = os.path.join(foldername, order_poly)
    
    # Automatically get .npy files from the folder
    files = [f for f in os.listdir(subfolder) if f.endswith('.npy')]
    filenames = [os.path.splitext(f)[0] for f in files]

    # Read and store results
    data_forms = {fname: {'h': [], 'error': []} for fname in filenames}
    for fname in filenames:
        path = os.path.join(subfolder, fname)
        viz = DDCM_visualization(path)
        h, error, _ = viz.read_and_get_values()
        data_forms[fname]['h'] = h
        data_forms[fname]['error'] = error
        
    # You may need to manually sort `filenames` to match your label logic.
    # Below assumes same order as original hardcoded list.
    filenames_sorted = sorted(filenames, key=lambda x: (
        0 if 'Natural' in x else 
        1 if 'Unstabilized' in x else 
        2 if 'Fully' in x else 
        3 if 'Minimally' in x else 4))

    h = data_forms[filenames_sorted[0]]['h']
    
    # Relate the data to the figure
    norm_to_fig = {
        1: [
            {'data': data_forms[filenames_sorted[0]]['error']['L2u'],  'marker': 'r--o', 'label': r'$\|u - u_h\|_{L^2} \,$ Natural'},
            {'data': data_forms[filenames_sorted[1]]['error']['L2u'],  'marker': 'b-.s', 'label': r'$\|u - u_h\|_{L^2} \,$ Unstabilized'},
            {'data': data_forms[filenames_sorted[2]]['error']['L2u'],  'marker': 'g:^', 'label': r'$\|u - u_h\|_{L^2} \,$  Fully stabilized'},
            {'data': data_forms[filenames_sorted[3]]['error']['L2u'],  'marker': 'm--p', 'label': r'$\|u - u_h\|_{L^2} \,$  Minimally stabilized'},
        ],
        2:[
            {'data': data_forms[filenames_sorted[0]]['error']['H1u'],  'marker': 'r--o', 'label': r'$\|u - u_h\|_{H^1} \,$ Natural'},
            {'data': data_forms[filenames_sorted[1]]['error']['H1u'],  'marker': 'b-.s', 'label': r'$\|u - u_h\|_{H^1} \,$ Unstabilized'},
            {'data': data_forms[filenames_sorted[2]]['error']['H1u'],  'marker': 'g:^', 'label': r'$\|u - u_h\|_{H^1} \,$ Fully stabilized'},
            {'data': data_forms[filenames_sorted[3]]['error']['H1u'],  'marker': 'm--p', 'label': r'$\|u - u_h\|_{H^1} \,$ Minimally stabilized'},        
        ],
        3:[
            {'data': data_forms[filenames_sorted[0]]['error']['EsL2'], 'marker': 'r--o', 'label': r'$\|\mathbf{s} - \mathbf{s}_h\|_{L^2} \,$ Natural'},
            {'data': data_forms[filenames_sorted[0]]['error']['EeL2'], 'marker': 'b-.s', 'label': r'$\|\mathbf{e} - \mathbf{e}_h\|_{L^2} \,$ Natural'},
            {'data': data_forms[filenames_sorted[0]]['error']['EmL2'], 'marker': 'g:^',  'label': r'$\|\boldsymbol{\mu} - \boldsymbol{\mu}_h\|_{L^2} \,$ Natural'}
        ],
        4:[
            {'data': data_forms[filenames_sorted[1]]['error']['EsL2'], 'marker': 'r--o', 'label': r'$\|\mathbf{s} - \mathbf{s}_h\|_{L^2} \,$ Unstabilized'},
            {'data': data_forms[filenames_sorted[2]]['error']['EsL2'], 'marker': 'b-.s', 'label': r'$\|\mathbf{s} - \mathbf{s}_h\|_{L^2} \,$ Fully stabilized'},
            {'data': data_forms[filenames_sorted[3]]['error']['EsL2'], 'marker': 'g:^',  'label': r'$\|\mathbf{s} - \mathbf{s}_h\|_{L^2} \,$ Minimally stabilized'}
        ],
        5:[
            {'data': data_forms[filenames_sorted[1]]['error']['EeL2'], 'marker': 'r--o', 'label': r'$\|\mathbf{e} - \mathbf{e}_h\|_{L^2} \,$ Unstabilized'},
            {'data': data_forms[filenames_sorted[2]]['error']['EeL2'], 'marker': 'b-.s', 'label': r'$\|\mathbf{e} - \mathbf{e}_h\|_{L^2} \,$ Fully stabilized'},
            {'data': data_forms[filenames_sorted[3]]['error']['EeL2'], 'marker': 'g:^',  'label': r'$\|\mathbf{e} - \mathbf{e}_h\|_{L^2} \,$ Minimally stabilized'}
            
        ],  
        6:[
            {'data': data_forms[filenames_sorted[1]]['error']['EmL2'], 'marker': 'r--o', 'label': r'$\|\boldsymbol{\mu} - \boldsymbol{\mu}_h\|_{L^2} \,$ Unstabilized'},
            {'data': data_forms[filenames_sorted[2]]['error']['EmL2'], 'marker': 'b-.s', 'label': r'$\|\boldsymbol{\mu} - \boldsymbol{\mu}_h\|_{L^2} \,$ Fully stabilized'},
            {'data': data_forms[filenames_sorted[3]]['error']['EmL2'], 'marker': 'g:^',  'label': r'$\|\boldsymbol{\mu} - \boldsymbol{\mu}_h\|_{L^2} \,$ Minimally stabilized'}
            
        ],   
        7:[
            {'data': data_forms[filenames_sorted[1]]['error']['EsHdiv'], 'marker': 'r--o', 'label': r'$\|\mathbf{s} - \mathbf{s}_h\|_{Hdiv} \,$ Unstabilized'},
            {'data': data_forms[filenames_sorted[2]]['error']['EsHdiv'], 'marker': 'b-.s', 'label': r'$\|\mathbf{s} - \mathbf{s}_h\|_{Hdiv} \,$ Fully stabilized'},
            {'data': data_forms[filenames_sorted[3]]['error']['EsHdiv'], 'marker': 'g:^',  'label': r'$\|\mathbf{s} - \mathbf{s}_h\|_{Hdiv} \,$ Minimally stabilized'}
        ],
        8:[
            {'data': data_forms[filenames_sorted[1]]['error']['EeHdiv'], 'marker': 'r--o', 'label': r'$\|\mathbf{e} - \mathbf{e}_h\|_{Hdiv} \,$ Unstabilized'},
            {'data': data_forms[filenames_sorted[2]]['error']['EeHdiv'], 'marker': 'b-.s', 'label': r'$\|\mathbf{e} - \mathbf{e}_h\|_{Hdiv} \,$ Fully stabilized'},
            {'data': data_forms[filenames_sorted[3]]['error']['EeHdiv'], 'marker': 'g:^',  'label': r'$\|\mathbf{e} - \mathbf{e}_h\|_{Hdiv} \,$ Minimally stabilized'}
            
        ],  
        9:[
            {'data': data_forms[filenames_sorted[1]]['error']['EmHdiv'], 'marker': 'r--o', 'label': r'$\|\boldsymbol{\mu} - \boldsymbol{\mu}_h\|_{Hdiv} \,$ Unstabilized'},
            {'data': data_forms[filenames_sorted[2]]['error']['EmHdiv'], 'marker': 'b-.s', 'label': r'$\|\boldsymbol{\mu} - \boldsymbol{\mu}_h\|_{Hdiv} \,$ Fully stabilized'},
            {'data': data_forms[filenames_sorted[3]]['error']['EmHdiv'], 'marker': 'g:^',  'label': r'$\|\boldsymbol{\mu} - \boldsymbol{\mu}_h\|_{Hdiv} \,$ Minimally stabilized'}
            
        ], 
    }
    
    # Plot and save figures 
    viz = DDCM_visualization("") 
    viz.initialize_figures(len(norm_to_fig))
    for fig_num, lines in norm_to_fig.items():
        for entry in lines:
            viz.plot_lines(fig_num, h, entry['data'], entry['marker'], entry['label'])
        viz.save_figures(fig_num, f'figure_{fig_num}', subfolder)
    plt.show()
    
    

def run_plots_test_case2():
    'under development' 
       
# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    main_selector = int(sys.argv[1])
    
    order_poly = 'Linear'
    if (main_selector == 1):
        run_plots_test_case1('01_test_case_convex_domain', order_poly)




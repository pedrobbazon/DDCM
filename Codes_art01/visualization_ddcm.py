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
        return data['h'], data['errors'], data['conv_rates']

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

    def plot_lines(self, fig_num, hhs, data, line, legend,
                   title_legend = None, title_font = None):
        fig = self.figures[fig_num]
        ax = fig.gca()
        ax.loglog(hhs, data, line, label=legend)
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
# Test Case 1: Convex Domain
# ------------------------------
def run_plots_test_case1(foldername, order_poly):
    
    folder = os.path.join(foldername, order_poly)
    
    # Get .npy files from the folder
    files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    
    # Read and store results
    data_forms = {fname: {'h': [], 'error': []} for fname in files}
    for fname in files:
        npy_file = os.path.join(folder, fname)
        viz = DDCM_visualization(npy_file)
        h, error, _ = viz.read_and_get_values()
        data_forms[fname]['h'] = h
        data_forms[fname]['error'] = error

        
    # You may need to manually sort `filenames` to match your label logic.
    # Below assumes same order as original hardcoded list.
    filenames_sorted = sorted(files, key=lambda x: (
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
        3: [
            {'data': data_forms[filenames_sorted[0]]['error']['L2l'],  'marker': 'r--o', 'label': r'$\|\lambda - \lambda_h\|_{L^2} \,$ Natural'},
            {'data': data_forms[filenames_sorted[1]]['error']['L2l'],  'marker': 'b-.s', 'label': r'$\|\lambda - \lambda_h\|_{L^2} \,$ Unstabilized'},
            {'data': data_forms[filenames_sorted[2]]['error']['L2l'],  'marker': 'g:^', 'label': r'$\|\lambda - \lambda_h\|_{L^2} \,$  Fully stabilized'},
            {'data': data_forms[filenames_sorted[3]]['error']['L2l'],  'marker': 'm--p', 'label': r'$\|\lambda - \lambda_h\|_{L^2} \,$  Minimally stabilized'},
        ],
        4:[
            {'data': data_forms[filenames_sorted[0]]['error']['H1l'],  'marker': 'r--o', 'label': r'$\|\lambda - \lambda_h\|_{H^1} \,$ Natural'},
            {'data': data_forms[filenames_sorted[1]]['error']['H1l'],  'marker': 'b-.s', 'label': r'$\|\lambda - \lambda_h\|_{H^1} \,$ Unstabilized'},
            {'data': data_forms[filenames_sorted[2]]['error']['H1l'],  'marker': 'g:^', 'label': r'$\|\lambda - \lambda_h\|_{H^1} \,$ Fully stabilized'},
            {'data': data_forms[filenames_sorted[3]]['error']['H1l'],  'marker': 'm--p', 'label': r'$\|\lambda - \lambda_h\|_{H^1} \,$ Minimally stabilized'},        
        ],
        5:[
            {'data': data_forms[filenames_sorted[0]]['error']['EsL2'], 'marker': 'r--o', 'label': r'$\|\mathbf{s} - \mathbf{s}_h\|_{L^2} \,$ Natural'},
            {'data': data_forms[filenames_sorted[0]]['error']['EeL2'], 'marker': 'b-.s', 'label': r'$\|\mathbf{e} - \mathbf{e}_h\|_{L^2} \,$ Natural'},
            {'data': data_forms[filenames_sorted[0]]['error']['EmL2'], 'marker': 'g:^',  'label': r'$\|\boldsymbol{\mu} - \boldsymbol{\mu}_h\|_{L^2} \,$ Natural'}
        ],
        6:[
            {'data': data_forms[filenames_sorted[1]]['error']['EsL2'], 'marker': 'r--o', 'label': r'$\|\mathbf{s} - \mathbf{s}_h\|_{L^2} \,$ Unstabilized'},
            {'data': data_forms[filenames_sorted[2]]['error']['EsL2'], 'marker': 'b-.s', 'label': r'$\|\mathbf{s} - \mathbf{s}_h\|_{L^2} \,$ Fully stabilized'},
            {'data': data_forms[filenames_sorted[3]]['error']['EsL2'], 'marker': 'g:^',  'label': r'$\|\mathbf{s} - \mathbf{s}_h\|_{L^2} \,$ Minimally stabilized'}
        ],
        7:[
            {'data': data_forms[filenames_sorted[1]]['error']['EeL2'], 'marker': 'r--o', 'label': r'$\|\mathbf{e} - \mathbf{e}_h\|_{L^2} \,$ Unstabilized'},
            {'data': data_forms[filenames_sorted[2]]['error']['EeL2'], 'marker': 'b-.s', 'label': r'$\|\mathbf{e} - \mathbf{e}_h\|_{L^2} \,$ Fully stabilized'},
            {'data': data_forms[filenames_sorted[3]]['error']['EeL2'], 'marker': 'g:^',  'label': r'$\|\mathbf{e} - \mathbf{e}_h\|_{L^2} \,$ Minimally stabilized'}
            
        ],  
        8:[
            {'data': data_forms[filenames_sorted[1]]['error']['EmL2'], 'marker': 'r--o', 'label': r'$\|\boldsymbol{\mu} - \boldsymbol{\mu}_h\|_{L^2} \,$ Unstabilized'},
            {'data': data_forms[filenames_sorted[2]]['error']['EmL2'], 'marker': 'b-.s', 'label': r'$\|\boldsymbol{\mu} - \boldsymbol{\mu}_h\|_{L^2} \,$ Fully stabilized'},
            {'data': data_forms[filenames_sorted[3]]['error']['EmL2'], 'marker': 'g:^',  'label': r'$\|\boldsymbol{\mu} - \boldsymbol{\mu}_h\|_{L^2} \,$ Minimally stabilized'}
            
        ],   
        9:[
            {'data': data_forms[filenames_sorted[1]]['error']['EsHdiv'], 'marker': 'r--o', 'label': r'$\|\mathbf{s} - \mathbf{s}_h\|_{Hdiv} \,$ Unstabilized'},
            {'data': data_forms[filenames_sorted[2]]['error']['EsHdiv'], 'marker': 'b-.s', 'label': r'$\|\mathbf{s} - \mathbf{s}_h\|_{Hdiv} \,$ Fully stabilized'},
            {'data': data_forms[filenames_sorted[3]]['error']['EsHdiv'], 'marker': 'g:^',  'label': r'$\|\mathbf{s} - \mathbf{s}_h\|_{Hdiv} \,$ Minimally stabilized'}
        ],
        10:[
            {'data': data_forms[filenames_sorted[1]]['error']['EeHdiv'], 'marker': 'r--o', 'label': r'$\|\mathbf{e} - \mathbf{e}_h\|_{Hdiv} \,$ Unstabilized'},
            {'data': data_forms[filenames_sorted[2]]['error']['EeHdiv'], 'marker': 'b-.s', 'label': r'$\|\mathbf{e} - \mathbf{e}_h\|_{Hdiv} \,$ Fully stabilized'},
            {'data': data_forms[filenames_sorted[3]]['error']['EeHdiv'], 'marker': 'g:^',  'label': r'$\|\mathbf{e} - \mathbf{e}_h\|_{Hdiv} \,$ Minimally stabilized'}
            
        ],  
        11:[
            {'data': data_forms[filenames_sorted[1]]['error']['EmHdiv'], 'marker': 'r--o', 'label': r'$\|\boldsymbol{\mu} - \boldsymbol{\mu}_h\|_{Hdiv} \,$ Unstabilized'},
            {'data': data_forms[filenames_sorted[2]]['error']['EmHdiv'], 'marker': 'b-.s', 'label': r'$\|\boldsymbol{\mu} - \boldsymbol{\mu}_h\|_{Hdiv} \,$ Fully stabilized'},
            {'data': data_forms[filenames_sorted[3]]['error']['EmHdiv'], 'marker': 'g:^',  'label': r'$\|\boldsymbol{\mu} - \boldsymbol{\mu}_h\|_{Hdiv} \,$ Minimally stabilized'}
            
        ], 
    }
    
    # Plot and save figures 
    viz.initialize_figures(len(norm_to_fig))
    for fig_num, lines in norm_to_fig.items():
        for entry in lines:
            viz.plot_lines(fig_num, h, entry['data'], entry['marker'], entry['label'])
        viz.save_figure(fig_num, f'figure_{fig_num}.svg', folder)
    plt.show()
    
# ------------------------------
# Test Case 2: Non-Convex Domain
# ------------------------------
def run_plots_test_case2():
    '''
    For the second numerical experiment, we opted to present the results in a table. Thus, we do not generate any figure. Feel free edit 
    this function to generate the figures you want.
    '''
    
# ------------------------------
# Test Case 3: Data Assignation
# ------------------------------
def run_plots_test_case3(foldername, order_poly):
    
    # Access the directory where the results were saved
    folder = os.path.join(foldername, order_poly)
    
    # Get .npy files from the folder without the .npy extension
    files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    
    # Read and store results
    data_forms = {fname: {'h': [], 'error': []} for fname in files}
    for fname in files:
        path = os.path.join(folder, fname)
        viz = DDCM_visualization(path)
        h, error, _ = viz.read_and_get_values()
        data_forms[fname]['h'] = h
        data_forms[fname]['error'] = error
    
    
    Nnorms = ['L2u', 'H1u', 'L2l', 'H1l', 'EsL2', 'EeL2', 'EmL2']
    Normlabels = ['$\|u - u_h\|_{L^2}$', '$\|u - u_h\|_{H^1}$', '$\|\lambda - \lambda_h\|_{L^2}$',
                  '$\|\lambda - \lambda_h\|_{H^1}$', '$\|\mathbf{s} - \mathbf{s}_h\|_{L^2}$',
                  '$\|\mathbf{e} - \mathbf{e}_h\|_{L^2}$', '$\|\\boldsymbol{\mu} - \\boldsymbol{\mu}_h\|_{L^2}$']
    
    colors = ['r--o', 'b--o', 'g--o', 'c--o', 'm--o', 'y--o']
    Nsamples = [64, 128, 256, 512, 1024, 2048]
    
    def plot_formulation_results(data_forms, formulation_key, Nsamples, Normlabels, colors):
        viz.initialize_figures(7)
        for i, norm in enumerate(Nnorms):
            fig_num = i+1
            for j, Ns in enumerate(Nsamples):
                key = f'table_{formulation_key}_Ns{Ns}.npy'
                viz.plot_lines(fig_num, 
                               data_forms[key]['h'],
                               data_forms[key]['error'][norm],
                               colors[j],
                               f'$N_{{samples}} = {Ns} \\times {Ns}$',
                               title_legend=Normlabels[i], title_font=12)
            viz.save_figure(fig_num, f'figure_{formulation_key}_{fig_num}.svg', folder)
        plt.show()
    
    plot_formulation_results(data_forms, 'Natural', Nsamples, Normlabels, colors)
    plot_formulation_results(data_forms, 'Equal-Order_Unstabilized', Nsamples, Normlabels, colors)
    plot_formulation_results(data_forms, 'Equal-Order_Fully_stabilized', Nsamples, Normlabels, colors)
    plot_formulation_results(data_forms, 'Equal-Order_Minimally_stabilized', Nsamples, Normlabels, colors)
    
               
# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    main_selector = int(sys.argv[1])
    order_poly = 'Linear'

    if main_selector == 1:
        run_plots_test_case1('01_test_case_convex_domain', order_poly)
    elif main_selector == 2:
        run_plots_test_case2()
    elif main_selector == 3:
        run_plots_test_case3('03_test_case_data_assignation', order_poly)
    else:
        raise ValueError('Invalid main selector')

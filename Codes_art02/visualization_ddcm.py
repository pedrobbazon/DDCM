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

# ------------- Visualization helper -------------
class DDCM_visualization:

    def __init__(self, npy_file):
        self.npy_file = npy_file
        self.figures, self.axes = {}, {}

    def read_and_get_values(self):
        data = np.load(self.npy_file, allow_pickle=True).item()
        return data['norms'], data['errors'], data['conv_rates']

    def initialize_figures(self, num_plots):
        for fig_num in range(1, num_plots + 1):
            fig = plt.figure(fig_num, figsize=(8, 6))
            ax = fig.gca()
            ax.set_xlabel(r'$h$', fontsize=14)
            ax.set_ylabel('Error', fontsize=14)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            self.figures[fig_num] = fig
            self.axes[fig_num] = ax

    def plot_line(self, fig_num, h, y, style, label):
        ax = self.axes[fig_num]
        ax.loglog(h, y, style, label=label)

    def finalize(self, fig_num, title=None, legend_loc='lower right', ncol=2):
        ax = self.axes[fig_num]
        if title:
            ax.set_title(title)
        ax.legend(loc=legend_loc, fontsize=11, ncol=ncol, title="Equal-Order (solid) |  VMS-Primal (dashed)", fancybox=True, shadow=True)

    def save_figure(self, fig_num, fig_name, folder='.', order_poly='Linear'):
        fig = self.figures[fig_num]
        folder_path = os.path.join(folder, order_poly)
        os.makedirs(folder_path, exist_ok=True)

        # full file path for saving
        fig_path = os.path.join(folder_path, fig_name)
        fig.savefig(fig_path, bbox_inches='tight', dpi=200)
        return fig_path

# ------------------------------
# Equal-Order (gus) vs VMS-Primal (cod)
# ------------------------------
def run_plots_test_case0(order_poly: str):

    folder_gus = os.path.join('00_Equal-Order', order_poly)
    folder_cod = os.path.join('00_VMS-Primal', order_poly)

    # Find the .npy file in each folder
    file_gus = next(f for f in os.listdir(folder_gus) if f.endswith('.npy'))
    file_cod = next(f for f in os.listdir(folder_cod) if f.endswith('.npy'))

    npy_file_gus = os.path.join(folder_gus, file_gus)
    npy_file_cod = os.path.join(folder_cod, file_cod)

    viz_gus = DDCM_visualization(npy_file_gus)
    viz_cod = DDCM_visualization(npy_file_cod)

    _, err_gus, _ = viz_gus.read_and_get_values()
    _, err_cod, _ = viz_cod.read_and_get_values()

    # assume both use the same h vector; if not, we’ll plot on their own h
    h_gus = np.asarray(err_gus['h'])
    h_cod = np.asarray(err_cod['h'])

    # 2 figures: 1) scalar norms; 2) vector norms
    viz_gus.initialize_figures(num_plots=2)

    # Fixed colors per norm (easier to compare gus vs cod); styles/markers differ by formulation (chat GPT proposed colors)
    scalar_norms = {
        'L2u': ( 'blue', r'$\|u-u_h\|_{L^2}$' ),
        'H1u': ( 'olive', r'$\|u-u_h\|_{H^1}$' ),
        'L2l': ( 'green', r'$\|\lambda-\lambda_h\|_{L^2}$' ),
        'H1l': ( 'red', r'$\|\lambda-\lambda_h\|_{H^1}$' ),
    }
    vector_norms = {
        'L2s': ( 'purple', r'$\|\mathbf{s}-\mathbf{s}_h\|_{L^2}$' ),
        'L2e': ( 'magenta', r'$\|\mathbf{e}-\mathbf{e}_h\|_{L^2}$' ),
        'L2m': ( 'cyan', r'$\|\boldsymbol{\mu}-\boldsymbol{\mu}_h\|_{L^2}$' ),
    }

    # styles per formulation (same color per norm)
    style_gus = dict(linestyle='-',  marker='o',  markersize=4,  linewidth=1.2)
    style_cod = dict(linestyle='--', marker='s',  markersize=4,  linewidth=1.2)

    # ---- Scalars: plot each norm twice (gus & cod) ----
    for key, (color, label_base) in scalar_norms.items():
        # gus
        viz_gus.plot_line(
            1, h_gus, np.asarray(err_gus[key]),
            style=f"{style_gus['linestyle']}{style_gus['marker']}",
            label=f"{label_base}"
        )
        viz_gus.axes[1].lines[-1].set_color(color)
        viz_gus.axes[1].lines[-1].set_markersize(style_gus['markersize'])
        viz_gus.axes[1].lines[-1].set_linewidth(style_gus['linewidth'])

    for key, (color, label_base) in scalar_norms.items():
        # cod
        viz_gus.plot_line(
            1, h_cod, np.asarray(err_cod[key]),
            style=f"{style_cod['linestyle']}{style_cod['marker']}",
            label=f"{label_base}"
        )
        viz_gus.axes[1].lines[-1].set_color(color)
        viz_gus.axes[1].lines[-1].set_markersize(style_cod['markersize'])
        viz_gus.axes[1].lines[-1].set_linewidth(style_cod['linewidth'])

    viz_gus.finalize(1, None, legend_loc='lower right', ncol=2)

    # ---- Vectors: plot each norm twice (gus & cod) ----
    for key, (color, label_base) in vector_norms.items():
        # gus
        viz_gus.plot_line(
            2, h_gus, np.asarray(err_gus[key]),
            style=f"{style_gus['linestyle']}{style_gus['marker']}",
            label=f"{label_base}"
        )
        viz_gus.axes[2].lines[-1].set_color(color)
        viz_gus.axes[2].lines[-1].set_markersize(style_gus['markersize'])
        viz_gus.axes[2].lines[-1].set_linewidth(style_gus['linewidth'])

    for key, (color, label_base) in vector_norms.items():
        # cod
        viz_gus.plot_line(
            2, h_cod, np.asarray(err_cod[key]),
            style=f"{style_cod['linestyle']}{style_cod['marker']}",
            label=f"{label_base}"
        )
        viz_gus.axes[2].lines[-1].set_color(color)
        viz_gus.axes[2].lines[-1].set_markersize(style_cod['markersize'])
        viz_gus.axes[2].lines[-1].set_linewidth(style_cod['linewidth'])

    viz_gus.finalize(2, title=None, legend_loc='lower right', ncol=2)

    # Optionally save
    p1 = viz_gus.save_figure(1, f"scalars_{order_poly}.svg", 'new_results', order_poly = order_poly)
    print("Saved:", p1)
    p2 = viz_gus.save_figure(2, f"vectors_{order_poly}.svg", 'new_results', order_poly = order_poly)
    print("Saved:", p2)


    plt.show()

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
        run_plots_test_case0(order_poly='Quadratic')
    elif main_selector == 2:
        run_plots_test_case1('01_VMS-Primal', order_poly = 'Linear')

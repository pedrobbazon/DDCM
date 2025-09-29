# ===============================================
# General libraries
# ===============================================
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# ===============================================
# Our libraries
# ===============================================
import constants as ct

# ===============================================
# Functions
# ===============================================
def read_and_get_values(npy_file):
    data = np.load(npy_file, allow_pickle=True).item()
    return data['errors'], data['conv_rates']

def initialize_figures(num_plots: int):
    figures, axes = {}, {}
    for fig_num in range(1, num_plots + 1):
        fig = plt.figure(fig_num, figsize=(8,6))
        ax = fig.gca()
        ax.set_xlabel('h', fontsize=14)
        ax.set_ylabel('Error', fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        figures[fig_num], axes[fig_num] = fig, ax
    return figures, axes

def plot_line(ax, h: list, y: list, color: str, linestyle: str='-', marker: str='o', label: str=None):
    ax.loglog(h, y, linestyle=linestyle, marker=marker, color=color, label=label)

def finalize_figure(ax, title=None, legend_loc='lower right', ncol=2):
    ax.legend(loc=legend_loc, fontsize=11, ncol=ncol, title=title,
              fancybox=True, shadow=True)

def save_figure(fig, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename, bbox_inches='tight', dpi=200)

# ===============================================
# Run
# ===============================================]

def plots_01(order_poly: str):
    # Colors and labels for scalar and vector norms
    scalar_norms = {
        'L2u': ('g', r'$\|u-u_h\|_{L^2}$'),
        'H1u': ('r', r'$\|u-u_h\|_{H^1}$'),
        'L2l': ('b', r'$\|\lambda-\lambda_h\|_{L^2}$'),
        'H1l': ('m', r'$\|\lambda-\lambda_h\|_{H^1}$'),
    }
    vector_norms = {
        'L2s': ('y', r'$\|\mathbf{s}-\mathbf{s}_h\|_{L^2}$'),
        'L2e': ('c', r'$\|\mathbf{e}-\mathbf{e}_h\|_{L^2}$'),
        'L2m': ('k', r'$\|\boldsymbol{\mu}-\boldsymbol{\mu}_h\|_{L^2}$'),
    }
    # Styles per formulation
    line_style = {
        'Equal-Order': dict(linestyle=':', marker='o'),
        'VMS-Primal': dict(linestyle='--', marker='s'),
        'VMS-Dual': dict(linestyle='-.', marker='^'),
    }

    pairs_of_forms = [['Equal-Order', 'VMS-Primal'], ['Equal-Order','VMS-Dual']]
    for forms in pairs_of_forms:
        # Initialize two figures: 1 for scalars, 2 for vectors
        figures, axes = initialize_figures(num_plots=2)

        for form in forms:
            folder = os.path.join(ct.RESULTS_CONVERGENCE_COMPARISON, form, order_poly)
            file_list = [f for f in os.listdir(folder) if f.endswith('.npy')]
            file_name = file_list[0]
            npy_file_path = os.path.join(folder, file_name)
            errors, _ = read_and_get_values(npy_file_path)

            # Select style for this formulation
            style = line_style[form]

            # Scalars
            for key, (color, label_base) in scalar_norms.items():
                plot_line(
                    axes[1], h=errors['h'], y=errors[key],
                    color=color, linestyle=style['linestyle'], marker=style['marker'], label=label_base
                )

            # Vectors
            for key, (color, label_base) in vector_norms.items():
                plot_line(
                    axes[2], h=errors['h'], y=errors[key],
                    color=color, linestyle=style['linestyle'], marker=style['marker'],label=label_base
                )

        # Finalize and save figures
        finalize_figure(axes[1], title=f"{forms[0]} -- | -- {forms[1]} ", legend_loc='lower right', ncol=2)
        finalize_figure(axes[2], title=f"{forms[0]} -- | -- {forms[1]} ", legend_loc='lower right', ncol=2)

        save_figure(figures[1], os.path.join(ct.RESULTS_CONVERGENCE_COMPARISON,
                                             ct.FIGURES, order_poly, f"scalars_{order_poly}_{forms[1]}.svg"))
        save_figure(figures[2], os.path.join(ct.RESULTS_CONVERGENCE_COMPARISON,
                                             ct.FIGURES, order_poly, f"vectors_{order_poly}_{forms[1]}.svg"))

        plt.show()

# ===============================================
# Main
# ===============================================
if __name__ == "__main__":
    main_selector = int(sys.argv[1])
    if main_selector == 1:
        plots_01(order_poly='Quadratic')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Enable LaTeX rendering for all text
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

fem_mesh = [25, 50, 100, 200]
# Read reference data for u
data_u_ref = pd.read_csv(f'u_ref_natural.csv')
# Read data for all num_points
data_u = [pd.read_csv(f'u_{mesh}x{mesh}_natural.csv') for mesh in fem_mesh]

# Plotting the data
fig, ax = plt.subplots(figsize=(10, 8))
marker_style = dict(marker='o', markersize=7, markerfacecolor='w')
line_style = dict(linestyle='')
colors = ['b', 'g', 'r', 'c']
# Plot data-driven discrete solutions
for i, mesh in enumerate(fem_mesh):
    x_values = data_u[i]['Points:0'].values
    y_values = data_u[i]['DDCM.Potential'].values
    ax.plot(x_values[::8], y_values[::8], markeredgecolor=colors[i % len(colors)], **marker_style, **line_style, 
            label=rf'$N_x \times N_y = {mesh} \times {mesh}$')
# Plot the reference solution
x_ref = data_u_ref['Points:0'].values
y_ref = data_u_ref['Reference.Potential'].values
ax.plot(x_ref, y_ref, '-k', linewidth=2.0, label='Exact')
ax.set_xlabel(r'Coordinate along diagonal', fontsize=24)
ax.set_ylabel(r'$u_h$', fontsize=24)  # Updated y-label with LaTeX formatting
ax.grid(linestyle='--', linewidth=0.5, alpha=0.7)
ax.minorticks_on()
ax.legend(loc='lower right', fontsize=16, title_fontsize=18, ncol=1, fancybox=True, shadow=True)
ax.tick_params(axis='both', which='major', labelsize=24)
ax.tick_params(axis='both', which='minor', labelsize=24)
plt.tight_layout()
plt.savefig(f'u_natural_plot_over_line.svg', format='svg', bbox_inches='tight')
plt.show()


# Read reference data for s
data_s_ref = pd.read_csv(f's_ref_natural.csv')
# Read data for all num_points
data_s = [pd.read_csv(f's_{mesh}x{mesh}_natural.csv') for mesh in fem_mesh]

# Plotting the data
fig, ax = plt.subplots(figsize=(10, 8))
marker_style = dict(marker='o', markersize=7, markerfacecolor='w')
line_style = dict(linestyle='')
colors = ['b', 'g', 'r', 'c']
# Plot data-driven discrete solutions
for i, mesh in enumerate(fem_mesh):
    x_values = data_u[i]['Points:0'].values
    s_vector = np.column_stack((data_s[i]['DDCM.Flux:0'].values, data_s[i]['DDCM.Flux:1'].values))
    y_values = np.linalg.norm(s_vector, ord=2, axis=1)
    ax.plot(x_values[::8], y_values[::8], markeredgecolor=colors[i % len(colors)], **marker_style, **line_style, 
            label=rf'$N_x \times N_y = {mesh} \times {mesh}$')
# Plot the reference solution
x_ref = data_u_ref['Points:0'].values
s_vector_ref = np.column_stack((data_s_ref['Reference.Flux:0'].values, data_s_ref['Reference.Flux:1'].values))
y_ref = np.linalg.norm(s_vector, ord=2, axis=1)
ax.plot(x_ref, y_ref, '-k', linewidth=2.0, label='Exact')
ax.set_xlabel(r'Coordinate along diagonal', fontsize=24)
ax.set_ylabel(r'$\mathbf{s}_h$', fontsize=24)  # Updated y-label with LaTeX formatting
ax.grid(linestyle='--', linewidth=0.5, alpha=0.7)
ax.minorticks_on()
ax.legend(loc='lower right', fontsize=16, title_fontsize=18, ncol=1, fancybox=True, shadow=True)
ax.tick_params(axis='both', which='major', labelsize=24)
ax.tick_params(axis='both', which='minor', labelsize=24)
plt.tight_layout()
plt.savefig(f's_natural_plot_over_line.svg', format='svg', bbox_inches='tight')
plt.show()

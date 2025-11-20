import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def manufactured_solution(x, sigma=1.0, gamma=10.0, alpha=100.0, r0=0.4, x0=np.array([0.5, 0.5])):
    """
    Evaluates the exact solution and derived terms for the diffusion-reaction problem.
    x: Spatial coordinate (array-like of dimension n)
    Returns: p, u, f, div_u
    """
    x = np.asarray(x)
    x0 = np.asarray(x0)
    r_vec = x - x0
    r = np.linalg.norm(r_vec)
    
    # Avoid division by zero at the center
    if r < 1e-12:
        r_unit = np.zeros_like(r_vec)
    else:
        r_unit = r_vec / r
        
    tanh_val = np.tanh(alpha * (r - r0))
    sech2_val = 1 - tanh_val**2  # sech²(z) = 1 - tanh²(z)
    
    # Exact scalar solution p
    p = 1 + tanh_val
    
    # Exact flux u = -sigma * grad p
    u = -sigma * alpha * sech2_val * r_unit
    
    # Divergence of u (n-dimensional)
    n = len(x)
    term1 = -2 * alpha**2 * tanh_val * sech2_val
    if r > 1e-12:
        term2 = ( (n-1) / r ) * alpha * sech2_val
    else:
        term2 = 0.0 
    laplacian_p = term1 + term2
    div_u = -sigma * laplacian_p
    
    # Forcing function f = div_u + gamma * p
    f = div_u + gamma * p
    
    return p, u, f, div_u

# Create a grid
nx, ny = 50, 50
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

# Parameters
sigma = 1.0
alpha = 20.0
r0 = 0.4
x0 = np.array([0.5, 0.5])

# Initialize arrays for p and u
P = np.zeros((ny, nx))
U_x = np.zeros((ny, nx))
U_y = np.zeros((ny, nx))
DIV_U = np.zeros((ny, nx))

# Calculate fields
for i in range(ny):
    for j in range(nx):
        p_val, u_val, f_val, div_u_val = manufactured_solution([X[i,j], Y[i,j]], 
                                                              sigma=sigma, alpha=alpha, r0=r0, x0=x0)
        P[i,j] = p_val
        U_x[i,j] = u_val[0]
        U_y[i,j] = u_val[1]
        DIV_U[i,j] = div_u_val

# Create the plot
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Scalar field p
im1 = axes[0,0].contourf(X, Y, P, levels=50, cmap='viridis')
axes[0,0].set_title('Scalar Field p(x,y) = 1 + tanh(α(r-r₀))', fontsize=14, fontweight='bold')
axes[0,0].set_xlabel('x')
axes[0,0].set_ylabel('y')
plt.colorbar(im1, ax=axes[0,0], label='p value')

# Add circle showing the transition location
circle = Circle((x0[0], x0[1]), r0, fill=False, color='red', linestyle='--', linewidth=2, label=f'r₀ = {r0}')
axes[0,0].add_patch(circle)
axes[0,0].legend()

# Plot 2: Flux magnitude |u|
flux_magnitude = np.sqrt(U_x**2 + U_y**2)
im2 = axes[0,1].contourf(X, Y, flux_magnitude, levels=50, cmap='plasma')
axes[0,1].set_title('Flux Magnitude |u(x,y)|', fontsize=14, fontweight='bold')
axes[0,1].set_xlabel('x')
axes[0,1].set_ylabel('y')
plt.colorbar(im2, ax=axes[0,1], label='|u|')

# Add the same circle for reference
circle2 = Circle((x0[0], x0[1]), r0, fill=False, color='red', linestyle='--', linewidth=2)
axes[0,1].add_patch(circle2)

# Plot 3: Vector field u (with subsampling for clarity)
subsample = 4
X_sub = X[::subsample, ::subsample]
Y_sub = Y[::subsample, ::subsample]
U_x_sub = U_x[::subsample, ::subsample]
U_y_sub = U_y[::subsample, ::subsample]

axes[1,0].quiver(X_sub, Y_sub, U_x_sub, U_y_sub, scale=50, color='blue', alpha=0.7)
axes[1,0].set_title('Flux Vector Field u(x,y)', fontsize=14, fontweight='bold')
axes[1,0].set_xlabel('x')
axes[1,0].set_ylabel('y')
axes[1,0].set_aspect('equal')

# Add circle and center point
circle3 = Circle((x0[0], x0[1]), r0, fill=False, color='red', linestyle='--', linewidth=2)
axes[1,0].add_patch(circle3)
axes[1,0].plot(x0[0], x0[1], 'ro', markersize=8, label='Center x₀')
axes[1,0].legend()

# Plot 4: Divergence of u
im4 = axes[1,1].contourf(X, Y, DIV_U, levels=50, cmap='RdBu_r')
axes[1,1].set_title('Divergence ∇·u(x,y)', fontsize=14, fontweight='bold')
axes[1,1].set_xlabel('x')
axes[1,1].set_ylabel('y')
plt.colorbar(im4, ax=axes[1,1], label='∇·u')

# Add circle for reference
circle4 = Circle((x0[0], x0[1]), r0, fill=False, color='red', linestyle='--', linewidth=2)
axes[1,1].add_patch(circle4)

plt.tight_layout()
plt.show()

# Print some statistics
print(f"Field Statistics:")
print(f"p: min = {P.min():.3f}, max = {P.max():.3f}, range = {P.max()-P.min():.3f}")
print(f"|u|: min = {flux_magnitude.min():.3f}, max = {flux_magnitude.max():.3f}")
print(f"∇·u: min = {DIV_U.min():.3f}, max = {DIV_U.max():.3f}")
print(f"\nParameters: α = {alpha}, r₀ = {r0}, σ = {sigma}")
